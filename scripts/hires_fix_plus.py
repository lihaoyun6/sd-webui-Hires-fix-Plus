import numpy as np
import gradio as gr
from PIL import Image
import torch, math, re, random
from distutils.version import StrictVersion

import modules.images as images
import modules.sd_models as sd_models
from modules import scripts, script_callbacks, shared, sd_samplers, devices, extra_networks

from modules.shared import opts
from modules.processing import program_version
from modules.processing import StableDiffusionProcessingTxt2Img, create_random_tensors, opt_C, opt_f, decode_first_stage, get_fixed_seed

suppver = "1.3.0"
version = re.search("v[\d\.]*", program_version())[0].replace('v','')
low = StrictVersion(version) < StrictVersion(suppver)
sample_org = StableDiffusionProcessingTxt2Img.sample

def sample(self, conditioning, unconditional_conditioning, seeds, subseeds, subseed_strength, prompts):
	#print("Running custom sample function...  ")
	self.sampler = sd_samplers.create_sampler(self.sampler_name, self.sd_model)
	
	latent_scale_mode = shared.latent_upscale_modes.get(self.hr_upscaler, None) if self.hr_upscaler is not None else shared.latent_upscale_modes.get(shared.latent_upscale_default_mode, "nearest")
	if self.enable_hr and latent_scale_mode is None:
		assert len([x for x in shared.sd_upscalers if x.name == self.hr_upscaler]) > 0, f"could not find upscaler named {self.hr_upscaler}"
		
	x = create_random_tensors([opt_C, self.height // opt_f, self.width // opt_f], seeds=seeds, subseeds=subseeds, subseed_strength=self.subseed_strength, seed_resize_from_h=self.seed_resize_from_h, seed_resize_from_w=self.seed_resize_from_w, p=self)
	samples = self.sampler.sample(self, x, conditioning, unconditional_conditioning, image_conditioning=self.txt2img_image_conditioning(x))
	
	if not self.enable_hr:
		return samples
	
	self.is_hr_pass = True
	
	target_width = self.hr_upscale_to_x
	target_height = self.hr_upscale_to_y

	rolling_factor = getattr(self, 'hfp_rolling_factor', 0)

	def save_intermediate(image, index):
		"""saves image before applying hires fix, if enabled in options; takes as an argument either an image or batch with latent space images"""
		
		if not opts.save or self.do_not_save_samples or not opts.save_images_before_highres_fix:
			return
		
		if not isinstance(image, Image.Image):
			image = sd_samplers.sample_to_image(image, index, approximation=0)
			
		info = create_infotext(self, self.all_prompts, self.all_seeds, self.all_subseeds, [], iteration=self.iteration, position_in_batch=index)
		images.save_image(image, self.outpath_samples, "", seeds[index], prompts[index], opts.samples_format, info=info, suffix="-before-highres-fix")
	
	if rolling_factor != 1 and rolling_factor < self.hr_upscale_to_x/self.width:

		rounds = math.ceil(math.log(self.hr_upscale_to_x/self.width)/math.log(rolling_factor))

		shared.state.job_count = rounds
		shared.total_tqdm.updateTotal(self.steps+get_steps(self) * (rounds - 1))
		
		for t in range(1, rounds):
			print(f"Generation round {t}/{rounds - 1}  ")
			target_width = int(self.width * math.pow(rolling_factor,t))
			target_height = int(self.height * math.pow(rolling_factor,t))
			seeds = list(map(lambda x: x + opts.data.get("hfp_jitter_step", 1), seeds)) if opts.data.get("hfp_jitter_seeds", False) else seeds
			seeds = [get_fixed_seed(-1)] * len(seeds) if opts.data.get("hfp_random_seeds", False) else seeds

			if t == rounds-1:
				target_width = self.hr_upscale_to_x
				target_height = self.hr_upscale_to_y

			if latent_scale_mode is not None:
				for i in range(samples.shape[0]):
					save_intermediate(samples, i)
					
				samples = torch.nn.functional.interpolate(samples, size=(target_height // opt_f, target_width // opt_f), mode=latent_scale_mode["mode"], antialias=latent_scale_mode["antialias"])
				
				# Avoid making the inpainting conditioning unless necessary as
				# this does need some extra compute to decode / encode the image again.
				if getattr(self, "inpainting_mask_weight", shared.opts.inpainting_mask_weight) < 1.0:
					image_conditioning = self.img2img_image_conditioning(decode_first_stage(self.sd_model, samples), samples)
				else:
					image_conditioning = self.txt2img_image_conditioning(samples)
			else:
				decoded_samples = decode_first_stage(self.sd_model, samples)
				lowres_samples = torch.clamp((decoded_samples + 1.0) / 2.0, min=0.0, max=1.0)
				
				batch_images = []
				for i, x_sample in enumerate(lowres_samples):
					x_sample = 255. * np.moveaxis(x_sample.cpu().numpy(), 0, 2)
					x_sample = x_sample.astype(np.uint8)
					image = Image.fromarray(x_sample)
					
					save_intermediate(image, i)
					
					image = images.resize_image(0, image, target_width, target_height, upscaler_name=self.hr_upscaler)
					image = np.array(image).astype(np.float32) / 255.0
					image = np.moveaxis(image, 2, 0)
					batch_images.append(image)
					
				decoded_samples = torch.from_numpy(np.array(batch_images))
				decoded_samples = decoded_samples.to(shared.device)
				decoded_samples = 2. * decoded_samples - 1.
				
				samples = self.sd_model.get_first_stage_encoding(self.sd_model.encode_first_stage(decoded_samples))
				
				image_conditioning = self.img2img_image_conditioning(decoded_samples, samples)
				
			shared.state.nextjob()
			
			img2img_sampler_name = self.hr_sampler_name or self.sampler_name
			
			if self.sampler_name in ['PLMS', 'UniPC']:  # PLMS/UniPC do not support img2img so we just silently switch to DDIM
				img2img_sampler_name = 'DDIM'
			
			self.sampler = sd_samplers.create_sampler(img2img_sampler_name, self.sd_model)
			
			samples = samples[:, :, self.truncate_y//2:samples.shape[2]-(self.truncate_y+1)//2, self.truncate_x//2:samples.shape[3]-(self.truncate_x+1)//2]

			noise = create_random_tensors(samples.shape[1:], seeds=seeds, subseeds=subseeds, subseed_strength=subseed_strength, p=self)
			
			# GC now before running the next img2img to prevent running out of memory
			x = None
			devices.torch_gc()
			
			if not self.disable_extra_networks:
				with devices.autocast():
					extra_networks.activate(self, self.hr_extra_network_data)
					
			sd_models.apply_token_merging(self.sd_model, self.get_token_merging_ratio(for_hr=True))
		
			cfg = self.cfg_scale
			self.cfg_scale = getattr(self, 'hfp_cfg', 0) or self.cfg_scale

			samples = self.sampler.sample_img2img(self, samples, noise, self.hr_c, self.hr_uc, steps=self.hr_second_pass_steps or self.steps, image_conditioning=image_conditioning)

			self.cfg_scale = cfg
		
			sd_models.apply_token_merging(self.sd_model, self.get_token_merging_ratio())
	else:
		if latent_scale_mode is not None:
			for i in range(samples.shape[0]):
				save_intermediate(samples, i)
				
			samples = torch.nn.functional.interpolate(samples, size=(target_height // opt_f, target_width // opt_f), mode=latent_scale_mode["mode"], antialias=latent_scale_mode["antialias"])
			
			# Avoid making the inpainting conditioning unless necessary as
			# this does need some extra compute to decode / encode the image again.
			if getattr(self, "inpainting_mask_weight", shared.opts.inpainting_mask_weight) < 1.0:
				image_conditioning = self.img2img_image_conditioning(decode_first_stage(self.sd_model, samples), samples)
			else:
				image_conditioning = self.txt2img_image_conditioning(samples)
		else:
			decoded_samples = decode_first_stage(self.sd_model, samples)
			lowres_samples = torch.clamp((decoded_samples + 1.0) / 2.0, min=0.0, max=1.0)
			
			batch_images = []
			for i, x_sample in enumerate(lowres_samples):
				x_sample = 255. * np.moveaxis(x_sample.cpu().numpy(), 0, 2)
				x_sample = x_sample.astype(np.uint8)
				image = Image.fromarray(x_sample)
				
				save_intermediate(image, i)
				
				image = images.resize_image(0, image, target_width, target_height, upscaler_name=self.hr_upscaler)
				image = np.array(image).astype(np.float32) / 255.0
				image = np.moveaxis(image, 2, 0)
				batch_images.append(image)
				
			decoded_samples = torch.from_numpy(np.array(batch_images))
			decoded_samples = decoded_samples.to(shared.device)
			decoded_samples = 2. * decoded_samples - 1.
			
			samples = self.sd_model.get_first_stage_encoding(self.sd_model.encode_first_stage(decoded_samples))
			
			image_conditioning = self.img2img_image_conditioning(decoded_samples, samples)
			
		shared.state.nextjob()
		
		img2img_sampler_name = self.hr_sampler_name or self.sampler_name
		
		if self.sampler_name in ['PLMS', 'UniPC']:  # PLMS/UniPC do not support img2img so we just silently switch to DDIM
			img2img_sampler_name = 'DDIM'
		
		self.sampler = sd_samplers.create_sampler(img2img_sampler_name, self.sd_model)
		
		samples = samples[:, :, self.truncate_y//2:samples.shape[2]-(self.truncate_y+1)//2, self.truncate_x//2:samples.shape[3]-(self.truncate_x+1)//2]
		
		noise = create_random_tensors(samples.shape[1:], seeds=seeds, subseeds=subseeds, subseed_strength=subseed_strength, p=self)
		
		# GC now before running the next img2img to prevent running out of memory
		x = None
		devices.torch_gc()
		
		if not self.disable_extra_networks:
			with devices.autocast():
				extra_networks.activate(self, self.hr_extra_network_data)
				
		sd_models.apply_token_merging(self.sd_model, self.get_token_merging_ratio(for_hr=True))
		
		cfg = self.cfg_scale
		self.cfg_scale = getattr(self, 'hfp_cfg', 0) or self.cfg_scale

		samples = self.sampler.sample_img2img(self, samples, noise, self.hr_c, self.hr_uc, steps=self.hr_second_pass_steps or self.steps, image_conditioning=image_conditioning)

		self.cfg_scale = cfg
		
		sd_models.apply_token_merging(self.sd_model, self.get_token_merging_ratio())
		
	self.is_hr_pass = False		
	return samples

def gr_show(visible=True, n=1):
	if n > 1:
		return [{"visible": visible, "__type__": "update"}] * n
	return {"visible": visible, "__type__": "update"}

def get_steps(p):
	log_steps = max(opts.data.get("hfp_smartstep_min", 9), round(math.log(10,p.steps)*p.steps*p.denoising_strength))
	steps = p.hr_second_pass_steps if p.hr_second_pass_steps !=0 else log_steps
	return steps

class HiresFixPlus(scripts.Script):
	def title(self):
		return 'Hires.fix Progressive'

	def describe(self):
		return "A progressive version of hires.fix implementation."

	def show(self, is_img2img):
		if not is_img2img:
			return scripts.AlwaysVisible

	def after_component(self, component, **kwargs):
		if low:
			if kwargs.get("elem_id") == f"txt2img_enable_hr":                                   
				self.warring_text = gr.HTML(value=f'Hires.fix+ requires WebUI v{suppver} or later<br>But you have {program_version()}, please update it.', elem_id="hfp_warring_text")
		else:
			if kwargs.get("elem_id") == f"txt2img_enable_hr":
				self.warring_text = gr.HTML(value='Set "Hires steps" to [0], if you need<br>Hires. fix+ to do steps optimization', elem_id="hfp_warring_text")
			if kwargs.get("elem_id") == f"txt2img_denoising_strength":
				self.hfp_cfg = gr.Slider(minimum=0.0, maximum=30.0, step=0.5, label='Hires CFG', value=0.0, elem_id="txt2img_hfp_cfg", interactive=True)
			if kwargs.get("elem_id") == f"txt2img_hr_resize_y":
				self.hfp_rolling_factor = gr.Slider(minimum=1.0, maximum=2.0, step=0.05, label='Rolling factor', value=1.0, elem_id="txt2img_hfp_rolling_factor", interactive=True)

	def ui(self, is_img2img):
		if not low:
			self.infotext_fields = [
				(self.hfp_cfg, "Hires CFG"),
				(self.hfp_rolling_factor, "Rolling factor")
			]
			
			self.paste_field_names = [
				(self.hfp_cfg, "Hires CFG"),
				(self.hfp_rolling_factor, "Rolling factor")
			]
	
			return [self.hfp_cfg, self.hfp_rolling_factor]

	def process(self, p, hfp_cfg:int = 0, hfp_rolling_factor:float = 1.0):
		if not low and p.enable_hr:
			print('Hijacking Hires. fix...  ')
			StableDiffusionProcessingTxt2Img.sample = sample
			self.hr_step = p.hr_second_pass_steps
			p.hr_second_pass_steps = get_steps(p)
			hires_cfg = (getattr(p, 'hfp_cfg', 0) or hfp_cfg) or p.cfg_scale
			setattr(p, "hfp_cfg", hires_cfg)
			setattr(p, "hfp_rolling_factor", hfp_rolling_factor)
			if hires_cfg != p.cfg_scale:
				p.extra_generation_params["Hires CFG"] = hfp_cfg
			if hfp_rolling_factor != 1:
				p.extra_generation_params["Rolling factor"] = hfp_rolling_factor

	def process_batch(self, p, *args, **kwargs):
		if not low and p.enable_hr:
			p.extra_generation_params["Hires steps"] = self.hr_step if self.hr_step != 0 else None

	def postprocess(self, p, processed, *args):
		if not low and p.enable_hr:
			StableDiffusionProcessingTxt2Img.sample = sample_org

def create_script_items():
	try:
		xyz_grid = [x for x in scripts.scripts_data if x.script_class.__module__ == "xyz_grid.py"][0].module
		
		def apply_hires_cfg(p, x, xs):
			setattr(p, "hfp_cfg", x)
		
		def apply_hires_sampler(p, x, xs):
			hr_sampler = sd_samplers.samplers_map.get(x.lower(), None)
			if hr_sampler is None:
				raise RuntimeError(f"Unknown sampler: {x}  ")
			setattr(p, "hr_sampler_name", hr_sampler)
		
		extra_axis_options = [
			xyz_grid.AxisOptionTxt2Img("Hires Sampler", str, apply_hires_sampler, choices=lambda: [x.name for x in sd_samplers.samplers_for_img2img]),
			xyz_grid.AxisOptionTxt2Img("Hires CFG", float, apply_hires_cfg)
		]
		if not any("[HF+]" in x.label for x in xyz_grid.axis_options):
			xyz_grid.axis_options.extend(extra_axis_options)
	except Exception as e:
		traceback.print_exc()
		print(f"Failed to add support for X/Y/Z Plot Script because: {e}  ")

def create_settings_items():
	section_hfp = ('hiresfix_plus', 'Hires. fix+')
	opts.add_option("hfp_smartstep_min", shared.OptionInfo(
		9, "If Smart-Step is enabled, the number of iterations for Hires. fix will never be less than this:",
		gr.Slider, {"minimum": 1, "maximum": 50, "step": 1}, section=section_hfp
	))
	opts.add_option("hfp_jitter_seeds", shared.OptionInfo(
		False, "Jitter the seeds of sub-generations when doing a rolling generation (Still deterministic)", section=section_hfp
	))
	opts.add_option("hfp_jitter_step", shared.OptionInfo(
		1, "Jitter step:",
		gr.Slider, {"minimum": 1, "maximum": 100, "step": 1}, section=section_hfp
	))
	opts.add_option("hfp_random_seeds", shared.OptionInfo(
		False, "Use random seeds for sub-generations when doing a rolling generation (WARNING!!! The result will be non-deterministic!!!)", section=section_hfp
	))

if low:
	print(f'Hires.fix+ requires WebUI v{suppver} or later. But you have {program_version()}, please update it.  ')
else:
	scripts.script_callbacks.on_ui_settings(create_settings_items)
	script_callbacks.on_before_ui(create_script_items)
