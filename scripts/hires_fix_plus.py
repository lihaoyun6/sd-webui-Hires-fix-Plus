import gradio as gr
import math, tomesd, os, re

from distutils.version import StrictVersion

from modules.shared import opts
from modules.processing import program_version
from modules.paths_internal import modules_path

from modules import scripts, script_callbacks, shared, sd_samplers

extf, ext = os.path.split(scripts.basedir())
version = re.search("v[\d\.]*", program_version())[0].replace('v','')
low = StrictVersion(version) < StrictVersion('1.3.0')

prefix = '!!!000'
procpy = os.path.join(modules_path, 'processing.py')
org = '        samples = self.sampler.sample_img2img(self, samples, noise, self.hr_c, self.hr_uc, steps=self.hr_second_pass_steps or self.steps, image_conditioning=image_conditioning)'
hack = '        cfg = self.cfg_scale; self.cfg_scale = opts.hfp_hires_cfg or cfg; samples = self.sampler.sample_img2img(self, samples, noise, self.hr_c, self.hr_uc, steps=self.hr_second_pass_steps or self.steps, image_conditioning=image_conditioning); self.cfg_scale = cfg'

def gr_show(visible=True, n=1):
	if n > 1:
		return [{"visible": visible, "__type__": "update"}] * n
	return {"visible": visible, "__type__": "update"}

def get_force():
	if ext.startswith(prefix):
		return True
	return False

def check_hack():
	global org, hack
	with open(procpy,'r') as f:
		if org in f.read():
			return True
	return False

def get_hack():
	global org, hack
	with open(procpy,'r') as f:
		if hack in f.read():
			return True
	return False

def hack_cfg():
	global org, hack
	if get_hack():
		print('Restoring the pipeline...  ')
		org, hack = hack, org
	else:
		print('Modifying pipeline...  ')
	if not check_hack():
		print('File content does not match, cannot hijack!  ')
		return
	file_data = ""
	with open(procpy, "r", encoding="utf-8") as f:
		file_data = f.read().replace(org,hack)
	with open(procpy,"w",encoding="utf-8") as f:
		f.write(file_data)
	print('Done! Please shutdown and re-run the WebUI.  ')

def make_first():
	if get_force():
		os.rename(os.path.join(extf, ext), os.path.join(extf, ext.replace(prefix, '')))
	else:
		os.rename(os.path.join(extf, ext), os.path.join(extf,prefix+ext))
	shared.state.interrupt()
	shared.state.need_restart = True

def get_steps(p,force_hfp_smartstep):
	#def_steps = int(min(p.denoising_strength, 0.999) * p.steps)
	log_steps = max(opts.data.get("hfp_smartstep_min", 10), round(math.log(10,p.steps)*p.steps*p.denoising_strength))
	steps = (p.hr_second_pass_steps or p.steps) if not force_hfp_smartstep else log_steps
	return steps

class HiresFixPlus(scripts.Script):
	def title(self):
		return 'Hires.fix Progressive'

	def describe(self):
		return "A progressive version of hires.fix implementation."

	def show(self, is_img2img):
		if low:
			print(f'[Hires. fix+] This extension only supports webui v1.3.0 or later! You are using {program_version()}  ')
			return False
		if not is_img2img:
			return scripts.AlwaysVisible

	def before_component(self, component, **kwargs):
		if kwargs.get("elem_id") == f"txt2img_hires_steps":
			self.force_hfp_smartstep = gr.Checkbox(label='Force Smart-Steps', value=True, elem_id="hfp_force_hfp_smartstep", interactive=True)
		if kwargs.get("elem_id") == f"txt2img_hr_scale":
			self.hfp_cfg = gr.Slider(visible=get_hack(), minimum=0.0, maximum=30.0, step=0.5, label='Hires CFG', value=0.0, elem_id="txt2img_hfp_cfg", interactive=True)

	def ui(self, is_img2img):
		with gr.Accordion('Hires. fix+', open=False):
			self.force_first = gr.Checkbox(label='Make Hires. fix+ run before any other extensions (will reload WebUI)', value=get_force(), interactive=True, elem_id="hfp_force_first")
			setattr(self.force_first,"do_not_save_to_config",True)
			self.hack_cfg = gr.Checkbox(label='Hijack WebUI to inject "Hires CFG" (take effect from the next startup)', value=get_hack(), interactive=True, elem_id="hfp_hack_cfg")
			setattr(self.hack_cfg,"do_not_save_to_config",True)

		self.force_first.change(
			fn=make_first,
			_js='restart_reload',
			inputs=[],
			outputs=[],
		)
		self.hack_cfg.change(
			fn=hack_cfg,
			inputs=[],
			outputs=[],
		)

		self.infotext_fields = [
			(self.hfp_cfg, "Hires CFG"),
			(self.force_hfp_smartstep, "SmartSteps")
		]
		
		self.paste_field_names = [
			(self.hfp_cfg, "Hires CFG"),
			(self.force_hfp_smartstep, "SmartSteps")
		]

		return [self.force_hfp_smartstep, self.hfp_cfg]

	def process(self, p, force_hfp_smartstep:bool, hfp_cfg:int):
		print('Hijacking Hires. fix...  ')
		self.hr_step = p.hr_second_pass_steps
		p.hr_second_pass_steps = get_steps(p,force_hfp_smartstep)
		hires_cfg = (getattr(p, 'hfp_cfg', 0) or hfp_cfg) or p.cfg_scale
		opts.hfp_hires_cfg = hires_cfg
		if hires_cfg != p.cfg_scale:
			p.extra_generation_params["Hires CFG"] = hfp_cfg
		if force_hfp_smartstep:
			p.extra_generation_params["SmartSteps"] = force_hfp_smartstep
		
	def process_batch(self, p, *args, **kwargs):
		p.extra_generation_params["Hires steps"] = self.hr_step if self.hr_step != 0 else None

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

script_callbacks.on_before_ui(create_script_items)

def create_settings_items():
	section_hfp = ('hiresfix_plus', 'Hires. fix+')
	opts.add_option("hfp_smartstep_min", shared.OptionInfo(
		9, "If Smart-Step is enabled, the number of iterations for Hires. fix will never be less than this:",
		gr.Slider, {"minimum": 1, "maximum": 50, "step": 1}, section=section_hfp
	))

scripts.script_callbacks.on_ui_settings(create_settings_items)