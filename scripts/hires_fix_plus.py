import gradio as gr
import math, tomesd
from modules import scripts, script_callbacks, shared, sd_samplers_common, sd_samplers
from modules.ui_components import FormRow
from modules.images import resize_image
from modules.processing import process_images, StableDiffusionProcessingTxt2Img, StableDiffusionProcessingImg2Img

HFP_ON = False
latent_upscaler = ["Latent", "Latent (antialiased)", "Latent (bicubic)", "Latent (bicubic antialiased)", "Latent (nearest)", "Latent (nearest-exact)"]
hfp_samplers = [sd_samplers_common.SamplerData(name='Use main sampler', constructor=None, aliases=None, options=None)] + sd_samplers.samplers_for_img2img

def gr_show(visible=True, n=1):
    if n > 1:
        return [{"visible": visible, "__type__": "update"}] * n
    return {"visible": visible, "__type__": "update"}

def txt2img_to_img2img(p:StableDiffusionProcessingTxt2Img) -> StableDiffusionProcessingImg2Img:
    KNOWN_KEYS = [
        'sd_model',
        'outpath_samples',
        'outpath_grids',
        'prompt',
        'negative_prompt',
        'styles',
        'seed',
        'subseed',
        'subseed_strength',
        'seed_resize_from_h',
        'seed_resize_from_w',
        'seed_enable_extras',
        'sampler_name',
        'batch_size',
        'n_iter',
        'steps',
        'cfg_scale',
        'width',
        'height',
        'restore_faces',
        'tiling',
        'do_not_save_samples',
        'do_not_save_grid',
        'extra_generation_params',
        'overlay_images',
        'negative_prompt',
        'eta',
        'do_not_reload_embeddings',
        'denoising_strength',
        'ddim_discretize',
        's_churn',
        's_tmax',
        's_tmin',
        's_noise',
        'override_settings',
        'override_settings_restore_afterwards',
        'sampler_index',
        'script_args',
    ]
    kwargs = { k: getattr(p, k) for k in dir(p) if k in KNOWN_KEYS } 
    return StableDiffusionProcessingImg2Img(**kwargs)

def get_steps(p,enable_hfp_smartstep,hfp_smartstep_min):
    steps = p.hr_second_pass_steps or int(min(p.denoising_strength, 0.999) * p.steps)
    if enable_hfp_smartstep:
        steps = p.hr_second_pass_steps or max(int(hfp_smartstep_min), round(math.log(10,p.steps)*p.steps*p.denoising_strength))

    return steps

class ToMe:

    def load(sd_model, ratio):
        print(' Loading tomesd...  ')
        tomesd.apply_patch(sd_model, ratio=ratio, max_downsample=1, sx=2, sy=2, use_rand=False, merge_attn=True, merge_crossattn=False, merge_mlp=False)
            
    def unload(sd_model):
        print(' Unloading tomesd...  ')
        tomesd.remove_patch(sd_model)

class Script(scripts.Script):

    def title(self):
        return 'Hires.fix Plus'

    def describe(self):
        return "Add more optional parameters to Hires. fix"

    def show(self, is_img2img):
        if not is_img2img:
            return scripts.AlwaysVisible

    def ui(self, is_img2img):
        global hfp_samplers
        with FormRow(elem_classes="checkboxes-row", variant="compact"):
            enable_hrplus = gr.Checkbox(label='Enable Hires. fix+', value=False, elem_id="hfp_enable_hfp_add")
            enable_hfp_smartstep = gr.Checkbox(visible=False, label='Use Smart-Steps', value=False, elem_id="hfp_enable_hfp_smartstep")
            enable_hfp_tome = gr.Checkbox(visible=False, label='ToMe for Hires. fix', value=False, elem_id="hfp_enable_hfp_tome", interactive=True)
            add_prompts = gr.Checkbox(visible=False, label='Append prompts, not replace', value=False, elem_id="hfp_add_prompts", interactive=True)

        with gr.Group(visible=False, elem_id="hfp_add") as hfp_options:
            with FormRow(elem_id="hfp_row_add", variant="compact"):
                hfp_sampler_index = gr.Dropdown(label='Hires Sampling method', elem_id="hfp_sampling", choices=[x.name for x in hfp_samplers], value=hfp_samplers[0].name, type="index", interactive=True)
                hfp_cfg = gr.Slider(minimum=0.0, maximum=30.0, step=0.5, label='Hires CFG', value=0.0, elem_id="txt2img_hfp_cfg")
                hfp_smartstep_min = gr.Slider(visible=False, minimum=1, maximum=30, step=1, label='Minimum steps', value=9, elem_id="txt2img_hfp_smartstep_min", interactive=True)
                hfp_tome_ratio = gr.Slider(visible=False, minimum=0.1, maximum=0.9, step=0.1, label='ToMe Merging Ratio', value=0.5, elem_id="txt2img_hfp_tome_ratio", interactive=True)

            with FormRow(elem_id="txt2img_hires_fix_prompt_row", variant="compact"):
                alter_prompt = gr.Textbox(label="Prompt", elem_id=f"hfp_prompt", show_label=False, lines=2, placeholder="Prompt (press Ctrl+Enter or Alt+Enter to generate)")
                alter_prompt_n = gr.Textbox(label="Negative prompt", elem_id=f"hfp_neg_prompt", show_label=False, lines=2, placeholder="Negative prompt (press Ctrl+Enter or Alt+Enter to generate)")

        enable_hrplus.change(
            fn=lambda x: gr_show(x,n=4),
            inputs=[enable_hrplus],
            outputs=[hfp_options, enable_hfp_smartstep, enable_hfp_tome,add_prompts],
            show_progress = False,
        )

        enable_hfp_smartstep.change(
            fn=lambda x: gr_show(x),
            inputs=[enable_hfp_smartstep],
            outputs=[hfp_smartstep_min],
            show_progress = False,
        )
        enable_hfp_tome.change(
            fn=lambda x: gr_show(x),
            inputs=[enable_hfp_tome],
            outputs=[hfp_tome_ratio],
            show_progress = False,
        )
        
        self.infotext_fields = [
            (enable_hrplus, "Hires plus"),
            (enable_hfp_smartstep, "Smart step"),
            (enable_hfp_tome, "Hires ToMe"),
            (hfp_sampler_index, "Hires sampler"),
            (hfp_cfg, "Hires CFG"),
            (hfp_smartstep_min, "SmartStep min"),
            (hfp_tome_ratio, "ToMe ratio"),
            (add_prompts, "Add prompts"),
            (alter_prompt, "Hires prompts"),
            (alter_prompt_n, "Hires N_prompts")
        ]
        
        self.paste_field_names = [
            (enable_hrplus, "Hires plus"),
            (enable_hfp_smartstep, "Smart step"),
            (enable_hfp_tome, "Hires ToMe"),
            (hfp_sampler_index, "Hires sampler"),
            (hfp_cfg, "Hires CFG"),
            (hfp_smartstep_min, "SmartStep min"),
            (hfp_tome_ratio, "ToMe ratio"),
            (add_prompts, "Add prompts"),
            (alter_prompt, "Hires prompts"),
            (alter_prompt_n, "Hires N_prompts")
        ]

        return [enable_hrplus, enable_hfp_smartstep, enable_hfp_tome, add_prompts, hfp_cfg, hfp_sampler_index, alter_prompt, alter_prompt_n, hfp_smartstep_min, hfp_tome_ratio]

    def process_batch(self, p:StableDiffusionProcessingTxt2Img, enable_hrplus:bool, enable_hfp_smartstep:bool, enable_hfp_tome:bool, add_prompts:bool, hfp_cfg:float, hfp_sampler_index:int, alter_prompt:str, alter_prompt_n:str, hfp_smartstep_min:int, hfp_tome_ratio:float, batch_number, prompts, seeds,  subseeds):
        global HFP_ON, hfp_samplers
        
        if p.enable_hr and (enable_hrplus or getattr(p, 'enable_hrplus', False)):
            if p.hr_resize_x == 0 and p.hr_resize_y == 0 and p.hr_scale == 1:
                return
            print(' Hijacking Hires. fix...  ')
            HFP_ON = True
            p.enable_hr = False
        else:
            HFP_ON = False
            if enable_hrplus:
                print(' Please enable Hires. fix first!  ')
            return

        b_count = 0
        p.do_not_save_samples = not shared.opts.save_images_before_highres_fix
        p.seed = p.seed + batch_number
        
        if enable_hrplus:
            p.extra_generation_params["Hires plus"] = enable_hrplus
            
        if enable_hfp_smartstep:
            p.extra_generation_params["Smart step"] = enable_hfp_smartstep
        
        if enable_hfp_tome:
            p.extra_generation_params["Hires ToMe"] = enable_hfp_tome

        sampler = hfp_samplers[hfp_sampler_index].name
        if hfp_sampler_index != 0 and sampler != p.sampler_name:
            p.extra_generation_params["Hires sampler"] = hfp_samplers[hfp_sampler_index].name
            
        if hfp_cfg:
            p.extra_generation_params["Hires CFG"] = hfp_cfg

        if hfp_smartstep_min != 9:
            p.extra_generation_params["SmartStep min"] = hfp_smartstep_min
            
        if hfp_tome_ratio != 0.5:
            p.extra_generation_params["ToMe ratio"] = hfp_tome_ratio

        if p.hr_resize_x == 0 and p.hr_resize_y == 0:
            p.extra_generation_params["Hires upscale"] = p.hr_scale
        else:
            p.extra_generation_params["Hires resize"] = f"{p.hr_resize_x}x{p.hr_resize_y}"
            
        #steps = get_steps(p,enable_hfp_smartstep,hfp_smartstep_min)
        #if p.steps != steps:
            #p.extra_generation_params["Hires steps"] = steps

        if p.hr_second_pass_steps:
            p.extra_generation_params["Hires steps"] = p.hr_second_pass_steps
            
        if p.hr_upscaler is not None:
            p.extra_generation_params["Hires upscaler"] = p.hr_upscaler
            
        if alter_prompt != "" or alter_prompt_n != "":
            p.extra_generation_params["Add prompts"] = add_prompts
            
        if alter_prompt != "":
            p.extra_generation_params["Hires prompts"] = alter_prompt

        if alter_prompt_n != "":
            p.extra_generation_params["Hires N_prompts"] = alter_prompt_n

        shared.total_tqdm.updateTotal(p.steps)

    def postprocess_image(self, p, pp, enable_hrplus:bool, enable_hfp_smartstep:bool, enable_hfp_tome:bool, add_prompts:bool, hfp_cfg:float, hfp_sampler_index:int, alter_prompt:str, alter_prompt_n:str, hfp_smartstep_min:int = 9, hfp_tome_ratio:float = 0.5):
        global HFP_ON, hfp_samplers
        
        if not HFP_ON:
            return
        p.enable_hr = True
        
        from modules import devices
        pt = StableDiffusionProcessingTxt2Img(width=p.width, height=p.height, enable_hr=True, hr_scale=p.hr_scale, hr_resize_x=p.hr_resize_x, hr_resize_y=p.hr_resize_y)
        with devices.autocast():
            pt.init([""], [0], [0])
        target_width = pt.hr_resize_x or pt.hr_upscale_to_x
        target_height = pt.hr_resize_y or pt.hr_upscale_to_y

        shared.opts.img2img_fix_steps = True
        step_opt = shared.opts.img2img_fix_steps
        steps = get_steps(p,enable_hfp_smartstep,hfp_smartstep_min)
        shared.total_tqdm.updateTotal(p.steps + steps)

        p2:StableDiffusionProcessingImg2Img = txt2img_to_img2img(p)

        if p.hr_upscaler in latent_upscaler:
            img = pp.image
            p2.resize_mode = 3
        else:
            img = resize_image(0, pp.image, target_width, target_height, upscaler_name=p.hr_upscaler)
            p2.resize_mode = 0
            
        if add_prompts:
            p2.prompt = p2.prompt.rstrip(',')+', '+alter_prompt
            p2.negative_prompt = p2.negative_prompt.rstrip(',')+', '+alter_prompt_n
        else:
            p2.prompt = alter_prompt if alter_prompt != "" else p2.prompt
            p2.negative_prompt = alter_prompt_n if alter_prompt_n != "" else p2.negative_prompt

        p2.init_images = [img]
        p2.steps = steps
        p2.n_iter = 1
        p2.batch_size = 1
        p2.do_not_save_samples = False
        p2.cfg_scale = (hfp_cfg or getattr(p, 'hfp_cfg', 0)) or p.cfg_scale
        p2.width  = target_width
        p2.height = target_height
        p2.sampler_name = getattr(p, "hfp_sampler_index", None) or hfp_samplers[hfp_sampler_index].name if hfp_sampler_index != 0 else p.sampler_name
        if p.sampler_name in ['PLMS', 'UniPC']:  # PLMS/UniPC do not support img2img so we just silently switch to DDIM
            p2.sampler_name = 'DDIM'
        p2.seed = p.seed
        p2.subseed = p.subseed
        p2.subseed_strength = p.subseed_strength
        
        p2.extra_generation_params["Steps"] = p.steps
        p2.extra_generation_params["Sampler"] = p.sampler_name
        p2.extra_generation_params["CFG scale"] = p.cfg_scale
        p2.extra_generation_params["Size"] = f"{p.width}x{p.height}"
        
        if enable_hfp_tome or getattr(p, "enable_hfp_tome", False):
            ToMe.load(p.sd_model, getattr(p, "hfp_tome_ratio", 0) or hfp_tome_ratio)
        proc = process_images(p2)
        if enable_hfp_tome or getattr(p, "enable_hfp_tome", False):
            ToMe.unload(p.sd_model)
            
        imgs = proc.images
        pp.image = imgs[0]
        
        shared.opts.img2img_fix_steps = step_opt
        
def make_axis_options():
    xyz_grid = [x for x in scripts.scripts_data if x.script_class.__module__ == "xyz_grid.py"][0].module
    
    def apply_hires_cfg(p, x, xs):
        setattr(p, "enable_hrplus", True)
        setattr(p, "hfp_cfg", x)
        
    def apply_hires_sampler(p, x, xs):
        hr_sampler = sd_samplers.samplers_map.get(x.lower(), None)
        if hr_sampler is None:
            raise RuntimeError(f"Unknown sampler: {x}")
        setattr(p, "enable_hrplus", True)
        setattr(p, "hfp_sampler_index", hr_sampler)
        
    def apply_tome_ratio(p, x, xs):
        if 0.9 >= x >= 0.1:
            setattr(p, "enable_hrplus", True)
            setattr(p, "enable_hfp_tome", True)
            setattr(p, "hfp_tome_ratio", x)
        else:
            raise RuntimeError(f"Invalid Merging Ratio: {x}")
        
    extra_axis_options = [
        xyz_grid.AxisOptionTxt2Img("[HF+] Hires Sampler", str, apply_hires_sampler, choices=lambda: [x.name for x in sd_samplers.samplers_for_img2img]),
        xyz_grid.AxisOptionTxt2Img("[HF+] Hires CFG", float, apply_hires_cfg),
        xyz_grid.AxisOptionTxt2Img("[HF+] ToMe ratio", float, apply_tome_ratio)
    ]
    if not any("[HF+]" in x.label for x in xyz_grid.axis_options):
        xyz_grid.axis_options.extend(extra_axis_options)
        
def callbackBeforeUi():
    try:
        make_axis_options()
    except Exception as e:
        traceback.print_exc()
        print(f"Failed to add support for X/Y/Z Plot Script because: {e}")
        
script_callbacks.on_before_ui(callbackBeforeUi)