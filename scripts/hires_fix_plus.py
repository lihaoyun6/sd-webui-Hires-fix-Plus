import gradio as gr
import math, tomesd, os
from modules import scripts, script_callbacks, shared, sd_samplers_common, sd_samplers
from modules.shared import opts
from modules.images import resize_image, save_image
from modules.processing import create_infotext, process_images, StableDiffusionProcessingTxt2Img, StableDiffusionProcessingImg2Img

latent_upscaler = ["Latent", "Latent (antialiased)", "Latent (bicubic)", "Latent (bicubic antialiased)", "Latent (nearest)", "Latent (nearest-exact)"]
hfp_samplers = [sd_samplers_common.SamplerData(name='Use main sampler', constructor=None, aliases=None, options=None)] + sd_samplers.samplers_for_img2img
extf, ext = os.path.split(scripts.basedir())
prefix = '!!!000'

def gr_show2(visible=False, visible2=False, visible3=False):
    return ([{"visible": visible, "__type__": "update"}] * 5) + ([{"visible": visible and visible2, "__type__": "update"}] * 2) + [{"visible": visible and visible3, "__type__": "update"}]

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

def get_force():
    if ext.startswith(prefix):
        return True
    return False

def make_first():
    if get_force():
        os.rename(os.path.join(extf, ext), os.path.join(extf, ext.replace(prefix, '')))
    else:
        os.rename(os.path.join(extf, ext), os.path.join(extf,prefix+ext))
    shared.state.interrupt()
    shared.state.need_restart = True

class ToMe:
    def load(sd_model, ratio: float=0.5):
        print(' Loading tomesd...  ')
        tomesd.apply_patch(
            sd_model,
            ratio=int(ratio),
            max_downsample=float(opts.token_merging_max_downsample),
            sx=int(opts.token_merging_stride_x),
            sy=int(opts.token_merging_stride_y),
            use_rand=bool(opts.token_merging_use_rand),
            merge_attn=bool(opts.token_merging_merge_attn),
            merge_crossattn=bool(opts.token_merging_merge_crossattn),
            merge_mlp=bool(opts.token_merging_merge_mlp)
        )
        
    def unload(sd_model):
        print(' Unloading tomesd...  ')
        tomesd.remove_patch(sd_model)

class HiresFixPlus(scripts.Script):
    def title(self):
        return 'Hires.fix Progressive'

    def describe(self):
        return "A progressive version of hires.fix implementation."

    def show(self, is_img2img):
        if not is_img2img:
            return scripts.AlwaysVisible

    def ui(self, is_img2img):
        HiresFixPlus.force_first = gr.Checkbox(label='Make Hires. fix+ run before any other extensions (will reload WebUI)', value=get_force(), interactive=True, elem_id="hfp_force_first")
        setattr(HiresFixPlus.force_first,"do_not_save_to_config",True)

        HiresFixPlus.force_first.change(
            fn=make_first,
            _js='restart_reload',
            inputs=[],
            outputs=[],
        )

        HiresFixPlus.enable_hrplus.change(
            fn=lambda x, y, z: gr_show2(x, y, z),
            inputs=[HiresFixPlus.enable_hrplus, HiresFixPlus.add_prompts, HiresFixPlus.enable_hfp_smartstep],
            outputs=[HiresFixPlus.enable_hfp_smartstep, HiresFixPlus.enable_hfp_tome, HiresFixPlus.add_prompts, HiresFixPlus.hfp_cfg, HiresFixPlus.hfp_sampler_index,HiresFixPlus.alter_prompt, HiresFixPlus.alter_prompt_n,HiresFixPlus.warring_text],
        )
        
        HiresFixPlus.enable_hfp_smartstep.change(
            fn=lambda x: gr_show(x),
            inputs=[HiresFixPlus.enable_hfp_smartstep],
            outputs=[HiresFixPlus.warring_text],
        )
        
        HiresFixPlus.add_prompts.change(
            fn=lambda x: gr_show(x, n=2),
            inputs=[HiresFixPlus.add_prompts],
            outputs=[HiresFixPlus.alter_prompt, HiresFixPlus.alter_prompt_n],
        )
        
        self.infotext_fields = [
            (HiresFixPlus.enable_hrplus, "Hires plus"),
            (HiresFixPlus.enable_hfp_smartstep, "Smart step"),
            (HiresFixPlus.enable_hfp_tome, "Hires ToMe"),
            (HiresFixPlus.hfp_sampler_index, "Hires sampler"),
            (HiresFixPlus.hfp_cfg, "Hires CFG"),
            #(HiresFixPlus.hfp_smartstep_min, "SmartStep min"),
            #(HiresFixPlus.hfp_tome_ratio, "ToMe ratio"),
            (HiresFixPlus.add_prompts, "Ext prompts"),
            (HiresFixPlus.alter_prompt, "Hires prompts"),
            (HiresFixPlus.alter_prompt_n, "Hires n_prompts")
        ]
        
        self.paste_field_names = [
            (HiresFixPlus.enable_hrplus, "Hires plus"),
            (HiresFixPlus.enable_hfp_smartstep, "Smart step"),
            (HiresFixPlus.enable_hfp_tome, "Hires ToMe"),
            (HiresFixPlus.hfp_sampler_index, "Hires sampler"),
            (HiresFixPlus.hfp_cfg, "Hires CFG"),
            #(HiresFixPlus.hfp_smartstep_min, "SmartStep min"),
            #(HiresFixPlus.hfp_tome_ratio, "ToMe ratio"),
            (HiresFixPlus.add_prompts, "Ext prompts"),
            (HiresFixPlus.alter_prompt, "Hires prompts"),
            (HiresFixPlus.alter_prompt_n, "Hires n_prompts")
        ]
        
        return [HiresFixPlus.enable_hrplus, HiresFixPlus.enable_hfp_smartstep, HiresFixPlus.enable_hfp_tome, HiresFixPlus.hfp_cfg, HiresFixPlus.hfp_sampler_index, HiresFixPlus.alter_prompt, HiresFixPlus.alter_prompt_n]

    def after_component(self, component, **kwargs):
        if kwargs.get("elem_id") == f"txtimg_hr_finalres":
            HiresFixPlus.warring_text = gr.HTML(visible=False, value="If <span style='font-weight: bold;color: #F4BF4F'>Hires steps</span> is set, <span style='font-weight: bold;color: #F4BF4F'>Use Smart-Steps</span> will do nothing!", elem_id="hfp_warring_text")

    def before_component(self, component, **kwargs):
        if kwargs.get("elem_id") == f"txt2img_prompt":
            HiresFixPlus.alter_prompt = gr.Textbox(visible=False, label="Hires Prompt", elem_id=f"hfp_prompt", lines=2, placeholder="Prompt (press Ctrl+Enter or Alt+Enter to generate)", interactive=True)
        if kwargs.get("elem_id") == f"txt2img_neg_prompt":
            HiresFixPlus.alter_prompt_n = gr.Textbox(visible=False, label="Hires Negative prompt", elem_id=f"hfp_neg_prompt", lines=2, placeholder="Negative prompt (press Ctrl+Enter or Alt+Enter to generate)", interactive=True)
        if kwargs.get("elem_id") == f"txt2img_hr_upscaler":
            HiresFixPlus.enable_hrplus = gr.Checkbox(label='Enable Hires. fix+', value=False, elem_id="hfp_enable_hfp_add")
            HiresFixPlus.enable_hfp_smartstep = gr.Checkbox(visible=False, label='Use Smart-Steps', value=False, elem_id="hfp_enable_hfp_smartstep", interactive=True)
            HiresFixPlus.enable_hfp_tome = gr.Checkbox(visible=False, label='ToMe for Hires. fix', value=False, elem_id="hfp_enable_hfp_tome", interactive=True)
            HiresFixPlus.add_prompts = gr.Checkbox(visible=False, label='Add new prompts', value=False, elem_id="hfp_add_hires_prompts", interactive=True)
        if kwargs.get("elem_id") == f"txt2img_hires_steps":
            HiresFixPlus.hfp_cfg = gr.Slider(visible=False, minimum=0.0, maximum=30.0, step=0.5, label='Hires CFG', value=0.0, elem_id="txt2img_hfp_cfg", interactive=True)
        if kwargs.get("elem_id") == f"txt2img_hr_scale":
            HiresFixPlus.hfp_sampler_index = gr.Dropdown(visible=False, label='Hires Sampling method', elem_id="hfp_sampling", choices=[x.name for x in hfp_samplers], value=hfp_samplers[0].name, type="index", interactive=True)
            
    def process_batch(self, p:StableDiffusionProcessingTxt2Img, enable_hrplus:bool, enable_hfp_smartstep:bool, enable_hfp_tome:bool, hfp_cfg:float, hfp_sampler_index:int, alter_prompt:str, alter_prompt_n:str, **kwargs):
        global hfp_samplers
        
        if p.enable_hr and (enable_hrplus or getattr(p, 'enable_hrplus', False)):
            if p.hr_resize_x == 0 and p.hr_resize_y == 0 and p.hr_scale == 1:
                return
            print(' Hijacking Hires. fix...  ')
            HiresFixPlus.on = True
            p.enable_hr = False
        else:
            HiresFixPlus.on = False
            if enable_hrplus:
                print(' Please enable Hires. fix first!  ')
            return
        
        p.do_not_save_samples = True
        HiresFixPlus.index = int(kwargs.get("batch_number"))
        #p.seed = p.seed + int(kwargs.get("batch_number"))
        
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
            
        #if opts.hfp_smartstep_min != 9 and opts.hfp_add_to_info:
        #    p.extra_generation_params["SmartStep min"] = opts.hfp_smartstep_min
            
        if opts.token_merging_ratio != 0.5 and opts.hfp_add_to_info:
            p.extra_generation_params["ToMe ratio"] = opts.token_merging_ratio
            
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
            p.extra_generation_params["Ext prompts"] = True
            
        if alter_prompt != "":
            p.extra_generation_params["Hires prompts"] = alter_prompt
            
        if alter_prompt_n != "":
            p.extra_generation_params["Hires n_prompts"] = alter_prompt_n
            
        shared.total_tqdm.updateTotal(p.steps)
        
    def postprocess_image(self, p, pp, enable_hrplus:bool, enable_hfp_smartstep:bool, enable_hfp_tome:bool, hfp_cfg:float, hfp_sampler_index:int, alter_prompt:str, alter_prompt_n:str):
        global hfp_samplers
        
        if not HiresFixPlus.on:
            return
        p.enable_hr = True

        if opts.save_images_before_highres_fix:
            info = create_infotext(p, p.all_prompts, p.all_seeds, p.all_subseeds, [], iteration=p.iteration)
            save_image(pp.image, p.outpath_samples, "", int(p.all_seeds[HiresFixPlus.index]), p.all_prompts[HiresFixPlus.index], opts.samples_format, info=info, suffix="-before-highres-fix")
        
        from modules import devices
        pt = StableDiffusionProcessingTxt2Img(width=p.width, height=p.height, enable_hr=True, hr_scale=p.hr_scale, hr_resize_x=p.hr_resize_x, hr_resize_y=p.hr_resize_y)
        with devices.autocast():
            pt.init([""], [0], [0])
        target_width = pt.hr_resize_x or pt.hr_upscale_to_x
        target_height = pt.hr_resize_y or pt.hr_upscale_to_y

        opts.img2img_fix_steps = True
        step_opt = opts.img2img_fix_steps
        steps = get_steps(p,enable_hfp_smartstep,opts.hfp_smartstep_min)
        shared.total_tqdm.updateTotal(p.steps + steps)
        
        p2:StableDiffusionProcessingImg2Img = txt2img_to_img2img(p)
        
        if p.hr_upscaler in latent_upscaler:
            img = pp.image
            p2.resize_mode = 3
        else:
            img = resize_image(0, pp.image, target_width, target_height, upscaler_name=p.hr_upscaler)
            p2.resize_mode = 0
            
        if opts.hfp_add_prompts:
            p2.prompt = p2.prompt.rstrip(',')+', '+alter_prompt if p2.prompt != '' else p2.prompt
            p2.negative_prompt = p2.negative_prompt.rstrip(',')+', '+alter_prompt_n if p2.negative_prompt != '' else p2.negative_prompt
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
        p2.seed = p.all_seeds[HiresFixPlus.index]
        p2.subseed = p.all_subseeds[HiresFixPlus.index]
        p2.subseed_strength = p.subseed_strength
        
        p2.extra_generation_params["Steps"] = p.steps
        p2.extra_generation_params["Sampler"] = p.sampler_name
        p2.extra_generation_params["CFG scale"] = p.cfg_scale
        p2.extra_generation_params["Size"] = f"{p.width}x{p.height}"
        if alter_prompt != "":
            p2.extra_generation_params["Prompt"] = p.prompt
        if alter_prompt_n != "":
            p2.extra_generation_params["Negative prompt"] = p.negative_prompt
        
        if enable_hfp_tome or getattr(p, "enable_hfp_tome", False):
            ToMe.load(p.sd_model, getattr(p, "hfp_tome_ratio", 0) or opts.token_merging_ratio)
        proc = process_images(p2)
        if enable_hfp_tome or getattr(p, "enable_hfp_tome", False):
            ToMe.unload(p.sd_model)
        
        batch_index = 0
        imgs = proc.images
        pp.image = imgs[0]
        
        opts.img2img_fix_steps = step_opt

def create_settings_items():
    section_tome = ('token_merging', 'Token Merging')
    section_hfp = ('hiresfix_plus', 'Hires. fix+')
    opts.add_option("hfp_add_prompts", shared.OptionInfo(
        True, "Append Hires prompts to the end of the original prompts instead of replacing it.", section=section_hfp
    ))
    opts.add_option("hfp_add_to_info", shared.OptionInfo(
        False, "Add Smart-Steps minimum step and ToMe merging ratio value to generation information.", section=section_hfp
    ))
    opts.add_option("hfp_smartstep_min", shared.OptionInfo(
        9, "If Smart-Step is enabled, the number of iterations for Hires. fix will never be less than this:",
        gr.Slider, {"minimum": 1, "maximum": 50, "step": 1}, section=section_hfp
    ))
    shared.opts.add_option("token_merging_ratio", shared.OptionInfo(
        0.5, "Token Merging - Ratio",
        gr.Slider, {"minimum": 0, "maximum": 0.75, "step": 0.1}, section=section_tome
    ))
    shared.opts.add_option("token_merging_max_downsample", shared.OptionInfo(
        "1", "Token Merging - Max downsample",
        gr.Radio, {"choices": ["1", "2", "4", "8"]}, section=section_tome
    ))
    shared.opts.add_option("token_merging_stride_x", shared.OptionInfo(
        2, "Token Merging - Stride X",
        gr.Slider, {"minimum": 2, "maximum": 8, "step": 2}, section=section_tome
    ))
    shared.opts.add_option("token_merging_stride_y", shared.OptionInfo(
        2, "Token Merging - Stride Y",
        gr.Slider, {"minimum": 2, "maximum": 8, "step": 2}, section=section_tome
    ))
    shared.opts.add_option('token_merging_use_rand', shared.OptionInfo(
        False, 'Token Merging - Use random perturbations', section=section_tome
    ))
    shared.opts.add_option('token_merging_merge_attn', shared.OptionInfo(
        True, 'Token Merging - Merge attention', section=section_tome
    ))
    shared.opts.add_option('token_merging_merge_crossattn', shared.OptionInfo(
        False, 'Token Merging - Merge cross-attention', section=section_tome
    ))
    shared.opts.add_option('token_merging_merge_mlp', shared.OptionInfo(
        False, 'Token Merging - Merge mlp layers', section=section_tome
    ))

def create_script_items():
    try:
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
    except Exception as e:
        traceback.print_exc()
        print(f"Failed to add support for X/Y/Z Plot Script because: {e}")
        
script_callbacks.on_before_ui(create_script_items)
scripts.script_callbacks.on_ui_settings(create_settings_items)