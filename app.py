
import os
import json
import torch
import random

import gradio as gr
from glob import glob
from omegaconf import OmegaConf
from datetime import datetime
from safetensors import safe_open

from diffusers import AutoencoderKL
from diffusers import DDIMScheduler, EulerDiscreteScheduler, PNDMScheduler
from diffusers.utils.import_utils import is_xformers_available
from transformers import CLIPTextModel, CLIPTokenizer

from animatediff.models.unet import UNet3DConditionModel
from animatediff.pipelines.pipeline_animation import AnimationPipeline
from animatediff.utils.util import save_videos_grid
from animatediff.utils.convert_from_ckpt import convert_ldm_unet_checkpoint, convert_ldm_clip_checkpoint, convert_ldm_vae_checkpoint
from animatediff.utils.convert_lora_safetensor_to_diffusers import convert_lora
from animatediff.utils.convert_lora_with_backup import load_loras


import requests
import tqdm
import re
import shutil
import concurrent.futures

import argparse
parser = argparse.ArgumentParser()

# Add boolean argument --live

parser.add_argument("--live", default=False, type=bool, help="live mode", action=argparse.BooleanOptionalAction)
args = parser.parse_args()

sample_idx     = 0
max_LoRAs      = 5
scheduler_dict = {
    "Euler": EulerDiscreteScheduler, # Doesn't work with Init image
    "DDIM": DDIMScheduler, # Works with Init image. Faster
    "PNDM": PNDMScheduler, # Works with Init image. Could be better than DDIM?
}

css = """
.toolbutton {
    margin-buttom: 0em 0em 0em 0em;
    max-width: 2.5em;
    min-width: 2.5em !important;
    height: 2.5em;
}
"""



def download_url(url, headers=None, filename=None, destination_path=None):
    if headers is None:
        headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.3"}

    response = requests.get(url, stream=True, headers=headers)
    response.raise_for_status()

    if filename is None:
        content_disposition = response.headers.get('content-disposition')
        if content_disposition:
            filename = re.findall('filename="?([^"]*)', content_disposition)
            if len(filename) != 0:
                filename = filename[0]
            else:
                raise ValueError("Cannot determine filename from headers. Please provide a filename.")
        else:
            raise ValueError("Cannot determine filename from headers. Please provide a filename.")

    total_size = response.headers.get('content-length')
    if total_size is not None:
        total_size = int(total_size)
        progress_bar = tqdm.tqdm(total=total_size, unit='iB', unit_scale=True)
    else:
        progress_bar = tqdm.tqdm(unit='iB', unit_scale=True)

    filepath = os.path.join(destination_path, filename)
    with open(filepath, 'wb') as file:
        for data in response.iter_content(chunk_size=1024):
            progress_bar.update(len(data))
            file.write(data)

    progress_bar.close()
    if total_size is not None and progress_bar.n != total_size:
        raise ValueError("ERROR, something went wrong")

    print(f"Download completed. File saved to {filepath}")

    return filepath

def download_urls(urls, destination_path, max_workers=5):
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(download_url, url, None, None, destination_path): url for url in urls}

        for future in concurrent.futures.as_completed(futures):
            url = futures[future]
            try:
                filepath = future.result()
            except Exception as exc:
                print('%r generated an exception: %s' % (url, exc))

def download_checkpoint(url, progress=gr.Progress(track_tqdm=True)):
    return download_url(url=url, destination_path=controller.checkpoints_dir)

def download_loras(url, progress=gr.Progress(track_tqdm=True)):
    return download_url(url=url, destination_path=controller.loras_dir)


class ProjectConfigs:
    def __init__(self):
        self.date_created = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        self.prompt = ""
        self.n_prompt = ""
        self.sampler = ""
        self.num_inference_steps = 20
        self.guidance_scale = 7.5
        self.width = 512
        self.height = 512
        self.video_length = 20
        self.seed = -1
        self.temporal_context = 20
        self.overlap = 20
        self.strides = 1
        self.fp16 = True
        self.loras = [{'path' : 'none', 'alpha': 0.8}]*5

    def toJSON(self):
        return json.dumps(self, default=lambda o: o.__dict__, 
            sort_keys=True, indent=4)

    def save_configs(self):
        with open(os.path.join(self.savedir, f"temp{self.date_created}.json"), "a") as f:
            f.write(self.toJSON())
            f.write("\n\n")

class AnimateController:
    def __init__(self):
        
        # config dirs
        self.basedir                = os.getcwd()
        self.stable_diffusion_dir   = os.path.join(self.basedir, "models", "StableDiffusion")
        self.motion_module_dir      = os.path.join(self.basedir, "models", "Motion_Module")
        self.checkpoints_dir        = os.path.join(self.basedir, "models", "checkpoints")
        self.loras_dir              = os.path.join(self.basedir, "models", "loras")
        self.init_images_dir        = os.path.join(self.basedir, "init_images")
        self.savedir                = os.path.join(self.basedir, "output")
        os.makedirs(self.savedir, exist_ok=True)

        self.lora_list = []
        self.project = ProjectConfigs()

        self.stable_diffusion_list   = []
        self.motion_module_list      = []
        self.checkpoints_list = []
        self.init_image_list = []
        
        self.refresh_stable_diffusion()
        self.refresh_motion_module()
        self.refresh_checkpoints()
        self.refresh_lora_models()
        self.refresh_init_images()

        # config models
        self.tokenizer             = None
        self.text_encoder          = None
        self.vae                   = None
        self.unet                  = None
        self.pipeline              = None
        self.lora_model_state_dict = {}
        
        self.inference_config      = OmegaConf.load("configs/inference/inference.yaml")

    def refresh_stable_diffusion(self):
        self.stable_diffusion_list = glob(os.path.join(self.stable_diffusion_dir, "*/"))

    def refresh_motion_module(self):
        motion_module_list = glob(os.path.join(self.motion_module_dir, "*.ckpt"))
        self.motion_module_list = [os.path.basename(p) for p in motion_module_list]

    def refresh_checkpoints(self):
        checkpoint_list = glob(os.path.join(self.checkpoints_dir, "*.safetensors"))
        self.checkpoints_list = [os.path.basename(p) for p in checkpoint_list]

    def refresh_lora_models(self):
        lora_list = glob(os.path.join(self.loras_dir, "*.safetensors"))
        self.lora_list = [os.path.basename(p) for p in lora_list]

    def refresh_init_images(self):
        self.init_image_list = glob(os.path.join(self.init_images_dir, "*"))

    def update_stable_diffusion(self, stable_diffusion_dropdown):
        self.tokenizer = CLIPTokenizer.from_pretrained(stable_diffusion_dropdown, subfolder="tokenizer")
        self.text_encoder = CLIPTextModel.from_pretrained(stable_diffusion_dropdown, subfolder="text_encoder").cuda()
        self.vae = AutoencoderKL.from_pretrained(stable_diffusion_dropdown, subfolder="vae").cuda()
        self.unet = UNet3DConditionModel.from_pretrained_2d(stable_diffusion_dropdown, subfolder="unet", unet_additional_kwargs=OmegaConf.to_container(self.inference_config.unet_additional_kwargs)).cuda()
        return gr.Dropdown.update()

    def update_motion_module(self, motion_module_dropdown):
        if self.unet is None:
            gr.Info(f"Please select a pretrained model path.")
            return gr.Dropdown.update(value=None)
        else:
            motion_module_dropdown = os.path.join(self.motion_module_dir, motion_module_dropdown)
            motion_module_state_dict = torch.load(motion_module_dropdown, map_location="cpu")
            missing, unexpected = self.unet.load_state_dict(motion_module_state_dict, strict=False)
            assert len(unexpected) == 0
            return gr.Dropdown.update()

    def update_base_model(self, checkpoint_dropdown):
        if self.unet is None:
            gr.Info(f"Please select a pretrained model path.")
            return gr.Dropdown.update(value=None)
        else:
            checkpoint_dropdown = os.path.join(self.checkpoints_dir, checkpoint_dropdown)
            base_model_state_dict = {}
            with safe_open(checkpoint_dropdown, framework="pt", device="cpu") as f:
                for key in f.keys():
                    base_model_state_dict[key] = f.get_tensor(key)
                    
            converted_vae_checkpoint = convert_ldm_vae_checkpoint(base_model_state_dict, self.vae.config)
            self.vae.load_state_dict(converted_vae_checkpoint)

            converted_unet_checkpoint = convert_ldm_unet_checkpoint(base_model_state_dict, self.unet.config)
            self.unet.load_state_dict(converted_unet_checkpoint, strict=False)

            self.text_encoder = convert_ldm_clip_checkpoint(base_model_state_dict)
            return gr.Dropdown.update()

    def process_lora_inputs(self, *args):
        lora_paths = []
        lora_alphas = []
        for arg in args:
            if isinstance(arg, str):
                lora_paths.append(arg)
            else:
                lora_alphas.append(arg)

        lora_list = []
        for index, lora_path in enumerate(lora_paths):
            if lora_path == "none":
                continue
            else:
                lora_path = os.path.join(self.loras_dir, lora_path)
            lora_list.append({
                "path": lora_path,
                "alpha": lora_alphas[index]
            })

        return lora_list

    # Load loras
    def load_lora(self, pipeline, lora_list):
        pipeline = load_loras(pipeline=pipeline, loras=lora_list, device="cuda")
        return pipeline

    def animate(
        self,
        stable_diffusion_dropdown,
        motion_module_dropdown,
        checkpoint_dropdown,
        init_image,
        prompt_textbox, 
        negative_prompt_textbox, 
        sampler_dropdown, 
        sample_step_slider, 
        width_slider, 
        length_slider, 
        height_slider, 
        cfg_scale_slider, 
        seed_textbox,
        enable_longer_videos,
        context_length,
        context_stride,
        context_overlap,
        fp16,
        # gif,
        lora_model_dropdown_0, # Need to find a better solution around this as Gradio doesn't allow dynamic number of inputs and refreshes values for direct inputs.
        lora_model_dropdown_1,
        lora_model_dropdown_2,
        lora_model_dropdown_3,
        lora_model_dropdown_4,
        lora_alpha_slider_0,
        lora_alpha_slider_1,
        lora_alpha_slider_2,
        lora_alpha_slider_3,
        lora_alpha_slider_4,

    ):    
        seed = seed_textbox
            
        lora_list = self.process_lora_inputs(
                                        lora_model_dropdown_0, # Need to find a better solution around this as Gradio doesn't allow dynamic number of inputs and refreshes values for direct inputs. Maybe use tuple as input?
                                        lora_model_dropdown_1,
                                        lora_model_dropdown_2,
                                        lora_model_dropdown_3,
                                        lora_model_dropdown_4,
                                        lora_alpha_slider_0,
                                        lora_alpha_slider_1,
                                        lora_alpha_slider_2,
                                        lora_alpha_slider_3,
                                        lora_alpha_slider_4
        )

        # Create project folder
        project_dir = os.path.join(self.savedir, f"run-{datetime.now().strftime('%Y-%m-%dT%H-%M-%S')}")

        sample_config = {
            "stable_diffusion": stable_diffusion_dropdown,
            "motion_model": motion_module_dropdown,
            "base_checkpoint": checkpoint_dropdown,
            "init_image": init_image,
            "prompt": prompt_textbox,
            "n_prompt": negative_prompt_textbox,
            "sampler": sampler_dropdown,
            "num_inference_steps": sample_step_slider,
            "guidance_scale": cfg_scale_slider,
            "width": width_slider,
            "height": height_slider,
            "video_length": length_slider,
            "seed": seed,
            "temporal_context": context_length,
            "strides": context_stride,
            "overlap": context_overlap,
            "fp16": fp16,
            "lora_list": lora_list
        }
        json_str = json.dumps(sample_config, indent=4)

        print(json_str)

        if args.live:
            if self.unet is None:
                raise gr.Error(f"Please select a pretrained model path.")
            if motion_module_dropdown == "": 
                raise gr.Error(f"Please select a motion module.")
            if checkpoint_dropdown == "":
                raise gr.Error(f"Please select a base DreamBooth model.")

            if is_xformers_available(): self.unet.enable_xformers_memory_efficient_attention()

            pipeline = AnimationPipeline(
                vae=self.vae, text_encoder=self.text_encoder, tokenizer=self.tokenizer, unet=self.unet,
                scheduler=scheduler_dict[sampler_dropdown](**OmegaConf.to_container(self.inference_config.noise_scheduler_kwargs))
            ).to("cuda")
            
            # if self.lora_model_state_dict != {}:
            #     pipeline = convert_lora(pipeline, self.lora_model_state_dict, alpha=lora_alpha_slider)


            # Load loras

            pipeline = self.load_lora(pipeline, lora_list)

            pipeline.to("cuda")

            if seed_textbox != "-1" and seed_textbox != "": torch.manual_seed(int(seed_textbox))
            else: torch.manual_seed(random.randint(1, 1e14))
            seed = torch.initial_seed()
            
            # Handle none init image
            if init_image == "none": init_image = None

            if not enable_longer_videos:
                context_length = 1000

            sample = pipeline(
                prompt              = prompt_textbox,
                init_image          = init_image,
                negative_prompt     = negative_prompt_textbox,
                num_inference_steps = sample_step_slider,
                guidance_scale      = cfg_scale_slider,
                width               = width_slider,
                height              = height_slider,
                video_length        = length_slider,
                temporal_context    = context_length,
                strides             = context_stride + 1,
                overlap             = context_overlap,
                fp16                = fp16,
            ).videos



            # Save as gif
            # if gif:
            #     save_sample_path = os.path.join(project_dir, f"output.gif")
            #     save_videos_grid(sample, save_sample_path, save_frames=True)

            # Save as Mp4
            save_sample_path = os.path.join(project_dir, f"output.mp4")
            save_videos_grid(sample, save_sample_path, save_frames=True, save_additional_gif=True)

            if init_image is not None:
                shutil.copy(init_image, f"{project_dir}/init_image.jpg")
    

            with open(os.path.join(project_dir, f"configs.json"), "a") as f:
                f.write(json_str)
                f.write("\n\n")
            


            return save_sample_path, save_sample_path.replace(".mp4", ".gif")
        return None, None
        

controller = AnimateController()


def base_model_selection_ui(live=False):
    with gr.Row():
        stable_diffusion_dropdown = gr.Dropdown(
            label="Pretrained Model Path",
            choices=controller.stable_diffusion_list,
            interactive=True,
            value=controller.stable_diffusion_list[0]
        )
        stable_diffusion_dropdown.change(fn=controller.update_stable_diffusion, inputs=[stable_diffusion_dropdown])
        
        stable_diffusion_refresh_button = gr.Button(value="\U0001F503", elem_classes="toolbutton")
        def update_stable_diffusion_list():
            controller.refresh_stable_diffusion()
            return gr.Dropdown.update(choices=controller.stable_diffusion_list)
        stable_diffusion_refresh_button.click(fn=update_stable_diffusion_list, inputs=[], outputs=[stable_diffusion_dropdown])

    with gr.Row():
        motion_module_dropdown = gr.Dropdown(
            label="Select motion module",
            choices=controller.motion_module_list,
            interactive=True,
            value=controller.motion_module_list[0]
        )
        motion_module_dropdown.change(fn=controller.update_motion_module, inputs=[motion_module_dropdown])
        
        motion_module_refresh_button = gr.Button(value="\U0001F503", elem_classes="toolbutton")
        def update_motion_module_list():
            controller.refresh_motion_module()
            return gr.Dropdown.update(choices=controller.motion_module_list)
        
        motion_module_refresh_button.click(fn=update_motion_module_list, inputs=[], outputs=[motion_module_dropdown])
        
        checkpoint_dropdown = gr.Dropdown(
            label="Select base Dreambooth model (required)",
            choices=controller.checkpoints_list,
            interactive=True,
            value=controller.checkpoints_list[0],
        )

        checkpoint_dropdown.change(fn=controller.update_base_model, inputs=[checkpoint_dropdown])
        
        checkpoint_refresh_button = gr.Button(value="\U0001F503", elem_classes="toolbutton")
        def update_checkpoints_list():
            controller.refresh_checkpoints()
            return gr.Dropdown.update(choices=controller.checkpoints_list)
        checkpoint_refresh_button.click(fn=update_checkpoints_list, inputs=[], outputs=[checkpoint_dropdown])

        # Load default models
        if live:
            controller.update_stable_diffusion(stable_diffusion_dropdown.value)
            controller.update_motion_module(motion_module_dropdown.value)
            controller.update_base_model(checkpoint_dropdown.value)

        return stable_diffusion_dropdown, motion_module_dropdown, checkpoint_dropdown

def lora_selection_ui():
    lora_dropdown_list = []
    lora_alpha_slider_list = []

    with gr.Row():
        gr.Markdown("Refresh Lora models")
        lora_refresh_button = gr.Button(value="\U0001F503", elem_classes="toolbutton")
    
    for i in range(max_LoRAs):
        with gr.Row():
            if i < 2 and i < len(controller.lora_list):
                value = controller.lora_list[i]
            else:
                value = "none"

            lora_model_dropdown = gr.Dropdown(
                    label=f"Select LoRA model {i} (optional)",
                    choices=["none"] + controller.lora_list,
                    value=value,
                    interactive=True,
                    elem_id=f"lora_model_dropdown-{i}",
                )
            
            
            lora_alpha_slider = gr.Slider(label="LoRA alpha", value=0.8, minimum=0, maximum=2, interactive=True)

            lora_dropdown_list.append(lora_model_dropdown)
            lora_alpha_slider_list.append(lora_alpha_slider)

    def update_lora_list():
        controller.refresh_lora_models()
        return [gr.Dropdown.update(choices=["none"] + controller.lora_list)]*max_LoRAs

    lora_refresh_button.click(fn=update_lora_list, inputs=[], outputs=lora_dropdown_list)

    return lora_dropdown_list, lora_alpha_slider_list


def generate_tab_ui(live=False):
    
        with gr.Accordion("1. Model checkpoints (select pretrained model path first", open=False):
            stable_diffusion_dropdown, motion_module_dropdown, checkpoint_dropdown = base_model_selection_ui(live=live)
            

        with gr.Column(variant="panel"):
            gr.Markdown(
                """
                ### 2. Configs for AnimateDiff.
                """
            )

            with gr.Tab(label="Prompts"):

                with gr.Row():
                    with gr.Column():
                        prompt_textbox = gr.Textbox(label="Prompt", lines=2, value="1girl, yoimiya (genshin impact), origen, line, comet, wink, Masterpiece ，BestQuality ，UltraDetailed")
                        negative_prompt_textbox = gr.Textbox(label="Negative prompt", lines=2, value="NSFW, lr, nsfw,(sketch, duplicate, ugly, huge eyes, text, logo, monochrome, worst face, (bad and mutated hands:1.3), (worst quality:2.0), (low quality:2.0), (blurry:2.0), horror, geometry, bad_prompt_v2, (bad hands), (missing fingers), multiple limbs, bad anatomy, (interlocked fingers:1.2), Ugly Fingers, (extra digit and hands and fingers and legs and arms:1.4), crown braid, ((2girl)), (deformed fingers:1.2), (long fingers:1.2),succubus wings,horn,succubus horn,succubus hairstyle, (bad-artist-anime), bad-artist, bad hand, grayscale, skin spots, acnes, skin blemishes")
                    

                    with gr.Column():

                        with gr.Row():
                            init_image_dropdown = gr.Dropdown(
                            label="Select init image",
                            info="Does not work with Euler sampling. Will default to DDIM if Euler was selected. PNDMScheduler is slower but could be better than DDIM. I'm not sure. Let me know if you find out.",
                            choices=["none"] + controller.init_image_list,
                            value="none",
                            interactive=True,
                        )
                            init_image_refresh_button = gr.Button(value="\U0001F503", elem_classes="toolbutton")

                        init_image_upload = gr.Image(label="Init image", interactive=True, type='filepath')

            with gr.Tab(label="LoRAs"):
                lora_dropdown_list, lora_alpha_slider_list = lora_selection_ui()
            
            with gr.Row().style(equal_height=False):
                with gr.Column():
                    with gr.Row():
                        sampler_dropdown   = gr.Dropdown(label="Sampling method", choices=list(scheduler_dict.keys()), value=list(scheduler_dict.keys())[0], info="Euler: 80s, PNDM: 110s, DDIM: 80s")
                        sample_step_slider = gr.Slider(label="Sampling steps", value=25, minimum=10, maximum=100, step=1, info="Increase this if you find the details lacking. Inference will take longer")

                    with gr.Row():
                        width_slider     = gr.Slider(label="Width", value=512, minimum=256, maximum=1024, step=64)
                        height_slider    = gr.Slider(label="Height", value=512, minimum=256, maximum=1024, step=64)

                    with gr.Row():
                        length_slider    = gr.Slider(label="Animation length", value=16,  minimum=8,   maximum=24,   step=1)
                        cfg_scale_slider = gr.Slider(label="CFG Scale", value=7.5, minimum=0,   maximum=20, info="Increase this if you find the details lacking. Balance it with the sampling steps.")

                    with gr.Row():
                        fp16 = gr.Checkbox(label="FP16", value=True, info="Generates videos ~2.7 times faster.")
                        # gif = gr.Checkbox(label="Enable GIF", value=True, info="Additionally creates GIF.")
                        enable_longer_videos = gr.Checkbox(label="Enable longer videos", value=False, info="Enable this if you want to generate videos longer than 24 frames. Inference will be ~2 times slower even for same length videos.")

                    with gr.Row(visible=False) as longer_video_row:
                        context_length  = gr.Slider(label="Context length", value=10, minimum=5,   maximum=24, step=1, info="Keep this same as [Animation length] unless you want to try animations longer than 24")
                        context_overlap = gr.Slider(label="Context overlap", value=5, minimum=5,   maximum=23, step=1, info="Condition: [Context length] * [Context stride] - [Context overlap] > 0. If not you'll get an error. Will simplify this eventually.")
                        context_stride = gr.Slider(label="Context stride", value=1, minimum=1,   maximum=3, step=1)


                    def update_enable_longer_videos(enable_longer_videos):
                        if enable_longer_videos:
                            return [gr.Slider.update(maximum=100), longer_video_row.update(visible=True)]
                        else:
                            # High number to never activate longer video function
                            return [gr.Slider.update(maximum=24), longer_video_row.update(visible=False)]
                        
                    enable_longer_videos.change(fn=update_enable_longer_videos, inputs=[enable_longer_videos], outputs=[length_slider, longer_video_row])

                    # def update_context_overlap(context_length, context_stride):
                    #     maximum = context_length * context_stride - 1
                    #     return gr.Slider.update(minimum=1, maximum=maximum)
                    
                    # context_length.change(fn=update_context_overlap, inputs=[context_length, context_overlap], outputs=[context_stride])

                    with gr.Row():
                        seed_textbox = gr.Textbox(label="Seed", value=-1)
                        seed_button  = gr.Button(value="\U0001F3B2", elem_classes="toolbutton")
                        seed_button.click(fn=lambda: gr.Textbox.update(value=random.randint(1, 1e8)), inputs=[], outputs=[seed_textbox])
            
                    generate_button = gr.Button(value="Generate", variant='primary')
                    

                with gr.Tab(label="GIF"):
                    result_gif = gr.Image(label="Generated GIF", interactive=False)

                with gr.Tab(label="MP4"):
                    result_video = gr.Video(label="Generated Video", interactive=False)

            def update_init_image_list():
                controller.refresh_init_images()
                return gr.Dropdown.update(choices=["none"] + controller.init_image_list)
            init_image_refresh_button.click(fn=update_init_image_list, inputs=[], outputs=[init_image_dropdown])

            def update_init_image_dropdown(init_image_dropdown, sampler_dropdown):
                sampler_choices = list(scheduler_dict.keys())
                if init_image_dropdown != "none":
                    sampler_choices.remove("Euler")

                sampler_value = "DDIM" if sampler_dropdown == "Euler" else sampler_dropdown
                    
                return gr.Dropdown.update(choices=sampler_choices, value=sampler_value)
            
            def update_init_image(init_image_upload, sampler_dropdown):
                if init_image_upload != None:
                    sampler_choices = list(scheduler_dict.keys())
                    sampler_choices.remove("Euler")
                    return [gr.Dropdown.update(value="none"), gr.Dropdown.update(choices=sampler_choices, value=sampler_value)]
                sampler_value = "DDIM" if sampler_dropdown == "Euler" else sampler_dropdown
                return [None, gr.Dropdown.update(choices=sampler_choices, value=sampler_value)]
            
            init_image_upload.change(fn=update_init_image, inputs=[init_image_upload, sampler_dropdown], outputs=[init_image_dropdown, sampler_dropdown])

            init_image_dropdown.change(fn=update_init_image_dropdown, inputs=[init_image_dropdown, sampler_dropdown], outputs=[sampler_dropdown])

            if init_image_upload == None:
                init_image = init_image_dropdown
            else:
                init_image = init_image_upload

            generate_button.click(
                fn=controller.animate,
                inputs=[
                    stable_diffusion_dropdown,
                    motion_module_dropdown,
                    checkpoint_dropdown,
                    init_image,
                    prompt_textbox, 
                    negative_prompt_textbox, 
                    sampler_dropdown, 
                    sample_step_slider, 
                    width_slider, 
                    length_slider, 
                    height_slider, 
                    cfg_scale_slider, 
                    seed_textbox,
                    enable_longer_videos,
                    context_length,
                    context_stride,
                    context_overlap,
                    fp16,
                    # gif
                ] + lora_dropdown_list
                + lora_alpha_slider_list,
                outputs=[result_video, result_gif]
            )

def download_tab_ui():
    with gr.Row() as checkpoint_row:
        checkpoint_url = gr.Textbox(label="Checkpoint URL", scale=5)
        checkpoint_download_button = gr.Button(value="Download checkpoint", variant='primary')

    with gr.Row() as lora_row:
        lora_url = gr.Textbox(label="LoRA URL", scale=5)
        lora_download_button = gr.Button(value="Download LoRA", variant='primary')

    checkpoint_download_button.click(fn=download_checkpoint, inputs=[checkpoint_url], outputs=[checkpoint_url])
    lora_download_button.click(fn=download_loras, inputs=[lora_url], outputs=[lora_url])

def configuration_tab_ui():
    gr.Markdown(
        """
        # WIP
        """
    )

def credits_tab_ui():
    gr.Markdown(
        """
        # [AnimateDiff: Animate Your Personalized Text-to-Image Diffusion Models without Specific Tuning](https://arxiv.org/abs/2307.04725)
        Yuwei Guo, Ceyuan Yang*, Anyi Rao, Yaohui Wang, Yu Qiao, Dahua Lin, Bo Dai (*Corresponding Author)<br>
        [Arxiv Report](https://arxiv.org/abs/2307.04725) | [Project Page](https://animatediff.github.io/) | [Github](https://github.com/guoyww/animatediff/)
        """
    )

# Get run params


def ui(live=args.live):
    with gr.Blocks(css=css) as demo:
        gr.Markdown(
            """
            # AnimateDiff Web UI
            """
        )

        with gr.Tab(label="Generate"):
            generate_tab_ui(live=live)

        with gr.Tab(label="Download"):
            download_tab_ui()

        with gr.Tab(label="Configurations"):
            configuration_tab_ui()

        with gr.Tab(label="Credits"):
            credits_tab_ui()

    return demo


if __name__ == "__main__":
    demo = ui()
    demo.queue(concurrency_count=3)
    demo.launch(share=True, debug=True)