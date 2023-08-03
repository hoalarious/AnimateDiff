<br><img src="__assets__/figs/gradio2.png" style="width: 50em; margin-top: 1em">
# AnimateDiff

This a fork of the official repo. Made specifically to run in Colab with gradio UI. Local requires 12GB VRAM

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/hoalarious/AnimateDiff/blob/main/AnimateDiff_colab.ipynb)

(Original colab by [@camenduru](https://github.com/camenduru))

## Features
- `31/07/2023` Additional GIF output. Default displays GIF since autolooping on gradio mp4 is not directly supported. Frames are also saved into the output folder. MP4 is still accessible via the mp4 tab in the output area of the UI.
- `30/07/2023` Longer videos via moving context window. Experimental. (Credit to https://github.com/dajes/AnimateDiff/tree/longer_videos)
- `29/07/2023` Init image. Euler sampler is disabled when using init images. (Credit to https://github.com/talesofai/AnimateDiff)
- `28/07/2023` Download models via UI
- `28/07/2023` 100% inference speed due to fp16 (Credit to https://github.com/dajes/AnimateDiff/tree/longer_videos)
- `28/07/2023` Loading of multiple LoRAs (Without degrading the network with each generation)

## Todo
- Testing
- Allow uploading of init images
- ClIP interrogator
- PNGinfo
- Load/Save from configs
- Configs tab + gdrive
- Batch operations
- Init image strength and decay options. Allowing you to control how strong the generation should follow to the initial image and how quickly it diverges from it.
- Implement rife-ncnn-vulkan (Idea from https://github.com/neggles/animatediff-cli)
- Explore implentation of control net with mov2mov

## Testing and replication

<table width="100%">
    <tr>
        <td width="20%">Example</td>
        <td width="20%">Caption</td>
        <td width="20%">Inference time</td>
        <td width="20%">Config</td>
    </tr>
    <tr>
        <td width="20%"><img src="__assets__/animations/example_euler_fp32.gif"></td>
        <td width="20%">Euler with FP32 instead of FP16 </td>
        <td width="20%">~200s </td>
        <td width="20%">sampler: "Euler"</br>FP16: False</td>
    </tr>
    <tr>
        <td width="20%"><img src="__assets__/animations/example_euler.gif"></td>
        <td width="20%">Euler</td>
        <td width="20%">~80s</td>
        <td width="20%">sampler: "Euler"</td>
    </tr>
    <tr>
        <td width="20%"><img src="__assets__/animations/example_DDIM.gif"></td>
        <td width="20%">DDIM</td>
        <td width="20%">~80s</td>
        <td width="20%">sampler: "DDIM"</td>
    </tr>
    <tr>
        <td width="20%"><img src="__assets__/animations/example_PNDM.gif"></td>
        <td width="20%">PNDM</td>
        <td width="20%">~110s</td>
        <td width="20%">sampler: "PNDM"</td>
    </tr>
    <tr>
        <td width="20%"><img src="__assets__/animations/example_init_image.gif"></td>
        <td width="20%">Euler with init image</td>
        <td width="20%">~80s</td>
        <td width="20%">sampler: "Euler"</br>init_image: "configs/prompts/yoimiya-init.jpg"</td>
    </tr>
    <tr>
        <td width="20%"><img src="__assets__/animations/example_init_image_longer_video.gif"></td>
        <td width="20%">Double length video DDIM with init image using sliding context </td>
        <td width="20%">~700s </td>
        <td width="20%">sampler: "DDIM"</br>init_image: "configs/prompts/yoimiya-init.jpg"</br>temporal_context: 20</br>overlap: 5</br>strides: 1</td>
    </tr>

</table>

Init image:

![Init image](configs/prompts/yoimiya-init.jpg)

Common config between tests:

```
{
    "stable_diffusion": "/content/AnimateDiff/models/StableDiffusion/stable-diffusion-v1-5/",
    "motion_model": "mm_sd_v14.ckpt",
    "base_checkpoint": "AnythingV5Ink_v5PrtRE.safetensors",
    "prompt": "1girl, yoimiya (genshin impact), origen, line, comet, wink, Masterpiece \uff0cBestQuality \uff0cUltraDetailed",
    "n_prompt": "NSFW, lr, nsfw,(sketch, duplicate, ugly, huge eyes, text, logo, monochrome, worst face, (bad and mutated hands:1.3), (worst quality:2.0), (low quality:2.0), (blurry:2.0), horror, geometry, bad_prompt_v2, (bad hands), (missing fingers), multiple limbs, bad anatomy, (interlocked fingers:1.2), Ugly Fingers, (extra digit and hands and fingers and legs and arms:1.4), crown braid, ((2girl)), (deformed fingers:1.2), (long fingers:1.2),succubus wings,horn,succubus horn,succubus hairstyle, (bad-artist-anime), bad-artist, bad hand, grayscale, skin spots, acnes, skin blemishes",
    "num_inference_steps": 25,
    "guidance_scale": 7.5,
    "width": 512,
    "height": 512,
    "video_length": 40,
    "seed": 83026725601855,
    "temporal_context": 20,
    "strides": 1,
    "overlap": 5,
    "fp16": true,
    "lora_list": [
        {
            "path": "/content/AnimateDiff/models/loras/LineLine2D.safetensors",
            "alpha": 0.8
        },
        {
            "path": "/content/AnimateDiff/models/loras/yomiya.safetensors",
            "alpha": 0.8
        }
    ]
}

```

## Why not build this as an A1111 extension?
- There's already an A1111 Extension at https://github.com/continue-revolution/sd-webui-animatediff
- Some features are easier to implement by building this seperately. It will allow us to explore different techniques to improve AnimateDiff.

## Definitions
```
context_length: the length of the sliding window (limited by motion modules capacity), default to L.
context_overlap: how much neighbouring contexts overlap. By default context_length / 2
context_stride: (2^context_stride) is a max stride between 2 neighbour frames. By default 0

By dajes @ https://github.com/guoyww/AnimateDiff/pull/25
```



## From original readme

**[AnimateDiff: Animate Your Personalized Text-to-Image Diffusion Models without Specific Tuning](https://arxiv.org/abs/2307.04725)**
</br>
Yuwei Guo,
Ceyuan Yang*,
Anyi Rao,
Yaohui Wang,
Yu Qiao,
Dahua Lin,
Bo Dai
<p style="font-size: 0.8em; margin-top: -1em">*Corresponding Author</p>

<!-- [Arxiv Report](https://arxiv.org/abs/2307.04725) | [Project Page](https://animatediff.github.io/) -->
[![arXiv](https://img.shields.io/badge/arXiv-2307.04725-b31b1b.svg)](https://arxiv.org/abs/2307.04725)
[![Project Page](https://img.shields.io/badge/Project-Website-green)](https://animatediff.github.io/)
[![Open in OpenXLab](https://cdn-static.openxlab.org.cn/app-center/openxlab_app.svg)](https://openxlab.org.cn/apps/detail/Masbfca/AnimateDiff)
[![Hugging Face Spaces](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Spaces-yellow)](https://huggingface.co/spaces/guoyww/AnimateDiff)

## Features
- GPU Memory Optimization, ~12GB VRAM to inference
- User Interface: [Gradio](#gradio-demo), A1111 WebUI Extension [sd-webui-animatediff](https://github.com/continue-revolution/sd-webui-animatediff) (by [@continue-revolution](https://github.com/continue-revolution))
- Google Colab: [Colab](https://colab.research.google.com/github/camenduru/AnimateDiff-colab/blob/main/AnimateDiff_colab.ipynb) (by [@camenduru](https://github.com/camenduru))

## Common Issues
<details>
<summary>Installation</summary>

Please ensure the installation of [xformer](https://github.com/facebookresearch/xformers) that is applied to reduce the inference memory.
</details>


<details>
<summary>Various resolution or number of frames</summary>
Currently, we recommend users to generate animation with 16 frames and 512 resolution that are aligned with our training settings. Notably, various resolution/frames may affect the quality more or less. 
</details>


<details>
<summary>How to use it without any coding</summary>

1) Get lora models: train lora model with [A1111](https://github.com/continue-revolution/sd-webui-animatediff) based on a collection of your own favorite images (e.g., tutorials [English](https://www.youtube.com/watch?v=mfaqqL5yOO4), [Japanese](https://www.youtube.com/watch?v=N1tXVR9lplM), [Chinese](https://www.bilibili.com/video/BV1fs4y1x7p2/)) 
or download Lora models from [Civitai](https://civitai.com/).

2) Animate lora models: using gradio interface or A1111 
(e.g., tutorials [English](https://github.com/continue-revolution/sd-webui-animatediff), [Japanese](https://www.youtube.com/watch?v=zss3xbtvOWw), [Chinese](https://941ai.com/sd-animatediff-webui-1203.html)) 

3) Be creative togther with other techniques, such as, super resolution, frame interpolation, music generation, etc.
</details>


<details>
<summary>Animating a given image</summary>

We totally agree that animating a given image is an appealing feature, which we would try to support officially in future. For now, you may enjoy other efforts from the [talesofai](https://github.com/talesofai/AnimateDiff).  
</details>

<details>
<summary>Contributions from community</summary>
Contributions are always welcome!! The <code>dev</code> branch is for community contributions. As for the main branch, we would like to align it with the original technical report :)
</details>


## Setup for Inference

### Prepare Environment

***We updated our inference code with xformers and a sequential decoding trick. Now AnimateDiff takes only ~12GB VRAM to inference, and run on a single RTX3090 !!***

```
git clone https://github.com/guoyww/AnimateDiff.git
cd AnimateDiff

conda env create -f environment.yaml
conda activate animatediff
```

if using --format mp4

`pip install "imageio[ffmpeg]"`

### Download Base T2I & Motion Module Checkpoints
We provide two versions of our Motion Module, which are trained on stable-diffusion-v1-4 and finetuned on v1-5 seperately.
It's recommanded to try both of them for best results.
```
git lfs install
git clone https://huggingface.co/runwayml/stable-diffusion-v1-5 models/StableDiffusion/

bash download_bashscripts/0-MotionModule.sh
```
You may also directly download the motion module checkpoints from [Google Drive](https://drive.google.com/drive/folders/1EqLC65eR1-W-sGD0Im7fkED6c8GkiNFI?usp=sharing) / [HuggingFace](https://huggingface.co/guoyww/animatediff) / [CivitAI](https://civitai.com/models/108836), then put them in `models/Motion_Module/` folder.

### Prepare Personalize T2I
Here we provide inference configs for 6 demo T2I on CivitAI.
You may run the following bash scripts to download these checkpoints.
```
bash download_bashscripts/1-ToonYou.sh
bash download_bashscripts/2-Lyriel.sh
bash download_bashscripts/3-RcnzCartoon.sh
bash download_bashscripts/4-MajicMix.sh
bash download_bashscripts/5-RealisticVision.sh
bash download_bashscripts/6-Tusun.sh
bash download_bashscripts/7-FilmVelvia.sh
bash download_bashscripts/8-GhibliBackground.sh
bash download_bashscripts/9-AdditionalNetworks.sh
```

### Inference
After downloading the above peronalized T2I checkpoints, run the following commands to generate animations. The results will automatically be saved to `samples/` folder.
```
python -m scripts.animate --config configs/prompts/1-ToonYou.yaml
python -m scripts.animate --config configs/prompts/2-Lyriel.yaml
python -m scripts.animate --config configs/prompts/3-RcnzCartoon.yaml
python -m scripts.animate --config configs/prompts/4-MajicMix.yaml
python -m scripts.animate --config configs/prompts/5-RealisticVision.yaml
python -m scripts.animate --config configs/prompts/6-Tusun.yaml
python -m scripts.animate --config configs/prompts/7-FilmVelvia.yaml
python -m scripts.animate --config configs/prompts/8-GhibliBackground.yaml
python -m scripts.animate --config configs/prompts/9-AdditionalNetworks.yml
```

To generate animations with a new DreamBooth/LoRA model, you may create a new config `.yaml` file in the following format:
```
NewModel:
  path: "[path to your DreamBooth/LoRA model .safetensors file]"
  base: "[path to LoRA base model .safetensors file, leave it empty string if not needed]"

  motion_module:
    - "models/Motion_Module/mm_sd_v14.ckpt"
    - "models/Motion_Module/mm_sd_v15.ckpt"
    
  steps:          25
  guidance_scale: 7.5

  prompt:
    - "[positive prompt]"

  n_prompt:
    - "[negative prompt]"
```
Then run the following commands:
```
python -m scripts.animate --config [path to the config file]
```

## Gradio Demo
We have created a Gradio demo to make AnimateDiff easier to use. To launch the demo, please run the following commands:
```
conda activate animatediff
python app.py
```
By default, the demo will run at `localhost:7860`.
<br><img src="__assets__/figs/gradio.jpg" style="width: 50em; margin-top: 1em">

## Gallery
Here we demonstrate several best results we found in our experiments.

<table class="center">
    <tr>
    <td><img src="__assets__/animations/model_01/01.gif"></td>
    <td><img src="__assets__/animations/model_01/02.gif"></td>
    <td><img src="__assets__/animations/model_01/03.gif"></td>
    <td><img src="__assets__/animations/model_01/04.gif"></td>
    </tr>
</table>
<p style="margin-left: 2em; margin-top: -1em">Model：<a href="https://civitai.com/models/30240/toonyou">ToonYou</a></p>

<table>
    <tr>
    <td><img src="__assets__/animations/model_02/01.gif"></td>
    <td><img src="__assets__/animations/model_02/02.gif"></td>
    <td><img src="__assets__/animations/model_02/03.gif"></td>
    <td><img src="__assets__/animations/model_02/04.gif"></td>
    </tr>
</table>
<p style="margin-left: 2em; margin-top: -1em">Model：<a href="https://civitai.com/models/4468/counterfeit-v30">Counterfeit V3.0</a></p>

<table>
    <tr>
    <td><img src="__assets__/animations/model_03/01.gif"></td>
    <td><img src="__assets__/animations/model_03/02.gif"></td>
    <td><img src="__assets__/animations/model_03/03.gif"></td>
    <td><img src="__assets__/animations/model_03/04.gif"></td>
    </tr>
</table>
<p style="margin-left: 2em; margin-top: -1em">Model：<a href="https://civitai.com/models/4201/realistic-vision-v20">Realistic Vision V2.0</a></p>

<table>
    <tr>
    <td><img src="__assets__/animations/model_04/01.gif"></td>
    <td><img src="__assets__/animations/model_04/02.gif"></td>
    <td><img src="__assets__/animations/model_04/03.gif"></td>
    <td><img src="__assets__/animations/model_04/04.gif"></td>
    </tr>
</table>
<p style="margin-left: 2em; margin-top: -1em">Model： <a href="https://civitai.com/models/43331/majicmix-realistic">majicMIX Realistic</a></p>

<table>
    <tr>
    <td><img src="__assets__/animations/model_05/01.gif"></td>
    <td><img src="__assets__/animations/model_05/02.gif"></td>
    <td><img src="__assets__/animations/model_05/03.gif"></td>
    <td><img src="__assets__/animations/model_05/04.gif"></td>
    </tr>
</table>
<p style="margin-left: 2em; margin-top: -1em">Model：<a href="https://civitai.com/models/66347/rcnz-cartoon-3d">RCNZ Cartoon</a></p>

<table>
    <tr>
    <td><img src="__assets__/animations/model_06/01.gif"></td>
    <td><img src="__assets__/animations/model_06/02.gif"></td>
    <td><img src="__assets__/animations/model_06/03.gif"></td>
    <td><img src="__assets__/animations/model_06/04.gif"></td>
    </tr>
</table>
<p style="margin-left: 2em; margin-top: -1em">Model：<a href="https://civitai.com/models/33208/filmgirl-film-grain-lora-and-loha">FilmVelvia</a></p>

### Longer generations
You can also generate longer animations by using overlapping sliding windows.
```
python -m scripts.animate --config configs/prompts/{your_config}.yaml --L 64 --context_length 16 
```
##### Sliding window related parameters:

```L``` - the length of the generated animation.

```context_length``` - the length of the sliding window (limited by motion modules capacity), default to ```L```.

```context_overlap``` - how much neighbouring contexts overlap. By default ```context_length``` / 2

```context_stride``` - (2^```context_stride```) is a max stride between 2 neighbour frames. By default 0

##### Extended this way gallery examples

<table class="center">
    <tr>
    <td><img src="__assets__/animations/model_01_4x/01.gif"></td>
    <td><img src="__assets__/animations/model_01_4x/02.gif"></td>
    <td><img src="__assets__/animations/model_01_4x/03.gif"></td>
    <td><img src="__assets__/animations/model_01_4x/04.gif"></td>
    </tr>
</table>
<p style="margin-left: 2em; margin-top: -1em">Model：<a href="https://civitai.com/models/30240/toonyou">ToonYou</a></p>

<table>
    <tr>
    <td><img src="__assets__/animations/model_03_4x/01.gif"></td>
    <td><img src="__assets__/animations/model_03_4x/02.gif"></td>
    <td><img src="__assets__/animations/model_03_4x/03.gif"></td>
    <td><img src="__assets__/animations/model_03_4x/04.gif"></td>
    </tr>
</table>
<p style="margin-left: 2em; margin-top: -1em">Model：<a href="https://civitai.com/models/4201/realistic-vision-v20">Realistic Vision V2.0</a></p>

#### Community Cases
Here are some samples contributed by the community artists. Create a Pull Request if you would like to show your results here😚.

<table>
    <tr>
    <td><img src="__assets__/animations/model_07/init.jpg"></td>
    <td><img src="__assets__/animations/model_07/01.gif"></td>
    <td><img src="__assets__/animations/model_07/02.gif"></td>
    <td><img src="__assets__/animations/model_07/03.gif"></td>
    <td><img src="__assets__/animations/model_07/04.gif"></td>
    </tr>
</table>
<p style="margin-left: 2em; margin-top: -1em">
Character Model：<a href="https://civitai.com/models/13237/genshen-impact-yoimiya">Yoimiya</a> 
(with an initial reference image, see <a href="https://github.com/talesofai/AnimateDiff">WIP fork</a> for the extended implementation.)


<table>
    <tr>
    <td><img src="__assets__/animations/model_08/01.gif"></td>
    <td><img src="__assets__/animations/model_08/02.gif"></td>
    <td><img src="__assets__/animations/model_08/03.gif"></td>
    <td><img src="__assets__/animations/model_08/04.gif"></td>
    </tr>
</table>
<p style="margin-left: 2em; margin-top: -1em">
Character Model：<a href="https://civitai.com/models/9850/paimon-genshin-impact">Paimon</a>;
Pose Model：<a href="https://civitai.com/models/107295/or-holdingsign">Hold Sign</a></p>

## BibTeX
```
@article{guo2023animatediff,
  title={AnimateDiff: Animate Your Personalized Text-to-Image Diffusion Models without Specific Tuning},
  author={Guo, Yuwei and Yang, Ceyuan and Rao, Anyi and Wang, Yaohui and Qiao, Yu and Lin, Dahua and Dai, Bo},
  journal={arXiv preprint arXiv:2307.04725},
  year={2023}
}
```

## Contact Us
**Yuwei Guo**: [guoyuwei@pjlab.org.cn](mailto:guoyuwei@pjlab.org.cn)  
**Ceyuan Yang**: [yangceyuan@pjlab.org.cn](mailto:yangceyuan@pjlab.org.cn)  
**Bo Dai**: [daibo@pjlab.org.cn](mailto:daibo@pjlab.org.cn)

## Acknowledgements
Codebase built upon [Tune-a-Video](https://github.com/showlab/Tune-A-Video).