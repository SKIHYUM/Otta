
#  Otta: One-Shot Video Tuning with Temporal Adapters on Diffusion





## Setup

### Requirements

```shell
conda create -n otta python=3.10 -y

conda activate otta

pip install -r requirements.txt
```



### Weights

**[Stable Diffusion]** [Stable Diffusion](https://arxiv.org/abs/2112.10752) is a latent text-to-image diffusion model capable of generating photo-realistic images given any text input. The pre-trained Stable Diffusion models can be downloaded from Hugging Face (e.g., [Stable Diffusion v1-4](https://huggingface.co/CompVis/stable-diffusion-v1-4), [v2-1](https://huggingface.co/stabilityai/stable-diffusion-2-1)). You can also use fine-tuned Stable Diffusion models trained on different styles (e.g, [Modern Disney](https://huggingface.co/nitrosocke/mo-di-diffusion), [Anything V4.0](https://huggingface.co/andite/anything-v4.0), [Redshift](https://huggingface.co/nitrosocke/redshift-diffusion), etc.).




## Usage

### Training

To fine-tune the text-to-image diffusion models for text-to-video generation, run this command:

```bash
CUDA_VISIBLE_DEVICES=0 accelerate launch --num_processes 1 train_otta.py --config="configs/car-turn.yaml"

```

Note: Tuning a 24-frame video usually takes `100~300` steps, about `10-2S0` minutes using one 4090 GPU. 
Reduce `n_sample_frames` if your GPU memory is limited.

### Inference

Once the training is done, run inference:

```python
from otta.pipelines.pipeline_otta import OttaPipeline
from otta.models.unet import UNet3DConditionModel
from otta.util import save_videos_mp4
import torch
import torch.nn.functional as F


pretrained_model_path = "./checkpoints/stable-diffusion-v1-4"
my_model_path = "./outputs/car-turn"
unet = UNet3DConditionModel.from_pretrained(my_model_path, subfolder='unet', torch_dtype=torch.float16).to('cuda')
pipe = OttaPipeline.from_pretrained(
    pretrained_model_path,
    unet=unet,
    torch_dtype=torch.float16,     
    low_cpu_mem_usage=False,
).to("cuda")
pipe.enable_xformers_memory_efficient_attention()
pipe.enable_vae_slicing()

pipe.unet.eval()
pipe.vae.eval()
# 
pipe.text_encoder.eval()

with torch.no_grad():
    prompt = "a jeep is moving on snow"
    ddim_inv_latent = torch.load(f"{my_model_path}/inv_latents/ddim_latent-300.pt")\
                        .to(torch.float16).to("cuda")
    video = pipe(
        prompt,
        latents=ddim_inv_latent,
        video_length=24,
        height=512,
        width=512,
        num_inference_steps=100,
        guidance_scale=15,
    ).videos

save_videos_mp4(video, f"./{prompt}.mp4")     
```




## Shoutouts

- This code builds on [diffusers](https://github.com/huggingface/diffusers) and [Tune-A-Video](https://github.com/showlab/Tune-A-Video). Thanks for open-sourcing!!!
