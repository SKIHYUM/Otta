from otta.pipelines.pipeline_xzh import xzhPipeline
from otta.models.unet import UNet3DConditionModel
from otta.util import save_videos_mp4
import torch
import torch.nn.functional as F


pretrained_model_path = "./checkpoints/stable-diffusion-v1-4"
my_model_path = "./outputs/car-turn"
unet = UNet3DConditionModel.from_pretrained(my_model_path, subfolder='unet', torch_dtype=torch.float16).to('cuda')
pipe = xzhPipeline.from_pretrained(
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

#
with torch.no_grad():
    prompt = "a jeep is moving on the snow"
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