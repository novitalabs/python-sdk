import os
import base64
from novita_client import NovitaClient
from novita_client.utils import base64_to_image

client = NovitaClient(os.getenv('NOVITA_API_KEY'), os.getenv('NOVITA_API_URI', None))
res = client.inpainting(
    model_name = "realisticVisionV40_v40VAE-inpainting_81543.safetensors",
    image="https://raw.githubusercontent.com/CompVis/latent-diffusion/main/data/inpainting_examples/overture-creations-5sI6fQgYIuo.png",
    mask="https://raw.githubusercontent.com/CompVis/latent-diffusion/main/data/inpainting_examples/overture-creations-5sI6fQgYIuo.png",
    seed=1,
    guidance_scale=15,
    steps = 20,
    image_num = 4,
    prompt = "black rabbit",
    negative_prompt = "white rabbit",
    sampler_name = "Euler a",
    inpainting_full_res = 1,
    inpainting_full_res_padding = 32,
    inpainting_mask_invert = 0,
    initial_noise_multiplier = 1,
    mask_blur = 1,
    clip_skip = 1,
    strength = 0.85,
)
with open("result/result_image/inpaintingsdk.jpeg", "wb") as image_file:
    image_file.write(base64.b64decode(res.images_encoded[0]))