import os
from novita_client import NovitaClient
from novita_client.utils import base64_to_image


client = NovitaClient(os.getenv('NOVITA_API_KEY'), os.getenv('NOVITA_API_URI', None))
res = client.relight(
        input_image = "https://raw.githubusercontent.com/CompVis/latent-diffusion/main/data/inpainting_examples/overture-creations-5sI6fQgYIuo.png",
        prompt = "a sunny day",
        model_name="realisticVisionV60B1_v60B1VAE_190174.safetensors",
        steps=15,
        sampler_name="Euler a",
        guidance_scale=10,
        strength=0.5,
        clip_skip=4,
        lighting_preference = "TOP_LIGHT",
    )
base64_to_image(res.image_file).save("relight.png")