import os

from novita_client import NovitaClient, Samplers, Txt2ImgV3HiresFix
from novita_client.utils import base64_to_image

from PIL import Image


client = NovitaClient(os.getenv('NOVITA_API_KEY'), os.getenv('NOVITA_API_URI', None))
res = client.txt2img_v3(
    model_name='dreamshaper_8_93211.safetensors',
    prompt="a cute girl",
    width=384,
    height=512,
    image_num=1,
    guidance_scale=7.5,
    seed=12345,
    sampler_name=Samplers.EULER_A,
    hires_fix=Txt2ImgV3HiresFix(
        # upscaler='Latent'
        target_width=768,
        target_height=1024,
        strength=0.5
    )
)


base64_to_image(res.images_encoded[0]).save("./txt2img_with_hiresfix.png")
