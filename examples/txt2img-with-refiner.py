import os

from novita_client import NovitaClient, Txt2ImgV3Refiner, Samplers
from novita_client.utils import base64_to_image
from PIL import Image


def make_image_grid(images, rows: int, cols: int, resize: int = None):
    """
    Prepares a single grid of images. Useful for visualization purposes.
    """
    assert len(images) == rows * cols

    if resize is not None:
        images = [img.resize((resize, resize)) for img in images]

    w, h = images[0].size
    grid = Image.new("RGB", size=(cols * w, rows * h))

    for i, img in enumerate(images):
        grid.paste(img, box=(i % cols * w, i // cols * h))
    return grid


client = NovitaClient(os.getenv('NOVITA_API_KEY'), os.getenv('NOVITA_API_URI', None))

r1 = client.txt2img_v3(
    model_name='sd_xl_base_1.0.safetensors',
    prompt='a astronaut riding a bike on the moon',
    width=1024,
    height=1024,
    image_num=1,
    guidance_scale=7.5,
    sampler_name=Samplers.EULER_A,
)

r2 = client.txt2img_v3(
    model_name='sd_xl_base_1.0.safetensors',
    prompt='a astronaut riding a bike on the moon',
    width=1024,
    height=1024,
    image_num=1,
    guidance_scale=7.5,
    sampler_name=Samplers.EULER_A,
    refiner=Txt2ImgV3Refiner(
        switch_at=0.7
    )
)

r3 = client.txt2img_v3(
    model_name='sd_xl_base_1.0.safetensors',
    prompt='a astronaut riding a bike on the moon',
    width=1024,
    height=1024,
    image_num=1,
    guidance_scale=7.5,
    sampler_name=Samplers.EULER_A,
    refiner=Txt2ImgV3Refiner(
        switch_at=0.5
    )
)


make_image_grid([base64_to_image(r1.images_encoded[0]), base64_to_image(r2.images_encoded[0]), base64_to_image(r3.images_encoded[0])], 1, 3, 1024).save("./txt2img-refiner-compare.png")
