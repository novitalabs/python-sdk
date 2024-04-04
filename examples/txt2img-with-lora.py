#!/usr/bin/env python
# -*- coding: UTF-8 -*-

import os
from novita_client import NovitaClient, Txt2ImgV3LoRA, Samplers, ProgressResponseStatusCode, ModelType, add_lora_to_prompt, save_image
from novita_client.utils import base64_to_image, input_image_to_pil
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

res1 = client.txt2img_v3(
    prompt="a photo of handsome man, close up",
    image_num=1,
    guidance_scale=7.0,
    sampler_name=Samplers.DPMPP_M_KARRAS,
    model_name="dreamshaper_8_93211.safetensors",
    height=512,
    width=512,
    seed=1024,
)
res2 = client.txt2img_v3(
    prompt="a photo of handsome man, close up",
    image_num=1,
    guidance_scale=7.0,
    sampler_name=Samplers.DPMPP_M_KARRAS,
    model_name="dreamshaper_8_93211.safetensors",
    height=512,
    width=512,
    seed=1024,
    loras=[
        Txt2ImgV3LoRA(
           model_name="add_detail_44319",
           strength=0.9,
        )
    ]
)

make_image_grid([base64_to_image(res1.images_encoded[0]), base64_to_image(res2.images_encoded[0])], 1, 2, 512).save("./txt2img-lora-compare.png")
