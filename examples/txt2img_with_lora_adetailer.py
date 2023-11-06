#!/usr/bin/env python
# -*- coding: UTF-8 -*-

import os
from PIL import Image
from io import BytesIO
from novita_client import NovitaClient, Txt2ImgRequest, Samplers, ProgressResponseStatusCode, ModelType, add_lora_to_prompt, save_image
import logging


client = NovitaClient(os.getenv('NOVITA_API_KEY'), os.getenv('NOVITA_API_URI', None))


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


images = []
for i in range(3):
    res = client.sync_txt2img(Txt2ImgRequest(
        prompt="ohwx girl, beautify, <lora:model_1699274468_60D1BEEF6A.safetensors:0.3>, (masterpiece), (extremely intricate:1.3), (realistic), portrait of a man, the most handsome in the world, (medieval armor), metal reflections, upper body, outdoors, intense sunlight, far away castle, professional photograph of a stunning woman detailed, sharp focus, dramatic, award winning, cinematic lighting, octane render unreal engine, volumetrics dtx, (film grain, blurry background, blurry foreground, bokeh, depth of field, sunset, motion blur:1.3), chainmail",
        negative_prompt="BadDream_53202, UnrealisticDream_53204",
        batch_size=1,
        cfg_scale=10.5,
        sampler_name=Samplers.DPMPP_M_KARRAS,
        model_name="dreamshaper_8_93211.safetensors",
        height=512,
        width=512,
        steps=30,
        seed=-1,
        clip_skip=None
    ))

    if res.data.status != ProgressResponseStatusCode.SUCCESSFUL:
        raise Exception('Failed to generate image with error: ' +
                        res.data.failed_reason)

    input_image = Image.open(BytesIO((res.data.imgs_bytes[0])))
    images.append(input_image)

    import pdb;
    res = client.adetailer(model_name="dreamshaper_8_93211.safetensors", image=input_image,
                           prompt="a close up photo of ohwx girl, masterpiece, <lora:model_1699274468_60D1BEEF6A:1.0>", negative_prompt="BadDream_53202, UnrealisticDream_53204", steps=50, strength=0.3)
    images.append(Image.open(BytesIO((res.data.imgs_bytes[0]))))
    # save_image(res.data.imgs_bytes[0], "test2.png")
make_image_grid(images, 2, 3).save("test.png")
