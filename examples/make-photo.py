#!/usr/bin/env python
# -*- coding: UTF-8 -*-

import os
from novita_client import NovitaClient, MakePhotoLoRA
import base64


client = NovitaClient(os.getenv('NOVITA_API_KEY'), os.getenv('NOVITA_API_URI', None))

res = client.make_photo(
    model_name="sd_xl_base_1.0.safetensors",
    prompt="anime artwork man img, portrait. anime style, key visual, vibrant, studio anime, highly detailed",
    negative_prompt="wrong, photo, deformed, black and white, realism, disfigured, low contrast",
    images=[
        "../testdataset/portrait2image/7.jpg"
    ],
    loras=[
        MakePhotoLoRA(
            model_name="sdxl_wrong_lora",
            strength=0.8
        )
    ],
    steps=25,
    guidance_scale=5,
    image_num=1,
    strength=0.3,
    seed=1024,
)


for idx in range(len(res.images_encoded)):
    with open(f"make_photo_{idx}.png", "wb") as f:
        f.write(base64.b64decode(res.images_encoded[idx]))
