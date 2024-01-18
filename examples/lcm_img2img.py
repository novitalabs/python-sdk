#!/usr/bin/env python
# -*- coding: UTF-8 -*-

import os
from novita_client import NovitaClient


client = NovitaClient(os.getenv('NOVITA_API_KEY'), os.getenv('NOVITA_API_URI', None))

res = client.lcm_img2img(
    model_name="dreamshaper_8_93211.safetensors",
    prompt="1 house",
    image="https://replicate.delivery/pbxt/JvLi9smWKKDfQpylBYosqQRfPKZPntuAziesp0VuPjidq61n/rocket.png",
    steps=4,
    guidance_scale=1,
    clip_skip=1,
    image_num=1,
)

print(res.to_json())
res.images[0]
