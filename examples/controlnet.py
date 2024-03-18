#!/usr/bin/env python
# -*- coding: UTF-8 -*-

import os

from novita_client import NovitaClient, Img2ImgV3Request, Img2ImgV3ControlNetUnit, ControlnetUnit, Samplers, Img2ImgV3Embedding
from novita_client.utils import base64_to_image


client = NovitaClient(os.getenv('NOVITA_API_KEY'), os.getenv('NOVITA_API_URI', None))
res = client.img2img_v3(
    input_image="https://img.freepik.com/premium-photo/close-up-dogs-face-with-big-smile-generative-ai_900101-62851.jpg",
    model_name="dreamshaper_8_93211.safetensors",
    prompt="a cute dog",
    sampler_name=Samplers.DPMPP_M_KARRAS,
    width=512,
    height=512,
    steps=30,
    controlnet_units=[
        Img2ImgV3ControlNetUnit(
            image_base64="https://img.freepik.com/premium-photo/close-up-dogs-face-with-big-smile-generative-ai_900101-62851.jpg",
            model_name="control_v11f1p_sd15_depth",
            strength=1.0
        )
    ],
    embeddings=[Img2ImgV3Embedding(model_name=_) for _ in [
        "BadDream_53202",
    ]],
    seed=-1,
)


base64_to_image(res.images_encoded[0]).save("./img2img-controlnet.png")
