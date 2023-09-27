#!/usr/bin/env python
# -*- coding: UTF-8 -*-

import os
from novita_client import NovitaClient, Txt2ImgRequest, Samplers, ProgressResponseStatusCode, ModelType, add_lora_to_prompt, save_image


client = NovitaClient(os.getenv('NOVITA_API_KEY'))

res = client.sync_txt2img(Txt2ImgRequest(
    prompt="modern american comic about superman vs batman, digital color comicbook style, Batman jumps out of the way as Superman charges towards him, but Superman quickly recovers and delivers a powerful kick",
    batch_size=1,
    cfg_scale=7.0,
    sampler_name=Samplers.DPMPP_M_KARRAS,
    model_name="sd_xl_base_1.0.safetensors",
    height=1024,
    width=1024,
    seed=2563190745,
))

if res.data.status != ProgressResponseStatusCode.SUCCESSFUL:
    raise Exception('Failed to generate image with error: ' +
                    res.data.failed_reason)
save_image(res.data.imgs_bytes[0], "test.png")
