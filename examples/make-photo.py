#!/usr/bin/env python
# -*- coding: UTF-8 -*-

import os
from novita_client import NovitaClient
import base64


client = NovitaClient(os.getenv('NOVITA_API_KEY'), os.getenv('NOVITA_API_URI', None))

res = client.make_photo(
    model_name="protovisionXLHighFidelity3D_release0620Bakedvae_131308.safetensors",
    prompt="instagram photo, portrait photo of a woman img, colorful, perfect face, natural skin, hard shadows, film grain",
    images=[
        "https://tencentarc-photomaker.hf.space/--replicas/i0n9t/file=/tmp/gradio/89a694e30ca80f6b76dc4a57a3bce83ad36c5c86/scarlett_0.jpg"
    ],
    steps=50,
    guidance_scale=5,
    image_num=1,
    strength=0.2,
    seed=-1,
)


for idx in range(len(res.images_encoded)):
    with open(f"make_photo_{idx}.png", "wb") as f:
        f.write(base64.b64decode(res.images_encoded[idx]))
