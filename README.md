# novita Python SDK

this SDK is based on the official [API documentation](https://docs.novita.ai/)

**join our discord server for help**

[![](https://dcbadge.vercel.app/api/server/Mqx7nWYzDF)](https://discord.com/invite/Mqx7nWYzDF) 

## New APIs

- [latent-consistency-txt2img](./examples/latent-consistency-txt2img.py) - latent consistency txt2img [based on this paper](https://latent-consistency-models.github.io/)
- [cleanup](./examples/cleanup.py) - remove all your generated images
- [remove-background](./examples/remove-background.py) - remove background from image
- [remove-text](./examples/remove-text.py) - remove text from image
- [reimagine](./examples/reimagine.py) - reimagine image
- [doodle](./examples/doodle.py) - doodle image
- [merge-face](./examples/merge-face.py) - merge face
- [mix-pose](./examples/mix-pose.py) - mix pose
- [outpainting](./examples/outpainting.py) - outpainting
- [replace-object](./examples/remove-object.py) - remove object
- [replace-background](./examples/replace-background.py) - replace background
- [replace-sky](./examples/replace-sky.py) - replace sky
- [create-tile](./examples/create-tile.py) - create tile



## Installation

```bash
pip install novita-client
```

## Quick Start

**Get api key refer to [https://novita.ai/get-started/](https://novita.ai/get-started/)**

```python
import os
from novita_client import NovitaClient, Txt2ImgRequest, Samplers, ModelType, save_image

client = NovitaClient(os.getenv('NOVITA_API_KEY'), os.getenv('NOVITA_API_URI', None))

req = Txt2ImgRequest(
    model_name='sd_xl_base_1.0.safetensors',
    prompt='a dog flying in the sky',
    batch_size=1,
    cfg_scale=7.5,
    height=1024,
    width=1024,
    sampler_name=Samplers.EULER_A,
)
save_image(client.sync_txt2img(req).data.imgs_bytes[0], 'output.png')
```

## Examples

[txt2img_with_lora.py](./examples/txt2img_with_lora.py)

```python
#!/usr/bin/env python
# -*- coding: UTF-8 -*-

import os
from novita_client import NovitaClient, Txt2ImgRequest, Samplers, ProgressResponseStatusCode, ModelType, add_lora_to_prompt, save_image


client = NovitaClient(os.getenv('NOVITA_API_KEY'), os.getenv('NOVITA_API_URI', None))
models = client.models()

# Anything V5/Ink, https://civitai.com/models/9409/or-anything-v5ink
checkpoint_model = models.filter_by_type(ModelType.CHECKPOINT).get_by_civitai_version_id(90854)

# Detail Tweaker LoRA, https://civitai.com/models/58390/detail-tweaker-lora-lora
lora_model = models.filter_by_type(ModelType.LORA).get_by_civitai_version_id(62833)

prompt = add_lora_to_prompt('a dog flying in the sky', lora_model.sd_name, "0.8")

res = client.sync_txt2img(Txt2ImgRequest(
    prompt=prompt,
    batch_size=1,
    cfg_scale=7.5,
    sampler_name=Samplers.EULER_A,
    model_name=checkpoint_model.sd_name,
    seed=103304,
))

if res.data.status != ProgressResponseStatusCode.SUCCESSFUL:
    raise Exception('Failed to generate image with error: ' +
                    res.data.failed_reason)
save_image(res.data.imgs_bytes[0], "test.png")
```

### Model Search

[model_search.py](./examples/model_search.py)

```python
from novita_client import NovitaClient, ModelType

client = NovitaClient(os.getenv('NOVITA_API_KEY'), os.getenv('NOVITA_API_URI', None))

# filter by model type
print("lora count", len(client.models().filter_by_type(ModelType.LORA)))
print("checkpoint count", len(client.models().filter_by_type(ModelType.CHECKPOINT)))
print("textinversion count", len(
    client.models().filter_by_type(ModelType.TEXT_INVERSION)))
print("vae count", len(client.models().filter_by_type(ModelType.VAE)))
print("controlnet count", len(client.models().filter_by_type(ModelType.CONTROLNET)))


# filter by civitai tags
client.models().filter_by_civi_tags('anime')

# filter by nsfw
client.models().filter_by_nsfw(False)  # or True

# sort by civitai download
client.models().sort_by_civitai_download()

# chain filters
client.models().\
    filter_by_type(ModelType.CHECKPOINT).\
    filter_by_nsfw(False).\
    filter_by_civitai_tags('anime')
```

### ControlNet QRCode

[controlnet_qrcode.py](./examples/controlnet_qrcode.py)

```python
import os

from novita_client import *

# get your api key refer to https://docs.novita.ai/get-started/
client = NovitaClient(os.getenv('NOVITA_API_KEY'), os.getenv('NOVITA_API_URI', None))

controlnet_model = client.models().filter_by_type(ModelType.CONTROLNET).get_by_name("control_v1p_sd15_qrcode_monster_v2")
if controlnet_model is None:
    raise Exception("controlnet model not found")

req = Txt2ImgRequest(
    prompt="a beautify butterfly in the colorful flowers, best quality, best details, masterpiece",
    sampler_name=Samplers.DPMPP_M_KARRAS,
    width=512,
    height=512,
    steps=30,
    controlnet_units=[
        ControlnetUnit(
            input_image=read_image_to_base64(os.path.join(os.path.abspath(os.path.dirname(__file__)), "fixtures/qrcode.png")),
            control_mode=ControlNetMode.BALANCED,
            model=controlnet_model.sd_name,
            module=ControlNetPreprocessor.NULL,
            resize_mode=ControlNetResizeMode.JUST_RESIZE,
            weight=2.0,
        )
    ]
)

res = client.sync_txt2img(req)
if res.data.status != ProgressResponseStatusCode.SUCCESSFUL:
    raise Exception('Failed to generate image with error: ' +
                    res.data.failed_reason)

save_image(res.data.imgs_bytes[0], "qrcode-art.png")
```

### Txt2Img with Hires.Fix

[txt2img_with_hiresfix.py](./examples/txt2img_with_hiresfix.py)

```python
import os

from novita_client import *

client = NovitaClient(os.getenv('NOVITA_API_KEY'), os.getenv('NOVITA_API_URI', None))
req = Txt2ImgRequest(
    model_name='dreamshaper_8_93211.safetensors',
    prompt='a dog flying in the sky',
    width=512,
    height=512,
    batch_size=1,
    cfg_scale=7.5,
    sampler_name=Samplers.EULER_A,
    enable_hr=True,
    hr_scale=2.0
)

res = client.sync_txt2img(req)
if res.data.status != ProgressResponseStatusCode.SUCCESSFUL:
    raise Exception('Failed to generate image with error: ' +
                    res.data.failed_reason)

save_image(res.data.imgs_bytes[0], "txt2img-hiresfix-1024.png")
```

### SDXL Refiner

[sdxl_refiner.py](./txt2img_with_refiner.py)

```python
import os

from novita_client import *

client = NovitaClient(os.getenv('NOVITA_API_KEY'), os.getenv('NOVITA_API_URI', None))
req = Txt2ImgRequest(
    model_name='sd_xl_base_1.0.safetensors',
    prompt='a dog flying in the sky',
    width=1024,
    height=1024,
    batch_size=1,
    cfg_scale=7.5,
    sampler_name=Samplers.EULER_A,
    sd_refiner=Refiner(
        checkpoint='sd_xl_refiner_1.0.safetensors',
        switch_at=0.5,
    ))

res = client.sync_txt2img(req)
if res.data.status != ProgressResponseStatusCode.SUCCESSFUL:
    raise Exception('Failed to generate image with error: ' +
                    res.data.failed_reason)

save_image(res.data.imgs_bytes[0], "txt2img-refiner.png")
```

## Testing

```
export NOVITA_API_KEY=<YOUR_API_KEY>

python -m pytest
```
