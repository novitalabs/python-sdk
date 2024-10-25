# Novita AI Python SDK

This SDK is based on the official [API documentation](https://docs.novita.ai/).

**Join our discord server for help:**

[![](https://dcbadge.vercel.app/api/server/Mqx7nWYzDF)](https://discord.com/invite/Mqx7nWYzDF)

## Installation

```bash
pip install novita-client
```

## Examples

- [fine tune example](https://colab.research.google.com/drive/1j_ii9TN67nuauvc3PiauwZnC2lT62tGF?usp=sharing)
- [cleanup](./examples/cleanup.py)
- [controlnet](./examples/controlnet.py)
- [img2img](./examples/img2img.py)
- [img2video](./examples/img2video.py)
- [inpainting](./examples/inpainting.py)
- [instantid](./examples/instantid.py)
- [merge-face](./examples/merge-face.py)
- [model-search](./examples/model-search.py)
- [outpainting](./examples/outpainting.py)
- [reimagine](./examples/reimagine.py)
- [remove-background](./examples/remove-background.py)
- [remove-text](./examples/remove-text.py)
- [replace-background](./examples/replace-background.py)
- [restore-face](./examples/restore-face.py)
- [txt2img-with-hiresfix](./examples/txt2img-with-hiresfix.py)
- [txt2img-with-lora](./examples/txt2img-with-lora.py)
- [txt2img-with-refiner](./examples/txt2img-with-refiner.py)
- [txt2video](./examples/txt2video.py)
## Code Examples

### cleanup
```python
import os

from novita_client import NovitaClient
from novita_client.utils import base64_to_image

client = NovitaClient(os.getenv('NOVITA_API_KEY'), os.getenv('NOVITA_API_URI', None))
res = client.cleanup(
    image="https://raw.githubusercontent.com/CompVis/latent-diffusion/main/data/inpainting_examples/overture-creations-5sI6fQgYIuo.png",
    mask="https://raw.githubusercontent.com/CompVis/latent-diffusion/main/data/inpainting_examples/overture-creations-5sI6fQgYIuo_mask.png"
)

base64_to_image(res.image_file).save("./cleanup.png")
```

### controlnet
```python
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
```

### img2img
```python
import pdb
import os

from novita_client import NovitaClient, Img2ImgV3ControlNetUnit, ControlNetPreprocessor, Img2ImgV3Embedding
from novita_client.utils import base64_to_image, input_image_to_pil

client = NovitaClient(os.getenv('NOVITA_API_KEY'), os.getenv('NOVITA_API_URI', None))
res = client.img2img_v3(
    model_name="MeinaHentai_V5.safetensors",
    steps=30,
    height=512,
    width=512,
    input_image="https://img.freepik.com/premium-photo/close-up-dogs-face-with-big-smile-generative-ai_900101-62851.jpg",
    prompt="1 cute dog",
    strength=0.5,
    guidance_scale=7,
    embeddings=[Img2ImgV3Embedding(model_name=_) for _ in [
        "bad-image-v2-39000",
        "verybadimagenegative_v1.3_21434",
        "BadDream_53202",
        "badhandv4_16755",
        "easynegative_8955.safetensors"]],
    seed=-1,
    sampler_name="DPM++ 2M Karras",
    clip_skip=2,
    # controlnet_units=[Img2ImgV3ControlNetUnit(
    #     model_name="control_v11f1p_sd15_depth",
    #     preprocessor="depth",
    #     image_base64="./20240309-003206.jpeg",
    #     strength=1.0
    # )]
)

base64_to_image(res.images_encoded[0]).save("./img2img.png")
```

### img2video
```python
import os

from novita_client import NovitaClient
from novita_client.utils import base64_to_image

client = NovitaClient(os.getenv('NOVITA_API_KEY'), os.getenv('NOVITA_API_URNOVITA_API_URII', None))
res = client.img2video(
    model_name="SVD-XT",
    steps=30,
    frames_num=25,
    image="https://replicate.delivery/pbxt/JvLi9smWKKDfQpylBYosqQRfPKZPntuAziesp0VuPjidq61n/rocket.png",
    enable_frame_interpolation=True
)


with open("test.mp4", "wb") as f:
    f.write(res.video_bytes[0])
```

### inpainting
```python
import os
import base64
from novita_client import NovitaClient
from novita_client.utils import base64_to_image

client = NovitaClient(os.getenv('NOVITA_API_KEY'), os.getenv('NOVITA_API_URI', None))
res = client.inpainting(
    model_name = "realisticVisionV40_v40VAE-inpainting_81543.safetensors",
    image="https://raw.githubusercontent.com/CompVis/latent-diffusion/main/data/inpainting_examples/overture-creations-5sI6fQgYIuo.png",
    mask="https://raw.githubusercontent.com/CompVis/latent-diffusion/main/data/inpainting_examples/overture-creations-5sI6fQgYIuo.png",
    seed=1,
    guidance_scale=15,
    steps = 20,
    image_num = 4,
    prompt = "black rabbit",
    negative_prompt = "white rabbit",
    sampler_name = "Euler a",
    inpainting_full_res = 1,
    inpainting_full_res_padding = 32,
    inpainting_mask_invert = 0,
    initial_noise_multiplier = 1,
    mask_blur = 1,
    clip_skip = 1,
    strength = 0.85,
)
with open("result/result_image/inpaintingsdk.jpeg", "wb") as image_file:
    image_file.write(base64.b64decode(res.images_encoded[0]))```

### instantid
```python

import os
from novita_client import NovitaClient, InstantIDControlnetUnit
import base64



if __name__ == '__main__':
	client = NovitaClient(os.getenv('NOVITA_API_KEY'), os.getenv('NOVITA_API_URI', None))

	res = client.instant_id(
		model_name="sdxlUnstableDiffusers_v8HEAVENSWRATH_133813.safetensors",
		face_images=[
			"https://raw.githubusercontent.com/InstantID/InstantID/main/examples/yann-lecun_resize.jpg",
		],
		prompt="Flat illustration, a Chinese a man, ancient style, wearing a red cloth, smile face, white skin, clean background, fireworks blooming, red lanterns",
		negative_prompt="(lowres, low quality, worst quality:1.2), (text:1.2), watermark, (frame:1.2), deformed, ugly, deformed eyes, blur, out of focus, blurry, deformed cat, deformed, photo, anthropomorphic cat, monochrome, photo, pet collar, gun, weapon, blue, 3d, drones, drone, buildings in background, green",
		id_strength=0.8,
		adapter_strength=0.8,
		steps=20,
		seed=42,
		width=1024,
		height=1024,
		controlnets=[
			InstantIDControlnetUnit(
				model_name='controlnet-openpose-sdxl-1.0',
				strength=0.4,
				preprocessor='openpose',
			),
			InstantIDControlnetUnit(
				model_name='controlnet-canny-sdxl-1.0',
				strength=0.3,
				preprocessor='canny',
			),
		],
		response_image_type='jpeg',
		enterprise_plan=False,
	)

	print('res:', res)

	if hasattr(res, 'images_encoded'):
		with open(f"instantid.png", "wb") as f:
			f.write(base64.b64decode(res.images_encoded[0]))
```

### merge-face
```python
import os

from novita_client import NovitaClient
from novita_client.utils import base64_to_image

client = NovitaClient(os.getenv('NOVITA_API_KEY'), os.getenv('NOVITA_API_URI', None))
res = client.merge_face(
    image="https://toppng.com/uploads/preview/cut-out-people-png-personas-en-formato-11563277290kozkuzsos5.png",
    face_image="https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcQDy7sXtuvCNUQoQZvTbLRbX6qK9_kP3PlQfg&s",
    enterprise_plan=False,
)

base64_to_image(res.image_file).save("./merge_face.png")
```

### model-search
```python
#!/usr/bin/env python
# -*- coding: UTF-8 -*-

from novita_client import NovitaClient, ModelType
# get your api key refer to https://docs.novita.ai/get-started/
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

### outpainting
```python
import os

from novita_client import NovitaClient
from novita_client.utils import base64_to_image

client = NovitaClient(os.getenv('NOVITA_API_KEY'), os.getenv('NOVITA_API_URI', None))
res = client.outpainting(
    image="https://raw.githubusercontent.com/CompVis/latent-diffusion/main/data/inpainting_examples/overture-creations-5sI6fQgYIuo.png",
    width=910,
    height=512,
    center_x=0,
    center_y=0,
)
base64_to_image(res.image_file).save("./outpainting.png")
```

### reimagine
```python
import os

from novita_client import NovitaClient
from novita_client.utils import base64_to_image

client = NovitaClient(os.getenv('NOVITA_API_KEY'), os.getenv('NOVITA_API_URI', None))
res = client.reimagine(
    image="/home/anyisalin/develop/novita-client-python/examples/doodle-generated.png"
)

base64_to_image(res.image_file).save("./reimagine.png")
```

### remove-background
```python
import os

from novita_client import NovitaClient
from novita_client.utils import base64_to_image

client = NovitaClient(os.getenv('NOVITA_API_KEY'), os.getenv('NOVITA_API_URI', None))
res = client.remove_background(
    image="https://raw.githubusercontent.com/CompVis/latent-diffusion/main/data/inpainting_examples/overture-creations-5sI6fQgYIuo.png",
)
base64_to_image(res.image_file).save("./remove_background.png")
```

### remove-text
```python
import os

from novita_client import NovitaClient
from novita_client.utils import base64_to_image

client = NovitaClient(os.getenv('NOVITA_API_KEY'), os.getenv('NOVITA_API_URI', None))
res = client.remove_text(
    image="https://images.uiiiuiii.com/wp-content/uploads/2023/07/i-banner-20230714-1.jpg"
)

base64_to_image(res.image_file).save("./remove_text.png")
```

### replace-background
```python
import os

from novita_client import NovitaClient
from novita_client.utils import base64_to_image

client = NovitaClient(os.getenv('NOVITA_API_KEY'), os.getenv('NOVITA_API_URI', None))
res = client.replace_background(
    image="./telegram-cloud-photo-size-2-5408823814353177899-y.jpg",
    prompt="in living room, Christmas tree",
)
base64_to_image(res.image_file).save("./replace_background.png")
```

### restore-face
```python
import os

from novita_client import NovitaClient
from novita_client.utils import base64_to_image

client = NovitaClient(os.getenv('NOVITA_API_KEY'), os.getenv('NOVITA_API_URI', None))
res = client.restore_face(
    image="https://xintao-gfpgan.hf.space/file=/home/user/app/lincoln.jpg",
    fidelity=0.5,#The fidelity of the original portrait, on a scale from 0 to 1.0, with higher scores indicating better fidelity. Range: [0, 1]
    enterprise_plan=False
    )
base64_to_image(res.image_file).save("./restore_face.png")
```

### txt2img-with-hiresfix
```python
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
```

### txt2img-with-lora
```python
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
```

### txt2img-with-refiner
```python
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
```

### txt2video
```python
import os

from novita_client import NovitaClient
from novita_client.utils import save_image

client = NovitaClient(os.getenv('NOVITA_API_KEY'), os.getenv('NOVITA_API_URI', None))
res = client.txt2video(
        model_name = "dreamshaper_8_93211.safetensors",
        prompts = [{
                    "prompt": "A girl, baby, portrait, 5 years old",
                    "frames": 16,},
                    {
                    "prompt": "A girl, child, portrait, 10 years old",
                    "frames": 16,
                    }
                    ],
        steps = 20,
        guidance_scale = 10,
        height = 512,
        width = 768,
        clip_skip = 4,
        negative_prompt = "a rainy day",
        response_video_type = "mp4",
    )
save_image(res.video_bytes[0], 'output.mp4')
```
