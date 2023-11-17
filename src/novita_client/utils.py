#!/usr/bin/env python
# -*- coding: UTF-8 -*-

from multiprocessing.pool import ThreadPool

import base64
import logging

import requests
from io import BytesIO

from . import settings
from .proto import *
from PIL import Image, ImageOps

logger = logging.getLogger(__name__)


def batch_download_images(image_links):
    def _download(image_link):
        attempts = settings.DEFAULT_DOWNLOAD_IMAGE_ATTEMPTS
        while attempts > 0:
            try:
                response = requests.get(
                    image_link, timeout=settings.DEFAULT_DOWNLOAD_ONE_IMAGE_TIMEOUT)
                return response.content
            except Exception:
                logger.warning("Failed to download image, retrying...")
            attempts -= 1
        return None

    pool = ThreadPool()
    applied = []
    for img_url in image_links:
        applied.append(pool.apply_async(_download, (img_url, )))
    ret = [r.get() for r in applied]
    return [_ for _ in ret if _ is not None]


def save_image(image_bytes, name):
    with open(name, "wb") as f:
        f.write(image_bytes)


def read_image(name):
    with open(name, "rb") as f:
        return f.read()


def read_image_to_base64(name):
    with open(name, "rb") as f:
        return base64.b64encode(f.read()).decode('utf-8')


def image_to_base64(image: Image.Image, format=None) -> str:
    buffered = BytesIO()
    if format is None:
        format = image.format
        if not format:
            format = "PNG"
    image.save(buffered, format)
    return base64.b64encode(buffered.getvalue()).decode('ascii')


def base64_to_image(base64_image: str) -> Image:
    # convert base64 string to image
    image = Image.open(BytesIO(base64.b64decode(base64_image)))
    image = ImageOps.exif_transpose(image)
    return image.convert("RGB")


def add_lora_to_prompt(prompt: str, lora_name: str, weight: float = 1.0) -> str:
    prompt_split = [s.strip() for s in prompt.split(",")]
    ret = []
    replace = False
    for prompt_chunk in prompt_split:
        if prompt_chunk.startswith("<lora:{}".format(lora_name)):
            ret.append("<lora:{}:{}>".format(lora_name, weight))
            replace = True
        else:
            ret.append(prompt_chunk)
    if not replace:
        ret.append("<lora:{}:{}>".format(lora_name, weight))
    return ", ".join(ret)


def input_image_to_pil(image) -> Image.Image:
    def _convert_to_pil(image):
        if isinstance(image, str):
            if os.path.exists(image):
                return Image.open(BytesIO(read_image(image)))

            if image.startswith("http") or image.startswith("https"):
                return Image.open(BytesIO(batch_download_images([image])[0]))

        if isinstance(image, os.PathLike):
            return Image.open(BytesIO(read_image(str(image))))

        if isinstance(image, Image.Image):
            return image
        raise ValueError("Unknown image type: {}".format(type(image)))

    return ImageOps.exif_transpose(_convert_to_pil(image))


def input_image_to_base64(image) -> str:
    if isinstance(image, str):
        if os.path.exists(image):
            return read_image_to_base64(image)

        if image.startswith("http") or image.startswith("https"):
            return base64.b64encode(batch_download_images([image])[0]).decode('ascii')

        # assume it is a base64 string
        return image

    if isinstance(image, os.PathLike):
        return read_image_to_base64(str(image))

    if isinstance(image, Image.Image):
        return image_to_base64(image)
    raise ValueError("Unknown image type: {}".format(type(image)))
