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
