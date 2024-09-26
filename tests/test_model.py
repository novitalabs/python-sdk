from novita_client import *
import os


def test_model_api():
    client = NovitaClient(os.getenv('NOVITA_API_KEY'), os.getenv('NOVITA_API_URI', None))
    models = client.models_v3()
    assert all([m.is_nsfw is True for m in models.filter_by_nsfw(True)])
    assert all([m.is_nsfw is False for m in models.filter_by_nsfw(False)])

    assert len(models. \
        filter_by_type(ModelType.LORA). \
        filter_by_nsfw(False)) > 0

    assert len(models.filter_by_type(ModelType.CHECKPOINT)) > 0
    assert len(models.filter_by_type(ModelType.LORA)) > 0
    assert len(models.filter_by_type(ModelType.TEXT_INVERSION)) > 0
    assert len(models.filter_by_type(ModelType.CONTROLNET)) > 0
