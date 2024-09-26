import pytest
from novita_client import *


@pytest.mark.dependency()
def test_txt2img_v3():
    client = NovitaClient(os.getenv('NOVITA_API_KEY'), os.getenv('NOVITA_API_URI', None))
    res = client.txt2img_v3(
        model_name='dreamshaper_8_93211.safetensors',
        prompt="a cute girl",
        width=384,
        height=512,
        image_num=1,
        guidance_scale=7.5,
        seed=12345,
        sampler_name=Samplers.EULER_A
    )
    assert (len(res.images) == 1)
    test_path = os.path.join(os.path.abspath(
        os.path.dirname(__name__)), "tests/data")
    if not os.path.exists(test_path):
        os.makedirs(test_path)
    base64_to_image(res.images_encoded[0]).save(os.path.join(
        test_path, 'test_txt2img_v3.png'))


@pytest.mark.dependency(depends=["test_txt2img_v3"])
def test_img2img_v3():
    client = NovitaClient(os.getenv('NOVITA_API_KEY'), os.getenv('NOVITA_API_URI', None))
    init_image = os.path.join(os.path.abspath(
        os.path.dirname(__name__)), "tests/data/test_txt2img_v3.png")
    res = client.img2img_v3(
        model_name='dreamshaper_8_93211.safetensors',
        prompt="a cute girl",
        width=384,
        height=512,
        image_num=1,
        guidance_scale=7.5,
        seed=12345,
        steps=20,
        sampler_name=Samplers.EULER_A,
        input_image=init_image
    )
    assert (len(res.images) == 1)
