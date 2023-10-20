from novita_client import *
from novita_client.utils import save_image, read_image_to_base64, base64_to_image
import os
from PIL import Image
import io


def test_cleanup():
    client = NovitaClient(os.getenv('NOVITA_API_KEY'))

    res = client.cleanup(
        image="https://raw.githubusercontent.com/CompVis/latent-diffusion/main/data/inpainting_examples/overture-creations-5sI6fQgYIuo.png",
        mask="https://raw.githubusercontent.com/CompVis/latent-diffusion/main/data/inpainting_examples/overture-creations-5sI6fQgYIuo_mask.png"
    )

    assert (res.image_file is not None)

    test_path = os.path.join(os.path.abspath(
        os.path.dirname(__name__)), "tests/data")
    if not os.path.exists(test_path):
        os.makedirs(test_path)
    base64_to_image(res.image_file).save(os.path.join(test_path, "test_cleanup.png"))


def test_outpainting():
    client = NovitaClient(os.getenv('NOVITA_API_KEY'))

    res = client.outpainting(
        image="https://raw.githubusercontent.com/CompVis/latent-diffusion/main/data/inpainting_examples/overture-creations-5sI6fQgYIuo.png",
        width=910,
        height=512,
        center_x=0,
        center_y=0,
    )

    assert (res.image_file is not None)

    test_path = os.path.join(os.path.abspath(
        os.path.dirname(__name__)), "tests/data")

    if not os.path.exists(test_path):
        os.makedirs(test_path)

    base64_to_image(res.image_file).save(os.path.join(test_path, f"test_outpainting.{res.image_type}"))


def test_remove_background():
    client = NovitaClient(os.getenv('NOVITA_API_KEY'))

    res = client.remove_background(
        image="https://raw.githubusercontent.com/CompVis/latent-diffusion/main/data/inpainting_examples/overture-creations-5sI6fQgYIuo.png",
    )

    assert (res.image_file is not None)

    test_path = os.path.join(os.path.abspath(
        os.path.dirname(__name__)), "tests/data")

    if not os.path.exists(test_path):
        os.makedirs(test_path)

    base64_to_image(res.image_file).save(os.path.join(test_path, f"test_remove_background.{res.image_type}"))


def test_remove_text():
    client = NovitaClient(os.getenv('NOVITA_API_KEY'))

    res = client.remove_text(
        image="https://images.uiiiuiii.com/wp-content/uploads/2023/07/i-banner-20230714-1.jpg"
    )

    assert (res.image_file is not None)

    test_path = os.path.join(os.path.abspath(
        os.path.dirname(__name__)), "tests/data")

    if not os.path.exists(test_path):
        os.makedirs(test_path)

    base64_to_image(res.image_file).save(os.path.join(test_path, f"test_remove_text.{res.image_type}"))


def test_reimagine():
    client = NovitaClient(os.getenv('NOVITA_API_KEY'))

    res = client.reimagine(
        image="https://madera.objects.liquidweb.services/photos/20371-yosemite-may-yosemite-falls-waterfalls-cooks-meadow-spring-2023-Rectangle-600x400.jpg",
    )

    assert (res.image_file is not None)

    test_path = os.path.join(os.path.abspath(
        os.path.dirname(__name__)), "tests/data")

    if res.image_type == "gif":
        return

    base64_to_image(res.image_file).save(os.path.join(os.path.abspath(test_path), f"test_reimagine.{res.image_type}"))


def test_doodle():
    client = NovitaClient(os.getenv('NOVITA_API_KEY'))

    res = client.doodle(
        image="https://img.freepik.com/premium-photo/close-up-dogs-face-with-big-smile-generative-ai_900101-62851.jpg",
        prompt="A cute dog",
    )

    assert (res.image_file is not None)

    test_path = os.path.join(os.path.abspath(
        os.path.dirname(__name__)), "tests/data")

    base64_to_image(res.image_file).save(os.path.join(os.path.abspath(test_path), f"test_doodle.{res.image_type}"))


def test_mixpose():
    client = NovitaClient(os.getenv('NOVITA_API_KEY'))
    
    res = client.mixpose(
        image="https://image.uniqlo.com/UQ/ST3/my/imagesgoods/455359/item/mygoods_23_455359.jpg?width=494",
        pose_image="https://image.uniqlo.com/UQ/ST3/ca/imagesgoods/455359/item/cagoods_02_455359.jpg?width=494",
    )

    assert (res.image_file is not None)
    
    test_path = os.path.join(os.path.abspath(
        os.path.dirname(__name__)), "tests/data")
    
    base64_to_image(res.image_file).save(os.path.join(os.path.abspath(test_path), f"test_mixpose.{res.image_type}"))


def test_replace_background():
    client = NovitaClient(os.getenv('NOVITA_API_KEY'))

    res = client.replace_background(
        image="https://raw.githubusercontent.com/CompVis/latent-diffusion/main/data/inpainting_examples/overture-creations-5sI6fQgYIuo.png",
        prompt="beautify beach"
    )

    assert (res.image_file is not None)

    test_path = os.path.join(os.path.abspath(
        os.path.dirname(__name__)), "tests/data")

    base64_to_image(res.image_file).save(os.path.join(os.path.abspath(test_path), f"test_replace_background.{res.image_type}"))

def test_replace_sky():
    client = NovitaClient(os.getenv('NOVITA_API_KEY'))

    res = client.replace_sky(
        image="https://dynamic-media-cdn.tripadvisor.com/media/photo-o/17/16/a6/88/con-la-primavera-in-giappone.jpg?w=700",
        sky="galaxy"
    )

    assert (res.image_file is not None)

    test_path = os.path.join(os.path.abspath(
        os.path.dirname(__name__)), "tests/data")

    base64_to_image(res.image_file).save(os.path.join(os.path.abspath(test_path), f"test_replace_sky.png"))

def test_replace_object():
    client = NovitaClient(os.getenv('NOVITA_API_KEY'))

    res = client.replace_object(
        image="https://raw.githubusercontent.com/CompVis/latent-diffusion/main/data/inpainting_examples/overture-creations-5sI6fQgYIuo.png",
        object_prompt="a dog",
        prompt="a cute cat"
    )

    assert (res.image_file is not None)

    test_path = os.path.join(os.path.abspath(
        os.path.dirname(__name__)), "tests/data")

    base64_to_image(res.image_file).save(os.path.join(os.path.abspath(test_path), f"test_replace_object.{res.image_type}"))

def test_create_tile():
    client = NovitaClient(os.getenv('NOVITA_API_KEY'))

    res = client.create_tile(
        prompt="a cute flower",
    )

    assert (res.image_file is not None)

    test_path = os.path.join(os.path.abspath(
        os.path.dirname(__name__)), "tests/data")
    
    base64_to_image(res.image_file).save(os.path.join(os.path.abspath(test_path), f"test_tile.{res.image_type}"))

def test_merge_face():
    client = NovitaClient(os.getenv('NOVITA_API_KEY'))
    
    res = client.merge_face(
        image="https://www.wgm8.com/wp-content/uploads/2016/06/images_wgm_online-only_Gaming_2016_30-06-16-1.jpg",
        face_image="https://p7.itc.cn/images01/20220220/285669b5682540a8a307a87d8745f530.jpeg",
    )

    assert (res.image_file is not None)
    
    test_path = os.path.join(os.path.abspath(
        os.path.dirname(__name__)), "tests/data")
    
    base64_to_image(res.image_file).save(os.path.join(os.path.abspath(test_path), f"test_merge_face.{res.image_type}"))