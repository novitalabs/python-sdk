from novita_client import *
from novita_client.utils import save_image, read_image_to_base64, base64_to_image
import os
from PIL import Image
import random


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


def test_cleanup():
    client = NovitaClient(os.getenv('NOVITA_API_KEY'), os.getenv('NOVITA_API_URI', None))

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
    client = NovitaClient(os.getenv('NOVITA_API_KEY'), os.getenv('NOVITA_API_URI', None))

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
    client = NovitaClient(os.getenv('NOVITA_API_KEY'), os.getenv('NOVITA_API_URI', None))

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
    client = NovitaClient(os.getenv('NOVITA_API_KEY'), os.getenv('NOVITA_API_URI', None))

    res = client.remove_text(
        image="https://cf-images.novitai.com/sdk-cases/remove_text_example_1.jpg/public"
    )

    assert (res.image_file is not None)

    test_path = os.path.join(os.path.abspath(
        os.path.dirname(__name__)), "tests/data")

    if not os.path.exists(test_path):
        os.makedirs(test_path)

    base64_to_image(res.image_file).save(os.path.join(test_path, f"test_remove_text.{res.image_type}"))


def test_reimagine():
    client = NovitaClient(os.getenv('NOVITA_API_KEY'), os.getenv('NOVITA_API_URI', None))

    res = client.reimagine(
        image="https://madera.objects.liquidweb.services/photos/20371-yosemite-may-yosemite-falls-waterfalls-cooks-meadow-spring-2023-Rectangle-600x400.jpg",
    )

    assert (res.image_file is not None)

    test_path = os.path.join(os.path.abspath(
        os.path.dirname(__name__)), "tests/data")

    if res.image_type == "gif":
        return

    base64_to_image(res.image_file).save(os.path.join(os.path.abspath(test_path), f"test_reimagine.{res.image_type}"))

def test_replace_background():
    client = NovitaClient(os.getenv('NOVITA_API_KEY'), os.getenv('NOVITA_API_URI', None))

    res = client.replace_background(
        image="https://raw.githubusercontent.com/CompVis/latent-diffusion/main/data/inpainting_examples/overture-creations-5sI6fQgYIuo.png",
        prompt="beautify beach"
    )

    assert (res.image_file is not None)

    test_path = os.path.join(os.path.abspath(
        os.path.dirname(__name__)), "tests/data")

    base64_to_image(res.image_file).save(os.path.join(os.path.abspath(test_path), f"test_replace_background.{res.image_type}"))

def test_merge_face():
    client = NovitaClient(os.getenv('NOVITA_API_KEY'), os.getenv('NOVITA_API_URI', None))

    res = client.merge_face(
        image="https://cf-images.novitai.com/sdk-cases/merge_face_example_1_1.png/public",
        face_image="https://cf-images.novitai.com/sdk-cases/merge_face_example_1_2.png/public",
    )

    assert (res.image_file is not None)

    test_path = os.path.join(os.path.abspath(
        os.path.dirname(__name__)), "tests/data")

    base64_to_image(res.image_file).save(os.path.join(os.path.abspath(test_path), f"test_merge_face.{res.image_type}"))
