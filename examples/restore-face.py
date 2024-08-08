import os

from novita_client import NovitaClient
from novita_client.utils import base64_to_image

client = NovitaClient(os.getenv('NOVITA_API_KEY'), os.getenv('NOVITA_API_URI', None))
res = client.restore_face(
    image="https://xintao-gfpgan.hf.space/file=/home/user/app/lincoln.jpg",
    fidelity=0.5,
    enterprise_plan=True
    )
base64_to_image(res.image_file).save("./restore_face.png")