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
