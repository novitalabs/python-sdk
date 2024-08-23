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
