import os

from novita_client import NovitaClient
from novita_client.utils import base64_to_image

client = NovitaClient(os.getenv('NOVITA_API_KEY'), os.getenv('NOVITA_API_URI', None))
res = client.merge_face(
    image="https://www.wgm8.com/wp-content/uploads/2016/06/images_wgm_online-only_Gaming_2016_30-06-16-1.jpg",
    face_image="https://p7.itc.cn/images01/20220220/285669b5682540a8a307a87d8745f530.jpeg",
)

base64_to_image(res.image_file).save("./merge_face.png")
