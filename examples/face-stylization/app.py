import gradio as gr
from novita_client import *
import logging
import traceback


logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(filename)s(%(lineno)d) %(message)s')


first_stage_activication_words = "a ohwx"
first_stage_lora_scale = 0.3
second_stage_activication_words = "a closeup photo of ohwx"
second_stage_lora_scale = 1.0

suggestion_checkpoints = [
    "dreamshaper_8_93211.safetensors",
    "epicrealism_pureEvolutionV5_97793.safetensors",
]
base_checkpoints = ["epicrealism_naturalSin_121250", "v1-5-pruned-emaonly"]


get_local_storage = """
    function() {
      globalThis.setStorage = (key, value)=>{
        localStorage.setItem(key, JSON.stringify(value))
      }
       globalThis.getStorage = (key, value)=>{
        return JSON.parse(localStorage.getItem(key))
      }

       const novita_key =  getStorage('novita_key')
       return [novita_key];
      }
    """


def get_noviata_client(novita_key):
    client = NovitaClient(novita_key, os.getenv('NOVITA_API_URI', None))
    client.set_extra_headers({"User-Agent": "stylization-playground"})
    return client


def create_ui():
    with gr.Blocks() as demo:
        gr.Markdown("""## Novita.AI - Face Stylization Playground
                       ### Get Novita.AI API Key from [novita.ai](https://novita.ai)
                    """)
        with gr.Row():
            with gr.Column(scale=1):
                novita_key = gr.Textbox(value="", label="Novita.AI API KEY (store in broweser)", placeholder="novita.ai api key", type="password")
            with gr.Column(scale=1):
                user_balance = gr.Textbox(label="User Balance", value="0.0")

        with gr.Tab(label="Training"):
            with gr.Row():
                with gr.Column(scale=1):
                    base_model = gr.Dropdown(choices=base_checkpoints, label="Base Model", value=base_checkpoints[0])
                    geneder = gr.Radio(choices=["man", "woman"], value="man", label="Geneder")
                    training_name = gr.Text(label="Training Name", placeholder="training name", elem_id="training_name", value="my-face-001")
                    max_train_steps = gr.Slider(minimum=200, maximum=4000, step=1, label="Max Train Steps", value=2000)
                    training_images = gr.File(file_types=["image"], file_count="multiple", label="6-10 face images.")
                    training_button = gr.Button(value="Train")
                    training_payload = gr.JSON(label="Training Payload, POST /v3/training/subject")
                with gr.Column(scale=1):
                    training_refresh_button = gr.Button(value="Refresh Training Status")
                    training_refresh_json = gr.JSON()

                def train(novita_key, gender, base_model, training_name, max_train_steps, training_images):
                    training_images = [_.name for _ in training_images]
                    try:
                        get_noviata_client(novita_key).create_training_subject(
                            base_model=base_model,
                            name=training_name,
                            instance_prompt=f"a closeup photo of ohwx person",
                            class_prompt="person",
                            max_train_steps=max_train_steps,
                            images=training_images,
                            components=FACE_TRAINING_DEFAULT_COMPONENTS,
                            learning_rate=3e-4,
                            seed=None,
                            lr_scheduler='cosine_with_restarts',
                            with_prior_preservation=True,
                            prior_loss_weight=1.0,
                            lora_r=32,
                            lora_alpha=32,
                            lora_text_encoder_r=32,
                            lora_text_encoder_alpha=32,
                        )

                        payload = dict(
                            name=training_name,
                            base_model=base_model,
                            image_dataset_items=["....assets_ids, please manually upload to novita.ai"],
                            expert_setting=TrainingExpertSetting(
                                instance_prompt=f"a closeup photo of ohwx person",
                                class_prompt="person",
                                max_train_steps=max_train_steps,
                                learning_rate="8e-5",
                                seed=None,
                                lr_scheduler='cosine_with_restarts',
                                with_prior_preservation=True,
                                prior_loss_weight=1.0,
                                lora_r=32,
                                lora_alpha=32,
                                lora_text_encoder_r=32,
                                lora_text_encoder_alpha=32,
                            ),
                            components=[_.to_dict() for _ in FACE_TRAINING_DEFAULT_COMPONENTS],
                        )
                    except Exception as e:
                        logging.error(e)
                        raise gr.Error(traceback.format_exc())

                    return gr.update(value=get_noviata_client(novita_key).list_training().sort_by_created_at()), payload

                training_refresh_button.click(
                    inputs=[novita_key],
                    outputs=training_refresh_json,
                    fn=lambda novita_key: gr.update(value=get_noviata_client(novita_key).list_training().sort_by_created_at())
                )

                training_button.click(
                    inputs=[novita_key, geneder, base_model, training_name, max_train_steps, training_images],
                    outputs=[training_refresh_json, training_payload],
                    fn=train
                )

        with gr.Tab(label="Inferencing"):
            with gr.Row():
                with gr.Column(scale=1):
                    style_prompt = gr.TextArea(lines=3, label="Style Prompt")
                    style_negative_prompt = gr.TextArea(lines=3, label="Style Negative Prompt")
                    inference_geneder = gr.Radio(choices=["man", "woman"], value="man", label="Gender")
                    style_model = gr.Dropdown(choices=suggestion_checkpoints, label="Style Model")
                    style_lora = gr.Dropdown(choices=[], label="Style LoRA", type="index")
                    _hide_lora_training_response = gr.JSON(visible=False)
                    # style_lora_scale = gr.Slider(minimum=0.0, maximum=1.0, step=0.01, label="Style LoRA Scale", value=1.0)
                    style_height = gr.Slider(minimum=1, maximum=1024, step=1, label="Style Height", value=512)
                    style_width = gr.Slider(minimum=1, maximum=1024, step=1, label="Style Width", value=512)
                    style_method = gr.Radio(choices=["txt2img", "controlnet-depth", "controlnet-pose", "controlnet-canny"], label="Style Method")
                    style_reference_image = gr.Image(label="Style Reference Image", height=512)

                with gr.Column(scale=1):
                    inference_refresh_button = gr.Button(value="Refresh Style LoRA")
                    generate_button = gr.Button(value="Generate")
                    num_images = gr.Slider(minimum=1, maximum=10, step=1, label="Num Images", value=1)
                    gallery = gr.Gallery(label="Gallery", height="auto", object_fit="scale-down")

                    def inference_refresh_button_fn(novita_key):
                        # trained_loras_models = [_.name for _ in get_noviata_client(novita_key).models_v3(refresh=True).filter_by_type("lora").filter_by_visibility("private")]
                        serving_models = [_.models[0].model_name for _ in get_noviata_client(novita_key).list_training().filter_by_model_status("SERVING")]
                        serving_models_labels = [_.task_name for _ in get_noviata_client(novita_key).list_training().filter_by_model_status("SERVING")]
                        return gr.update(choices=serving_models_labels), gr.update(value=serving_models)

                    inference_refresh_button.click(
                        inputs=[novita_key],
                        outputs=[style_lora, _hide_lora_training_response],
                        fn=inference_refresh_button_fn
                    )

                    templates = [
                        {
                            "style_prompt": "(masterpiece), (extremely intricate:1.3), (realistic), portrait of a person, the most handsome in the world, (medieval armor), metal reflections, upper body, outdoors, intense sunlight, far away castle, professional photograph of a stunning person detailed, sharp focus, dramatic, award winning, cinematic lighting, octane render unreal engine, volumetrics dtx, (film grain, blurry background, blurry foreground, bokeh, depth of field, sunset, motion blur:1.3), chainmail",
                            "style_negative_prompt": "BadDream_53202, UnrealisticDream_53204",
                            "style_model": "dreamshaper_8_93211.safetensors",
                            "style_method": "txt2img",
                            "style_height": 768,
                            "style_width": 512,
                            "style_reference_image": "./00001.jpg",
                        },
                        # {
                        #     "style_prompt": "upper body, ((masterpiece)), 1990s style , Student, ID photo, Vintage, Retro, School, Nostalgia",
                        #     "style_negative_prompt": "BadDream, UnrealisticDream",
                        #     "style_model": "checkpoint/dreamshaper_8",
                        #     "style_lora_model": "lora/junmoxiao.safetensors",
                        #     "style_lora_scale": 1.0,
                        #     "style_method": "img2img",
                        #     "style_embeddings": [
                        #         "embedding/BadDream.pt",
                        #         "embedding/UnrealisticDream.pt"
                        #     ],
                        #     "style_reference_image": "examples/style-2.png",
                        # }
                    ]

                    first_stage_request_body = gr.JSON(label="First Stage Request Body, POST /api/v2/txt2img")
                    second_stage_request_body = gr.JSON(label="Second Stage Request Body, POST /api/v2/adetailer")

                    def mirror(*args):
                        return args

                    examples = gr.Examples(
                        [
                            [
                                _.get("style_prompt", ""),
                                _.get("style_negative_prompt", ""),
                                _.get("style_model", ""),
                                _.get("style_height", 512),
                                _.get("style_width", 512),
                                _.get("style_method", "txt2img"),
                                _.get("style_reference_image", ""),
                            ] for _ in templates
                        ],
                        [
                            style_prompt,
                            style_negative_prompt,
                            style_model,
                            style_height,
                            style_width,
                            style_method,
                            style_reference_image,
                        ],
                        [
                            style_prompt,
                            style_negative_prompt,
                            style_model,
                            style_height,
                            style_width,
                            style_method,
                            style_reference_image,
                        ],
                        mirror,
                        cache_examples=False,
                    )

            def generate(novita_key, gender, style_prompt, style_negative_prompt, style_model, style_lora, _hide_lora_training_response, style_hegiht, style_width, style_method, style_reference_image, num_images):

                def style(gender, style_prompt, style_negative_prompt, style_model, style_lora, _hide_lora_training_response, style_hegiht, style_width, style_method, style_reference_image,):
                    style_reference_image = Image.fromarray(style_reference_image)
                    if isinstance(style_lora, int):
                        style_lora = _hide_lora_training_response[style_lora].replace(".safetensors", "")
                    else:
                        style_lora = style_lora.replace(".safetensors", "")

                    height = int(style_hegiht)
                    width = int(style_width)

                    style_prompt = f"{first_stage_activication_words} {gender}, <lora:{style_lora}:{first_stage_lora_scale}>, {style_prompt}"

                    if style_method == "txt2img":
                        req = Txt2ImgRequest(
                            prompt=style_prompt,
                            negative_prompt=style_negative_prompt,
                            width=width,
                            height=height,
                            model_name=style_model,
                            steps=30,
                        )
                    elif style_method == "controlnet-depth":
                        req = Txt2ImgRequest(
                            prompt=style_prompt,
                            negative_prompt=style_negative_prompt,
                            width=width,
                            height=height,
                            model_name=style_model,
                            steps=30,
                            controlnet_units=[
                                ControlnetUnit(
                                    input_image=image_to_base64(style_reference_image),
                                    control_mode=ControlNetMode.BALANCED,
                                    model="control_v11f1p_sd15_depth",
                                    module=ControlNetPreprocessor.DEPTH,
                                    resize_mode=ControlNetResizeMode.RESIZE_OR_CORP,
                                    weight=1.0,
                                )
                            ]
                        )
                    elif style_method == "controlnet-pose":
                        req = Txt2ImgRequest(
                            prompt=style_prompt,
                            negative_prompt=style_negative_prompt,
                            width=width,
                            height=height,
                            model_name=style_model,
                            steps=30,
                            controlnet_units=[
                                ControlnetUnit(
                                    input_image=image_to_base64(style_reference_image),
                                    control_mode=ControlNetMode.BALANCED,
                                    model="control_v11p_sd15_openpose",
                                    module=ControlNetPreprocessor.OPENPOSE,
                                    resize_mode=ControlNetResizeMode.RESIZE_OR_CORP,
                                    weight=1.0,
                                )
                            ]
                        )
                    elif style_method == "controlnet-canny":
                        req = Txt2ImgRequest(
                            prompt=style_prompt,
                            negative_prompt=style_negative_prompt,
                            width=width,
                            height=height,
                            model_name=style_model,
                            steps=30,
                            controlnet_units=[
                                ControlnetUnit(
                                    input_image=image_to_base64(style_reference_image),
                                    control_mode=ControlNetMode.BALANCED,
                                    model="control_v11p_sd15_canny",
                                    module=ControlNetPreprocessor.CANNY,
                                    resize_mode=ControlNetResizeMode.RESIZE_OR_CORP,
                                    weight=1.0,
                                )
                            ]
                        )

                    res = get_noviata_client(novita_key).sync_txt2img(req)
                    style_image = Image.open(BytesIO(res.data.imgs_bytes[0]))

                    detailer_face_prompt = f"{second_stage_activication_words} {gender}, masterpiece, <lora:{style_lora}:{second_stage_lora_scale}>"
                    detailer_face_negative_prompt = style_negative_prompt

                    first_stage_request_body = req.to_dict()
                    second_stage_request_body = {
                        "prompt": detailer_face_prompt,
                        "negative_prompt": detailer_face_negative_prompt,
                        "model_name": style_model,
                        "image": "<INPUT_IMAGE>",
                        "strength": 0.3,
                        "steps": 50,
                    }

                    return Image.open(BytesIO(get_noviata_client(novita_key).adetailer(
                        prompt=detailer_face_prompt,
                        negative_prompt=detailer_face_negative_prompt,
                        model_name=style_model,
                        image=style_image,
                        strength=0.3,
                        steps=50,
                    ).data.imgs_bytes[0])), first_stage_request_body, second_stage_request_body
                images = []
                for _ in range(num_images):
                    try:
                        image, first_stage_request_body, second_stage_request_body = style(gender, style_prompt, style_negative_prompt, style_model, style_lora, _hide_lora_training_response,
                                                                                           style_hegiht, style_width, style_method, style_reference_image)
                        images.append(image)
                    except:
                        raise gr.Error(traceback.format_exc())

                return gr.update(value=images), first_stage_request_body, second_stage_request_body

            generate_button.click(
                inputs=[novita_key, inference_geneder, style_prompt, style_negative_prompt, style_model, style_lora, _hide_lora_training_response,
                        style_height, style_width, style_method, style_reference_image, num_images],
                outputs=[gallery, first_stage_request_body, second_stage_request_body],
                fn=generate
            )

        def onload(novita_key):
            if novita_key is None or novita_key == "":
                return novita_key, gr.update(choices=[], value=None), gr.update(value=None), f"$ UNKNOWN"
            try:
                user_info_json = get_noviata_client(novita_key).user_info()
                serving_models = [_.models[0].model_name for _ in get_noviata_client(novita_key).list_training().filter_by_model_status("SERVING")]
                serving_models_labels = [_.task_name for _ in get_noviata_client(novita_key).list_training().filter_by_model_status("SERVING")]
            except Exception as e:
                logging.error(e)
                return novita_key, gr.update(choices=[], value=None), gr.update(value=None), f"$ UNKNOWN"
            return novita_key, gr.update(choices=serving_models_labels), gr.update(value=serving_models), f"$ {user_info_json.credit_balance / 100 / 100:.2f}"

        novita_key.change(onload, inputs=novita_key, outputs=[novita_key, style_lora, _hide_lora_training_response, user_balance], _js="(v)=>{ setStorage('novita_key',v); return [v]; }")

        demo.load(
            inputs=[novita_key],
            outputs=[novita_key, style_lora, _hide_lora_training_response, user_balance],
            fn=onload,
            _js=get_local_storage,
        )

    return demo

    # style_method.change(
    #     inputs=[style_method],
    #     outputs=[style_reference_image],
    #     fn=lambda method: gr.update(visible=method in ["controlnet", "img2img", "ip-adapater"])
    # )


if __name__ == '__main__':
    demo = create_ui()
    demo.queue(api_open=False, concurrency_count=20)
    demo.launch()
