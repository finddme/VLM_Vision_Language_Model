"""
https://github.com/Kwai-Kolors/Kolors/tree/master/kolors
https://huggingface.co/spaces/gokaygokay/Kolors/blob/main/app.py
"""

import gradio as gr
import numpy as np
import random
import torch
from diffusers import DiffusionPipeline
from diffusers import StableDiffusionPipeline
from diffusers import StableDiffusion3Pipeline, SD3Transformer2DModel, FlowMatchEulerDiscreteScheduler
from diffusers import AuraFlowPipeline
from huggingface_hub import login
import os
import torch
import random
from huggingface_hub import snapshot_download
from kolors.pipelines.pipeline_stable_diffusion_xl_chatglm_256 import StableDiffusionXLPipeline
from kolors.models.modeling_chatglm import ChatGLMModel
from kolors.models.tokenization_chatglm import ChatGLMTokenizer
from diffusers import UNet2DConditionModel, AutoencoderKL
from diffusers import EulerDiscreteScheduler
import gradio as gr
login("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")


import openai, os
Openai_API_KEY = "sk-proj-CFsSaackBkfJnN8eAtCDT3BlbkFJaFtkZfjEcubhPJ7pX3sA"
openai.api_key =os.getenv(Openai_API_KEY)
from openai import OpenAI
client = OpenAI(api_key=Openai_API_KEY)

def translate(text):
    gpt_prompt=f"""Translate the following Korean text to English:

            {text}
            """
    # response = openai.ChatCompletion.create(
    response = client.chat.completions.create(
        model="gpt-4o",
        # api_key=Openai_API_KEY,
        messages=[{"role": "system", "content": gpt_prompt}],
        max_tokens=1024)
    # return response.choices[0]["message"]["content"]
    return response.choices[0].message.content

# Download the model files
ckpt_dir = snapshot_download(repo_id="Kwai-Kolors/Kolors")

# Load the models
text_encoder = ChatGLMModel.from_pretrained(
    os.path.join(ckpt_dir, 'text_encoder'),
    torch_dtype=torch.float16).half()
tokenizer = ChatGLMTokenizer.from_pretrained(os.path.join(ckpt_dir, 'text_encoder'))
vae = AutoencoderKL.from_pretrained(os.path.join(ckpt_dir, "vae"), revision=None).half()
scheduler = EulerDiscreteScheduler.from_pretrained(os.path.join(ckpt_dir, "scheduler"))
unet = UNet2DConditionModel.from_pretrained(os.path.join(ckpt_dir, "unet"), revision=None).half()

pipe = StableDiffusionXLPipeline(
        vae=vae,
        text_encoder=text_encoder,
        tokenizer=tokenizer,
        unet=unet,
        scheduler=scheduler,
        force_zeros_for_empty_prompt=False)
pipe = pipe.to("cuda")

# 84463312
# 1279465766
def generate_image(prompt, 
                    negative_prompt=None, 
                    width=1024, height=1024, 
                    num_inference_steps=60, 
                    guidance_scale=0.5, num_images_per_prompt=1, 
                    use_random_seed=True, seed=0, progress=gr.Progress(track_tqdm=True)):
    if use_random_seed:
        seed = random.randint(0, 2**32 - 1)
    else:
        seed = int(seed)  # Ensure seed is an integer
    # seed = int(1279465766)
    prompt=translate(prompt)
    input_prompt=f"""{prompt}. (it must be drawn to look exactly like a real photo.)
    """
    image = pipe(
        prompt=input_prompt,
        negative_prompt=negative_prompt,
        height=height,
        width=width,
        num_inference_steps=num_inference_steps,
        guidance_scale=guidance_scale,
        num_images_per_prompt=num_images_per_prompt,
        generator=torch.Generator(pipe.device).manual_seed(seed)
    ).images[0] 
    return image, seed,prompt

MAX_SEED = np.iinfo(np.int32).max
MAX_IMAGE_SIZE = 1344

# 1980389032
# 43455650
def infer(prompt, negative_prompt=None, 
            seed=43455650, randomize_seed=True, width=1024, height=1024, guidance_scale=5.0, 
                    num_inference_steps=28, progress=gr.Progress(track_tqdm=True)):
    if randomize_seed:
        seed = random.randint(0, MAX_SEED)
    generator = torch.Generator().manual_seed(666)
    prompt=translate(prompt)
    prompt=f""" {prompt}

    """
    image = pipe(
        prompt = prompt, 
        negative_prompt = negative_prompt,
        guidance_scale = 3.5, 
        num_inference_steps = 50, 
        width = width, 
        height = height,
        generator = generator
    ).images[0] 
    
    return image, seed
    # return image

examples = [
    "날아라 고양이, 구름이랑 해가 있는 파란 하늘","파란색 얼룩말","겨울 해변 그림을 그려줘"
]

css="""
#col-container {
    margin: 0 auto;
    max-width: 756px;
}
"""

with gr.Blocks(css=css) as demo:
    
    with gr.Column(elem_id="col-container"):
        gr.Markdown(f"""
        """)
        
        with gr.Row():
            
            prompt = gr.Text(
                label="Prompt",
                show_label=False,
                max_lines=1,
                placeholder="Prompt",
                container=False,
            )
            
            run_button = gr.Button("Run", scale=0)
        
        result = gr.Image(label="Result", show_label=False)
        translate_res = gr.Text(label="Translate Result")

        with gr.Accordion("Advanced Settings", open=False):
            
        #     negative_prompt = gr.Text(
        #         label="Negative prompt",
        #         max_lines=1,
        #         placeholder="Enter a negative prompt",
        #     )
            
            seed = gr.Slider(
                label="Seed",
                minimum=0,
                maximum=MAX_SEED,
                step=1,
                value=0,
            )
            
            # randomize_seed = gr.Checkbox(label="Randomize seed", value=True)
            
            # with gr.Row():
                
            #     width = gr.Slider(
            #         label="Width",
            #         minimum=256,
            #         maximum=MAX_IMAGE_SIZE,
            #         step=64,
            #         value=1024,
            #     )
                
            #     height = gr.Slider(
            #         label="Height",
            #         minimum=256,
            #         maximum=MAX_IMAGE_SIZE,
            #         step=64,
            #         value=1024,
            #     )
            
            # with gr.Row():
                
            #     guidance_scale = gr.Slider(
            #         label="Guidance scale",
            #         minimum=0.0,
            #         maximum=10.0,
            #         step=0.1,
            #         value=5.0,
            #     )
                
            #     num_inference_steps = gr.Slider(
            #         label="Number of inference steps",
            #         minimum=1,
            #         maximum=50,
            #         step=1,
            #         value=28,
            #     )
        
        gr.Examples(
            examples = examples,
            inputs = [prompt]
        )
    gr.on(
        triggers=[run_button.click, prompt.submit, 
        # negative_prompt.submit
        ],
        fn = generate_image,
        inputs = [prompt, 
                # negative_prompt, seed, randomize_seed, width, height, guidance_scale, num_inference_steps
                ],
        outputs = [result, 
                    seed,
                    translate_res
                    ]
    )

demo.launch(server_name = "0.0.0.0",share=True)