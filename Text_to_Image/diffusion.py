import gradio as gr
import numpy as np
import random
import torch
from diffusers import StableDiffusion3Pipeline, SD3Transformer2DModel, FlowMatchEulerDiscreteScheduler

from huggingface_hub import login
login("")


def translate(text):
    gpt_prompt=f"""Translate the following English text to Korean:

            \"\"\"{text}\"\"\"
            """
    # response = openai.ChatCompletion.create(
    response = client.chat.completions.create(
        model="gpt-4o",
        # api_key=Openai_API_KEY,
        messages=[{"role": "system", "content": gpt_prompt}],
        max_tokens=1024)
    # return response.choices[0]["message"]["content"]
    return response.choices[0].message.content


device = "cuda" if torch.cuda.is_available() else "cpu"
dtype = torch.float16

repo = "stabilityai/stable-diffusion-3-medium-diffusers"
# repo =  "Bingsu/my-korean-stable-diffusion-v1-5"
pipe = StableDiffusion3Pipeline.from_pretrained(repo, torch_dtype=torch.float16).to(device)

MAX_SEED = np.iinfo(np.int32).max
MAX_IMAGE_SIZE = 1344

# 1980389032
# 43455650
def infer(prompt, negative_prompt=None, 
            seed=43455650, randomize_seed=False, width=1024, height=1024, guidance_scale=5.0, 
                    num_inference_steps=28, progress=gr.Progress(track_tqdm=True)):
    if randomize_seed:
        seed = random.randint(0, MAX_SEED)
    generator = torch.Generator().manual_seed(seed)
    prompt=translate(prompt)
    prompt=f""" {prompt}

    """
    image = pipe(
        prompt = prompt, 
        negative_prompt = negative_prompt,
        guidance_scale = guidance_scale, 
        num_inference_steps = 50, 
        width = width, 
        height = height,
        generator = generator
    ).images[0] 
    
    # return image, seed
    return image

examples = [
    "날아라 고양이, 구름이랑 해가 있는 파란 하늘","겨울 해변 그림을 그려줘"
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

        # with gr.Accordion("Advanced Settings", open=False):
            
        #     negative_prompt = gr.Text(
        #         label="Negative prompt",
        #         max_lines=1,
        #         placeholder="Enter a negative prompt",
        #     )
            
            # seed = gr.Slider(
            #     label="Seed",
            #     minimum=0,
            #     maximum=MAX_SEED,
            #     step=1,
            #     value=0,
            # )
            
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
        fn = infer,
        inputs = [prompt, 
                # negative_prompt, seed, randomize_seed, width, height, guidance_scale, num_inference_steps
                ],
        outputs = [result, 
                    # seed
                    ]
    )

demo.launch(server_name = "0.0.0.0",share=True)