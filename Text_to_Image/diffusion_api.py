from fastapi import FastAPI, File, UploadFile
from fastapi import FastAPI, File, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
import uvicorn
import gradio as gr
import numpy as np
import random
import torch
import openai, os, gc
from diffusers import StableDiffusion3Pipeline, SD3Transformer2DModel, FlowMatchEulerDiscreteScheduler
from fastapi.middleware.cors import CORSMiddleware
from huggingface_hub import login
import string
from fastapi.responses import StreamingResponse
import io
from io import BytesIO
login("")


# FastAPI
app = FastAPI(
    title="stable Diffusion"
)

origins = ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

async def translate(text):
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

device = "cuda:1" if torch.cuda.is_available() else "cpu"
dtype = torch.float16

repo = "stabilityai/stable-diffusion-3-medium-diffusers"
pipe = StableDiffusion3Pipeline.from_pretrained(repo, torch_dtype=torch.float16).to(device)


async def inference_diffusion(pipe,prompt, 
                                seed=43455650, 
                                randomize_seed=False, 
                                width=1024, height=1024, 
                                guidance_scale=5.0, 
                                num_inference_steps=28):

    generator = torch.Generator().manual_seed(seed)
    prompt= await translate(prompt)
    prompt=f""" {prompt}

    """
    image = pipe(
        prompt = prompt, 
        # negative_prompt = negative_prompt,
        guidance_scale = guidance_scale, 
        num_inference_steps = num_inference_steps, 
        width = width, 
        height = height,
        generator = generator
    ).images[0] 
    
    # return image, seed
    return image

@app.get("/")
def home():
    return {"message": "stable Diffusion"}

from pydantic import BaseModel as BM
class UserInput(BM):
    user_input: str

@app.post("/genrate_image")
async def process_diffusion(user_input: UserInput):
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
    gc.collect()

    user_input=user_input.user_input

    torch.cuda.empty_cache()
    gc.collect()

    file_name="".join(random.choice(string.ascii_lowercase + string.digits) for _ in range(8))
    save_as=f"./{file_name}.png"
    img_res= await inference_diffusion(pipe,user_input)

    # bytes_io = io.BytesIO()
    # # img_res.save(save_as)
    # img_res.save(bytes_io, format="png")
    memory_stream = io.BytesIO()
    img_res.save(memory_stream, format="PNG")
    memory_stream.seek(0)
    # return Response(bytes_io.getvalue(), media_type="image/png")
    return StreamingResponse(memory_stream, media_type="image/png")

if __name__ == '__main__':
    uvicorn.run(app, host="0.0.0.0", port=8782)