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
from diffusers import DiffusionPipeline
from fastapi.responses import StreamingResponse
import io
from io import BytesIO
login("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")
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
# os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["TORCH_USE_CUDA_DSA"] = '1'

Openai_API_KEY = ""
openai.api_key =os.getenv(Openai_API_KEY)
from openai import OpenAI
client = OpenAI(api_key=Openai_API_KEY)

# FastAPI
app = FastAPI(
    title="FLUX schnell"
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
    gpt_prompt=f"""You are an AI language model tasked with translating given Korean text into English text. 

            Korean: {text}
            English:
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

pipe = DiffusionPipeline.from_pretrained("black-forest-labs/FLUX.1-schnell", torch_dtype=dtype).to(device)


MAX_SEED = np.iinfo(np.int32).max
MAX_IMAGE_SIZE = 2048

async def infer_flux(pipe, prompt, 
                seed=42, randomize_seed=False, 
                width=1024, height=1024, 
                num_inference_steps=4):
    if randomize_seed:
        seed = random.randint(0, MAX_SEED)
    generator = torch.Generator().manual_seed(seed)
    image = pipe(
            prompt = prompt, 
            width = width,
            height = height,
            num_inference_steps = num_inference_steps, 
            generator = generator,
            guidance_scale=0.0
    ).images[0] 
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

    input_prompt=""
    input_prompt+= await translate(user_input)

    file_name="".join(random.choice(string.ascii_lowercase + string.digits) for _ in range(8))
    save_as=f"./{file_name}.png"
    img_res= await infer_flux(pipe,input_prompt)

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