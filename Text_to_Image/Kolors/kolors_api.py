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
from huggingface_hub import snapshot_download
from kolors.pipelines.pipeline_stable_diffusion_xl_chatglm_256 import StableDiffusionXLPipeline
from kolors.models.modeling_chatglm import ChatGLMModel
from kolors.models.tokenization_chatglm import ChatGLMTokenizer
from huggingface_hub import login
import string
from fastapi.responses import StreamingResponse
from diffusers import UNet2DConditionModel, AutoencoderKL
from diffusers import EulerDiscreteScheduler
import io
from io import BytesIO
import time
from transformers import AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer
from threading import Thread
from huggingface_hub import login
import huggingface_hub

# FastAPI
app = FastAPI(
    title="Kolors"
)

origins = ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

"""
################ Translate model prepare ################
"""

login("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")

Openai_API_KEY = "sk-proj-CFsSaackBkfJnN8eAtCDT3BlbkFJaFtkZfjEcubhPJ7pX3sA"
openai.api_key =os.getenv(Openai_API_KEY)
from openai import OpenAI
client = OpenAI(api_key=Openai_API_KEY)

translate_model_flag="mistral" #gpt

async def gpt_translate(text):
    gpt_prompt=f"""Translate the following Korean text to Chinese:

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

def load_mistral():

    device = "cuda:1" # for GPU usage or "cpu" for CPU usage

    tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-Nemo-Instruct-2407")
    model = AutoModelForCausalLM.from_pretrained(
        "mistralai/Mistral-Nemo-Instruct-2407",
        token="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",
        torch_dtype=torch.bfloat16,
        device_map=device,
        ignore_mismatched_sizes=True,
        # low_cpu_mem_usage=True
        )
    return tokenizer,model

if translate_model_flag=="mistral":
    translate_tokenizer,translate_model=load_mistral()

async def nemo_translate(text):
    global translate_tokenizer
    global translate_model

    temperature = 0.3
    max_new_tokens= 1024
    top_p = 1.0
    top_k = 20
    buffer = ""
    nemo_prompt=f"""You are an AI language model tasked with translating given Korean text into Chinese text. 
                The response must include only the translation result.
                Examples:

                    Korean: "안녕하세요, 오늘 날씨가 참 좋네요."
                    Chinese: "你好，今天天气真好。"

                    Korean: "학교에서 배우는 과목 중에서 수학이 제일 어려워요."
                    Chinese: "在学校学的科目中，数学最难。"

                    Korean: "이 문제를 해결하기 위해서는 많은 노력이 필요합니다."
                    Chinese: "解决这个问题需要很多努力。"

                Korean:{text}
                Chinese:
                """
    inputs = translate_tokenizer.encode(nemo_prompt, return_tensors="pt").to("cuda:1")
    streamer = TextIteratorStreamer(translate_tokenizer, timeout=60.0, skip_prompt=True, skip_special_tokens=True)
    generate_kwargs = dict(
        input_ids=inputs, 
        max_new_tokens = max_new_tokens,
        do_sample = False if temperature == 0 else True,
        top_p = top_p,
        top_k = top_k,
        temperature = temperature,
        streamer=streamer,
        pad_token_id = 10,
    )

    with torch.no_grad():
        thread = Thread(target=translate_model.generate, kwargs=generate_kwargs)
        thread.start()

        # while 1:
        #     time.sleep(0.)

    for new_text in streamer:
        buffer += new_text
    return buffer

async def llama_model(prompt):
    url = "http://115.71.28.95:8584/model_generate"
    payload = json.dumps({"prompts": prompt})
    headers = { 'Content-Type': 'application/json' }
    response = requests.request("POST", url , headers=headers, data=payload)
    return response.json()["answer"][0]

async def llama_translate(text):
    system_prompt=f"""You are an AI language model tasked with translating given Korean text into Chinese text. 
                    The response must include only the translation result.
                    Examples:
        
                        Korean: "안녕하세요, 오늘 날씨가 참 좋네요."
                        Chinese: "你好，今天天气真好。"
        
                        Korean: "학교에서 배우는 과목 중에서 수학이 제일 어려워요."
                        Chinese: "在学校学的科目中，数学最难。"
        
                        Korean: "이 문제를 해결하기 위해서는 많은 노력이 필요합니다."
                        Chinese: "解决这个问题需要很多努力。"
                    """
    user_prompt=f"""
                    Korean: {text}
                    Chinese:
                """
    prompt=f"""<|begin_of_text|><|start_header_id|>system<|end_header_id|>
                {system_prompt}<|eot_id|><|start_header_id|>user<|end_header_id|>
                {user_prompt}<|eot_id|>"""
    response=await llama_model(prompt)
    return response.replace('<|start_header_id|>assistant<|end_header_id|>','').replace('\n\n','')

"""
################ Image generation model prepare ################
"""

ckpt_dir = snapshot_download(repo_id="Kwai-Kolors/Kolors")

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

async def inference_kolors(pipe,prompt, translate_model_flag,
                    negative_prompt=None, 
                    width=1024, height=1024, 
                    num_inference_steps=50, 
                    guidance_scale=3.0, num_images_per_prompt=1, 
                    use_random_seed=True, seed=0, progress=gr.Progress(track_tqdm=True)):

    if use_random_seed:
        seed = random.randint(0, 2**32 - 1)
    else:
        seed = int(seed)  # Ensure seed is an integer
    # seed = int(1043513496) # 조금 유화 같음
    # seed = int(2835313038) # 붓자국이 덜보임
    seed = int(1603815676)
    print(f"seed: {seed}")

    input_prompt=""
    if translate_model_flag=="mistral":
        print("TRANSLATE WITH MISTRAL")
        input_prompt+= await nemo_translate(prompt)
    elif translate_model_flag=="llama":
        print("TRANSLATE WITH LLAMA")
        input_prompt+= await llama_translate(prompt)
    else:
        print("TRANSLATE WITH GPT-4")
        input_prompt+= await gpt_translate(prompt)

    # Generate the image like a real photograph. It must not have a painting-like feel.
    #  生成的图像必须像真实照片，不能有绘画感。
    input_prompt=f"""{input_prompt}.
    """
    print(f"-------------------->{input_prompt}")
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
    return image

"""
################ API ################
"""

@app.get("/")
def home():
    return {"message": "Kolors"}

from pydantic import BaseModel as BM
class UserInput(BM):
    user_input: str

@app.post("/genrate_image")
async def process_kolors(user_input: UserInput):
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
    gc.collect()

    user_input=user_input.user_input

    torch.cuda.empty_cache()
    gc.collect()

    global translate_model_flag

    file_name="".join(random.choice(string.ascii_lowercase + string.digits) for _ in range(8))
    save_as=f"./{file_name}.png"
    img_res= await inference_kolors(pipe,user_input,translate_model_flag)

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