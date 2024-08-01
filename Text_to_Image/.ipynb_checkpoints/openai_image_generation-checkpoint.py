with gr.Blocks(css=css) as demo:
from openai import OpenAI
client = OpenAI(api_key="sk-proj-CFsSaackBkfJnN8eAtCDT3BlbkFJaFtkZfjEcubhPJ7pX3sA")
# import openai, os
# Openai_API_KEY = "sk-proj-CFsSaackBkfJnN8eAtCDT3BlbkFJaFtkZfjEcubhPJ7pX3sA"
# openai.api_key =os.getenv(Openai_API_KEY)

response = client.images.generate(
  model="dall-e-3",
  prompt="a white siamese cat",
  size="1024x1024",
  quality="standard",
  n=1,
)

image_url = response.data[0].url