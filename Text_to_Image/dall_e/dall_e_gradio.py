import gradio as gr

def infer(prompt):

    response = client.images.generate(
      model="dall-e-3",
      prompt=prompt,
      size="1024x1024",
      quality="standard",
      n=1,
    )

    image_url = response.data[0].url
    
    return image_url
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
        
        # result = gr.Text(label="Result")
        result = gr.Image(label="Image Output")
        gr.Examples(
            examples = examples,
            inputs = [prompt]
        )
    gr.on(
        triggers=[run_button.click, prompt.submit],
        fn = infer,
        inputs = [prompt],
        outputs = [result]
    )
demo.launch(server_name = "0.0.0.0",share=True, server_port=7643)