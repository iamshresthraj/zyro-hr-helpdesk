import os
import gc
import torch
import gradio as gr
from PIL import Image
from transformers import AutoModelForCausalLM, AutoTokenizer
from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler

# ── Configuration ──────────────────────────────────────────────
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.float16 if DEVICE == "cuda" else torch.float32

LLM_MODEL_ID = "Qwen/Qwen2.5-0.5B-Instruct"
SD_MODEL_ID = "runwayml/stable-diffusion-v1-5"

# ── System Prompts ──────────────────────────────────────────────
SYSTEM_PROMPT = """You are a professional product photographer and prompt engineer specializing in e-commerce imagery.
Your task: convert a simple product description into a concise, high-quality Stable Diffusion prompt.

Rules:
- Output ONLY the prompt, no explanations or extra text.
- Keep the prompt under 60 words.
- Always include: product type, material/texture, studio lighting, clean background, camera angle, and quality keywords.
- Always include these CLIP-aligned keywords: "product photography", "studio lighting", "white background", "sharp focus", "ultra realistic", "8k".
- Use a consistent structure: [product] [details] [lighting] [background] [camera] [quality].
- Do NOT include negative prompts, special tokens, or markdown formatting."""

NEGATIVE_PROMPT = (
    "blurry, low quality, distorted, deformed, ugly, noisy, text, watermark, "
    "oversaturated, cartoon, illustration, painting, sketch, bad proportions, "
    "cropped, out of frame, duplicate, morbid, mutilated"
)

# ── Load Models ──────────────────────────────────────────────
print(f"📦 Loading Models on {DEVICE}...")

# Load LLM
llm_tokenizer = AutoTokenizer.from_pretrained(LLM_MODEL_ID, trust_remote_code=True)
llm_model = AutoModelForCausalLM.from_pretrained(
    LLM_MODEL_ID,
    torch_dtype=DTYPE,
    device_map="auto" if DEVICE == "cuda" else None,
    trust_remote_code=True,
)
llm_model.eval()

# Load Stable Diffusion
sd_pipe = StableDiffusionPipeline.from_pretrained(
    SD_MODEL_ID,
    torch_dtype=DTYPE,
    safety_checker=None,
    requires_safety_checker=False,
)
sd_pipe.scheduler = DPMSolverMultistepScheduler.from_config(sd_pipe.scheduler.config)
sd_pipe = sd_pipe.to(DEVICE)

# Memory optimizations
if DEVICE == "cuda":
    sd_pipe.enable_attention_slicing()
    try:
        sd_pipe.enable_xformers_memory_efficient_attention()
    except Exception:
        pass

# ── Core Logic ──────────────────────────────────────────────
def generate_engineered_prompt(description: str) -> str:
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"Product: {description}"},
    ]

    text = llm_tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )

    inputs = llm_tokenizer(text, return_tensors="pt").to(llm_model.device)

    with torch.no_grad():
        output_ids = llm_model.generate(
            **inputs,
            max_new_tokens=100,
            temperature=0.7,
            top_p=0.9,
            do_sample=True,
            repetition_penalty=1.15,
            pad_token_id=llm_tokenizer.eos_token_id,
        )

    prompt = llm_tokenizer.decode(
        output_ids[0][inputs["input_ids"].shape[1]:],
        skip_special_tokens=True,
    ).strip()

    # Post-processing
    prompt = prompt.strip('"').strip("'").strip()
    for prefix in ["Prompt:", "prompt:", "Output:", "output:"]:
        if prompt.startswith(prefix):
            prompt = prompt[len(prefix):].strip()

    # Ensure keywords
    clip_keywords = ["product photography", "studio lighting", "sharp focus"]
    for kw in clip_keywords:
        if kw.lower() not in prompt.lower():
            prompt += f", {kw}"
    
    return prompt

def process_generation(description: str, progress=gr.Progress()):
    if not description or not description.strip():
        raise gr.Error("Please enter a product description.")

    progress(0.1, desc="🤖 Engineering prompt...")
    prompt = generate_engineered_prompt(description.strip())
    
    progress(0.4, desc="🎨 Generating image...")
    generator = torch.Generator(device=DEVICE).manual_seed(42)
    
    with torch.no_grad(), torch.cuda.amp.autocast(enabled=(DEVICE == "cuda")):
        result = sd_pipe(
            prompt=prompt,
            negative_prompt=NEGATIVE_PROMPT,
            num_inference_steps=30,
            guidance_scale=7.5,
            width=512,
            height=512,
            generator=generator,
        )
    
    image = result.images[0]
    progress(1.0, desc="✅ Done!")
    
    return prompt, image

# ── Premium UI Design ──────────────────────────────────────────────
CSS = """
.container { 
    max-width: 1000px; 
    margin: auto; 
    padding: 20px; 
}
.header {
    text-align: center;
    margin-bottom: 30px;
    background: linear-gradient(90deg, #4F46E5, #9333EA);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}
.footer {
    text-align: center;
    margin-top: 50px;
    font-size: 0.9em;
    color: #6B7280;
}
.generate-btn {
    background: linear-gradient(90deg, #4F46E5, #7C3AED) !important;
    border: none !important;
    transition: transform 0.2s !important;
}
.generate-btn:hover {
    transform: translateY(-2px) !important;
    filter: brightness(1.1) !important;
}
.output-card {
    border-radius: 12px;
    box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -1px rgba(0, 0, 0, 0.06);
    background: #ffffff;
    padding: 10px;
}
"""

with gr.Blocks(theme=gr.themes.Default(primary_hue="indigo", font=["Inter", "sans-serif"]), css=CSS) as demo:
    with gr.Div(elem_classes="container"):
        gr.HTML("<h1 class='header'>📸 Premium AI Product Generator</h1>")
        gr.Markdown(
            "Transform your product ideas into professional studio-quality photographs instantly. "
            "Our pipeline leverages **Qwen2.5** for expert prompt engineering and **Stable Diffusion** for photorealistic rendering."
        )

        with gr.Tabs():
            with gr.TabItem("🎯 Single Generation"):
                with gr.Row(variant="compact"):
                    with gr.Column(scale=5):
                        input_text = gr.Textbox(
                            label="Product Description",
                            placeholder="Describe your product (e.g., 'luxury leather watch with minimalist face')",
                            lines=2,
                            container=False
                        )
                    with gr.Column(scale=1):
                        generate_btn = gr.Button("🚀 Generate", variant="primary", elem_classes="generate-btn")

                with gr.Row():
                    with gr.Column(scale=1):
                        gr.Markdown("### 🤖 Meta Output")
                        output_prompt = gr.Textbox(
                            label="Engineered Prompt", 
                            lines=6, 
                            interactive=False,
                            show_copy_button=True
                        )
                    
                    with gr.Column(scale=1):
                        gr.Markdown("### 🖼️ Result")
                        output_image = gr.Image(
                            label="Generated Image", 
                            type="pil", 
                            height=450,
                            show_label=False,
                            elem_classes="output-card"
                        )
                
                gr.Examples(
                    examples=[
                        ["minimalist white sneakers on a marble pedestal"],
                        ["vintage copper coffee maker in soft morning light"],
                        ["organic skincare bottle with fresh leaves"],
                        ["high-end wireless headphones, matte gold finish"],
                        ["ceramic teapot with intricate indigo pattern"]
                    ],
                    inputs=input_text,
                    label="Inspiration"
                )

            with gr.TabItem("📦 Batch Showcase"):
                gr.Markdown("### 📊 Pre-computed Portfolio")
                gr.Markdown("Browse the results of our specialized 15-product batch pipeline.")
                
                with gr.Row():
                    refresh_btn = gr.Button("🔄 Refresh Gallery", variant="secondary")
                
                gallery = gr.Gallery(
                    label="Product Catalog",
                    show_label=False,
                    columns=[3, 5],
                    rows=[3],
                    object_fit="contain",
                    height="auto"
                )
                
                def load_gallery():
                    image_folder = "images"
                    if not os.path.exists(image_folder):
                        return []
                    
                    images = []
                    # Load img_1.png through img_15.png
                    for i in range(1, 16):
                        img_path = os.path.join(image_folder, f"img_{i}.png")
                        if os.path.exists(img_path):
                            images.append((img_path, f"Product {i}"))
                    return images

                refresh_btn.click(fn=load_gallery, outputs=gallery)
                demo.load(fn=load_gallery, outputs=gallery)
                
                gr.Info("Click 'Refresh' or run the Batch cell in your notebook to update this gallery.")

        gr.HTML("<div class='footer'>Built with Qwen2.5 & Stable Diffusion v1.5 • NIAT x VGU Masterclass</div>")

    generate_btn.click(
        fn=process_generation,
        inputs=input_text,
        outputs=[output_prompt, output_image],
    )

if __name__ == "__main__":
    demo.launch()
