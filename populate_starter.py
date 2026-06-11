import nbformat as nbf
import os

def populate_starter(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        nb = nbf.read(f, as_version=4)

    # Define the code for each target cell
    # Note: Indices based on the view_file output
    
    # Cell 7 (index 7): Load Dataset
    nb['cells'][7]['source'] = [
        "import pandas as pd\n",
        "\n",
        "df = pd.read_csv('starter_notebook.ipynb.csv')\n",
        "print(f'Loaded {len(df)} product descriptions')\n",
        "df.head()"
    ]

    # Cell 10 (index 10): Load LLM
    nb['cells'][10]['source'] = [
        "from transformers import AutoModelForCausalLM, AutoTokenizer\n",
        "import torch\n",
        "\n",
        "DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'\n",
        "DTYPE = torch.float16 if DEVICE == 'cuda' else torch.float32\n",
        "\n",
        "LLM_MODEL_ID = 'Qwen/Qwen2.5-0.5B-Instruct'\n",
        "\n",
        "print(f'📦 Loading LLM: {LLM_MODEL_ID} ...')\n",
        "llm_tokenizer = AutoTokenizer.from_pretrained(LLM_MODEL_ID, trust_remote_code=True)\n",
        "llm_model = AutoModelForCausalLM.from_pretrained(\n",
        "    LLM_MODEL_ID,\n",
        "    torch_dtype=DTYPE,\n",
        "    device_map='auto' if DEVICE == 'cuda' else None,\n",
        "    trust_remote_code=True,\n",
        ")\n",
        "llm_model.eval()\n",
        "print('✅ LLM loaded.')"
    ]

    # Cell 11 (index 11): Prompt Engineering Function
    nb['cells'][11]['source'] = [
        "SYSTEM_PROMPT = \"\"\"You are a professional product photographer and prompt engineer specializing in e-commerce imagery.\n",
        "Your task: convert a simple product description into a concise, high-quality Stable Diffusion prompt.\n",
        "\n",
        "Rules:\n",
        "- Output ONLY the prompt, no explanations or extra text.\n",
        "- Keep the prompt under 60 words.\n",
        "- Always include: product type, material/texture, studio lighting, clean background, camera angle, and quality keywords.\n",
        "- Always include these CLIP-aligned keywords: 'product photography', 'studio lighting', 'white background', 'sharp focus', 'ultra realistic', '8k'.\n",
        "- Use a consistent structure: [product] [details] [lighting] [background] [camera] [quality].\n",
        "- Do NOT include negative prompts, special tokens, or markdown formatting.\"\"\"\n",
        "\n",
        "def engineer_prompt(product_description):\n",
        "    messages = [\n",
        "        {'role': 'system', 'content': SYSTEM_PROMPT},\n",
        "        {'role': 'user', 'content': f'Product: {product_description}'},\n",
        "    ]\n",
        "\n",
        "    text = llm_tokenizer.apply_chat_template(\n",
        "        messages,\n",
        "        tokenize=False,\n",
        "        add_generation_prompt=True,\n",
        "    )\n",
        "\n",
        "    inputs = llm_tokenizer(text, return_tensors='pt').to(llm_model.device)\n",
        "\n",
        "    with torch.no_grad():\n",
        "        output_ids = llm_model.generate(\n",
        "            **inputs,\n",
        "            max_new_tokens=100,\n",
        "            temperature=0.7,\n",
        "            top_p=0.9,\n",
        "            do_sample=True,\n",
        "            repetition_penalty=1.15,\n",
        "            pad_token_id=llm_tokenizer.eos_token_id,\n",
        "        )\n",
        "\n",
        "    prompt = llm_tokenizer.decode(\n",
        "        output_ids[0][inputs['input_ids'].shape[1]:],\n",
        "        skip_special_tokens=True,\n",
        "    ).strip()\n",
        "\n",
        "    # Clean up\n",
        "    prompt = prompt.strip('\"').strip(\"'\").strip()\n",
        "    for prefix in ['Prompt:', 'prompt:', 'Output:', 'output:']:\n",
        "        if prompt.startswith(prefix):\n",
        "            prompt = prompt[len(prefix):].strip()\n",
        "\n",
        "    # Ensure keywords\n",
        "    clip_keywords = ['product photography', 'studio lighting', 'sharp focus']\n",
        "    for kw in clip_keywords:\n",
        "        if kw.lower() not in prompt.lower():\n",
        "            prompt += f', {kw}'\n",
        "\n",
        "    return prompt\n",
        "\n",
        "# Test it\n",
        "test_val = engineer_prompt('red leather handbag')\n",
        "print(f'Test prompt: {test_val}')"
    ]

    # Cell 14 (index 14): Load Stable Diffusion
    nb['cells'][14]['source'] = [
        "from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler\n",
        "\n",
        "SD_MODEL_ID = 'runwayml/stable-diffusion-v1-5'\n",
        "\n",
        "print(f'📦 Loading Stable Diffusion: {SD_MODEL_ID} ...')\n",
        "sd_pipe = StableDiffusionPipeline.from_pretrained(\n",
        "    SD_MODEL_ID,\n",
        "    torch_dtype=DTYPE,\n",
        "    safety_checker=None,\n",
        "    requires_safety_checker=False,\n",
        ")\n",
        "sd_pipe.scheduler = DPMSolverMultistepScheduler.from_config(sd_pipe.scheduler.config)\n",
        "sd_pipe = sd_pipe.to(DEVICE)\n",
        "\n",
        "# Memory optimizations\n",
        "if DEVICE == 'cuda':\n",
        "    sd_pipe.enable_attention_slicing()\n",
        "    try:\n",
        "        sd_pipe.enable_xformers_memory_efficient_attention()\n",
        "        print('✓ xformers enabled')\n",
        "    except Exception:\n",
        "        print('⚠ xformers not available')\n",
        "\n",
        "print('✅ Stable Diffusion pipeline ready.')"
    ]

    # Cell 15 (index 15): Generate Image Function
    nb['cells'][15]['source'] = [
        "NEGATIVE_PROMPT = (\n",
        "    'blurry, low quality, distorted, deformed, ugly, noisy, text, watermark, '\n",
        "    'oversaturated, cartoon, illustration, painting, sketch, bad proportions, '\n",
        "    'cropped, out of frame, duplicate, morbid, mutilated'\n",
        ")\n",
        "\n",
        "def generate_image(engineered_prompt):\n",
        "    generator = torch.Generator(device=DEVICE).manual_seed(42)\n",
        "    with torch.no_grad(), torch.cuda.amp.autocast(enabled=(DEVICE == 'cuda')):\n",
        "        result = sd_pipe(\n",
        "            prompt=engineered_prompt,\n",
        "            negative_prompt=NEGATIVE_PROMPT,\n",
        "            num_inference_steps=30,\n",
        "            guidance_scale=7.5,\n",
        "            width=512,\n",
        "            height=512,\n",
        "            generator=generator,\n",
        "        )\n",
        "    return result.images[0]\n",
        "\n",
        "# Test it\n",
        "img = generate_image('A professional photo of red headphones on white background')\n",
        "display(img)"
    ]

    # Cell 18 (index 18): Gradio Interface
    nb['cells'][18]['source'] = [
        "import gradio as gr\n",
        "\n",
        "def gradio_generate(description):\n",
        "    if not description or not description.strip():\n",
        "        return 'Please enter a description.', None\n",
        "    prompt = engineer_prompt(description.strip())\n",
        "    image = generate_image(prompt)\n",
        "    return prompt, image\n",
        "\n",
        "with gr.Blocks(title='AI Product Image Generator') as demo:\n",
        "    gr.Markdown('# 🎨 AI Product Image Generator')\n",
        "    with gr.Row():\n",
        "        with gr.Column():\n",
        "            input_text = gr.Textbox(label='Product Description', placeholder='e.g. red vintage camera')\n",
        "            btn = gr.Button('Generate', variant='primary')\n",
        "        with gr.Column():\n",
        "            output_prompt = gr.Textbox(label='Engineered Prompt', interactive=False)\n",
        "            output_image = gr.Image(label='Generated Image', type='pil')\n",
        "\n",
        "    btn.click(fn=gradio_generate, inputs=input_text, outputs=[output_prompt, output_image])\n",
        "\n",
        "# To launch the interface, uncomment and run:\n",
        "# demo.launch(inline=True)"
    ]

    # Cell 20 (index 20): Batch Processing
    nb['cells'][20]['source'] = [
        "import os\n",
        "import time\n",
        "\n",
        "results = []\n",
        "os.makedirs('images', exist_ok=True)\n",
        "\n",
        "print('🚀 Starting Batch Processing for 15 Products...')\n",
        "start_time = time.time()\n",
        "\n",
        "for idx, row in df.iterrows():\n",
        "    description = row['product_description']\n",
        "    pid = int(row['id'])\n",
        "    \n",
        "    print(f'[{pid}/15] Processing: {description}')\n",
        "    \n",
        "    # 1. Engineer Prompt\n",
        "    prompt = engineer_prompt(description)\n",
        "    \n",
        "    # 2. Generate Image\n",
        "    image = generate_image(prompt)\n",
        "    \n",
        "    # 3. Save Image (Constraint: must be images/img_X.png)\n",
        "    image_filename = f'img_{pid}.png'\n",
        "    image_path = os.path.join('images', image_filename)\n",
        "    image.save(image_path)\n",
        "    \n",
        "    results.append({\n",
        "        'id': pid,\n",
        "        'engineered_prompt': prompt\n",
        "    })\n",
        "\n",
        "# 4. Save submission.csv\n",
        "submission_df = pd.DataFrame(results)\n",
        "submission_df.to_csv('submission.csv', index=False)\n",
        "\n",
        "end_time = time.time()\n",
        "print(f'\\n✅ Batch Processing Complete! Total time: {end_time - start_time:.1f}s')\n",
        "print('📄 submission.csv saved!')\n",
        "print(\"🖼️  Images saved in 'images/' folder.\")"
    ]

    with open(file_path, 'w', encoding='utf-8') as f:
        nbf.write(nb, f)
    print(f'Successfully populated {file_path}')

if __name__ == '__main__':
    path = r'c:\Contents - NIAT x VGU\Masterclass\Masterclass 3 Project - Shresth Raj\starter notebook.ipynb'
    populate_starter(path)
