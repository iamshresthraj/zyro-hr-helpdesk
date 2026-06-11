import nbformat as nbf
import os

def beautify_starter_notebook(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        nb = nbf.read(f, as_version=4)

    # Define premium markdown content for each section
    content = {
        0: (
            "<div align='center'>\n"
            "  <img src='https://img.shields.io/badge/NIAT%20Assignment-Masterclass_3-orange?style=for-the-badge' />\n"
            "  <h1>🎨 AI Product Image Generator with Gradio</h1>\n"
            "  <p><i>Official Starter Template — Optimized for T4 GPUs</i></p>\n"
            "</div>\n\n"
            "---\n\n"
            "### 🎯 Assignment Goal\n"
            "Design and implement a complete AI-powered product photography pipeline that transforms simple text into professional-grade visuals.\n\n"
            "**Requirements:**\n"
            "1. **LLM Integration**: Expert prompt engineering.\n"
            "2. **Diffusion Synthesis**: Studio-quality image generation.\n"
            "3. **Web Interface**: Interactive Gradio UI.\n\n"
            "**⚙️ Hardware Tip:** Enable GPU acceleration (**T4 x2**) in Settings before starting."
        ),
        1: "## 🛠️ Step 0: Environment & Verification\nWe initialize the environment, verify GPU availability, and install the core generative AI stack.",
        5: "## 📊 Step 1: Data Acquisition\nLoading the official evaluation dataset containing 15 high-potential product categories.",
        7: "## 🧠 Step 2: Intelligent Prompt Engineering\nTranslating user-friendly descriptions into rich, descriptive prompts for Stable Diffusion using `Qwen2.5`.",
        10: "## 🎨 Step 3: High-Fidelity Image Synthesis\nConfiguring the Stable Diffusion pipeline for photorealistic output with optimized memory usage.",
        13: "## ✨ Step 4: The Web Experience\nCreating a bridge between your models and the end-user with a modern Gradio interface.",
        15: "## 🚀 Step 5: Portfolio Generation\nExecuting the full pipeline across the entire dataset to generate evaluation-ready artifacts.",
        17: "## ⚖️ Step 6: Automated Evaluation\n*Compute scores using the official competition metrics.* **(Do not modify this section)**"
    }

    # Update markdown cells
    cells = nb['cells']
    cell_idx = 0
    for i in range(len(cells)):
        if cells[i]['cell_type'] == 'markdown':
            if cell_idx in content:
                cells[i]['source'] = content[cell_idx]
            cell_idx += 1

    # Save the updated notebook
    with open(file_path, 'w', encoding='utf-8') as f:
        nbf.write(nb, f)
    print(f'Successfully beautified {file_path}')

if __name__ == '__main__':
    path = r'c:\Contents - NIAT x VGU\Masterclass\Masterclass 3 Project - Shresth Raj\starter notebook.ipynb'
    beautify_starter_notebook(path)
