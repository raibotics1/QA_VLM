import os
from pathlib import Path
import torch
import gradio as gr
from transformers import (
    AutoProcessor,
    AutoModelForImageTextToText
)

# =====================================================
# CONFIG
# =====================================================

MODEL_NAME = os.getenv("MODEL_NAME")

HF_TOKEN = os.getenv("HF_TOKEN", None)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.float16 if DEVICE == "cuda" else torch.float32

# =====================================================
# LOAD MODEL
# =====================================================

def load_model():
    print(DEVICE)
    processor = AutoProcessor.from_pretrained(
        MODEL_NAME,
        token=HF_TOKEN,
        trust_remote_code=True
    )

    model = AutoModelForImageTextToText.from_pretrained(
        MODEL_NAME,
        token=HF_TOKEN,
        trust_remote_code=True,
        device_map="auto",
        dtype=DTYPE
    )

    model.eval()
    return processor, model


processor, model = load_model()

# =====================================================
# STATE
# =====================================================

current_image = None

# =====================================================
# PROMPT BUILDER
# =====================================================

def build_prompt(user_text: str):
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": user_text}
            ]
        }
    ]

    return processor.apply_chat_template(
        messages,
        add_generation_prompt=True
    )

# =====================================================
# DECODE (КЛЮЧЕВОЕ ИСПРАВЛЕНИЕ)
# =====================================================

def decode_answer(output_ids, input_ids):
    generated = output_ids[0][input_ids.shape[-1]:]
    return processor.tokenizer.decode(
        generated,
        skip_special_tokens=True
    ).strip()

# =====================================================
# TASKS
# =====================================================

def caption():
    if current_image is None:
        return "❌ Сначала загрузите изображение"

    prompt = build_prompt("Describe the image.")

    inputs = processor(
        images=current_image,
        text=prompt,
        return_tensors="pt"
    ).to(model.device)

    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=128,
            do_sample=True,
            temperature=0.7,
            top_p=0.9
        )

    return decode_answer(output, inputs["input_ids"])


def vqa(question):
    if current_image is None:
        return "❌ Сначала загрузите изображение"

    if not question or not question.strip():
        return "❌ Введите вопрос"

    prompt = build_prompt(question)

    inputs = processor(
        images=current_image,
        text=prompt,
        return_tensors="pt"
    ).to(model.device)

    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=128,
            do_sample=True,
            temperature=0.7,
            top_p=0.9
        )

    return decode_answer(output, inputs["input_ids"])


def run_ocr():
    if current_image is None:
        return None, "❌ Сначала загрузите изображение"

    prompt = build_prompt(
        "Extract all text from the image. Return plain text only."
    )

    inputs = processor(
        images=current_image,
        text=prompt,
        return_tensors="pt"
    ).to(model.device)

    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=512,
            do_sample=False
        )

    text = decode_answer(output, inputs["input_ids"])

    out_file = Path("ocr_result.txt")
    out_file.write_text(text, encoding="utf-8")

    return str(out_file), text

# =====================================================
# UI
# =====================================================

with gr.Blocks(theme=gr.themes.Soft(), title="SmolVLM2 Vision Demo") as app:
    gr.Markdown(
        """
        # SmolVLM2-2.2B Vision Demo

        • Image Captioning  
        • Visual Question Answering  
        • Optical Character Recognition  
        """
    )

    with gr.Row():
        with gr.Column(scale=1):
            image_input = gr.Image(
                type="pil",
                label="Изображение"
            )
            image_status = gr.Markdown("")

            def set_image(img):
                global current_image
                if img is None:
                    return "❌ Некорректный файл"
                current_image = img
                return "✅ Изображение загружено"

            image_input.change(
                set_image,
                image_input,
                image_status
            )

        with gr.Column(scale=2):
            with gr.Tab("Caption"):
                caption_btn = gr.Button("Сгенерировать описание")
                caption_out = gr.Textbox(lines=4)
                caption_btn.click(
                    caption,
                    outputs=caption_out
                )

            with gr.Tab("VQA"):
                question = gr.Textbox(
                    label="Вопрос",
                    placeholder="Что изображено на картинке?"
                )
                vqa_btn = gr.Button("Ответить")
                vqa_out = gr.Textbox(lines=4)
                vqa_btn.click(
                    vqa,
                    inputs=question,
                    outputs=vqa_out
                )

            with gr.Tab("OCR"):
                ocr_btn = gr.Button("Распознать текст")
                ocr_file = gr.File(label="Файл")
                ocr_text = gr.Textbox(lines=8)
                ocr_btn.click(
                    run_ocr,
                    outputs=[ocr_file, ocr_text]
                )

# =====================================================
# LAUNCH
# =====================================================

app.launch(
    server_name="0.0.0.0",
    server_port=int(os.getenv("GRADIO_PORT", 7860))
)
