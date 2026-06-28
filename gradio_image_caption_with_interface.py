import gradio as gr
import torch
from transformers import (
    BlipProcessor,
    BlipForConditionalGeneration,
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
)

# Device
device = "cuda" if torch.cuda.is_available() else "cpu"


processor = BlipProcessor.from_pretrained(
    "Salesforce/blip-image-captioning-base"
)

caption_model = BlipForConditionalGeneration.from_pretrained(
    "Salesforce/blip-image-captioning-base"
).to(device)


translator_tokenizer = AutoTokenizer.from_pretrained(
    "Helsinki-NLP/opus-mt-en-ar"
)

translator_model = AutoModelForSeq2SeqLM.from_pretrained(
    "Helsinki-NLP/opus-mt-en-ar"
).to(device)


def generate_caption(image):
    inputs = processor(images=image, return_tensors="pt").to(device)
    output = caption_model.generate(**inputs)
    return processor.decode(output[0], skip_special_tokens=True)


def translate_to_arabic(text):
    inputs = translator_tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        padding=True
    ).to(device)

    output = translator_model.generate(**inputs)

    return translator_tokenizer.decode(
        output[0],
        skip_special_tokens=True
    )


def caption_image(image):
    try:
        english_caption = generate_caption(image)
        arabic_caption = translate_to_arabic(english_caption)

        return english_caption, arabic_caption

    except Exception as e:
        return "", str(e)


demo = gr.Interface(
    fn=caption_image,
    inputs=gr.Image(type="pil"),
    outputs=[
        gr.Textbox(label="English Caption"),
        gr.Textbox(label="Arabic Caption")
    ],
    title="BLIP Image Captioning",
    description="Upload an image to generate captions in English and Arabic."
)

demo.launch()
