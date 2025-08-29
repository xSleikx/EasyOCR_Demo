import easyocr
import gradio as gr
from PIL import Image
import numpy as np
import torch
import cv2
from load_models import agent

# Agent global answer
agent_output = None
# Try to improve OCR with Agent
def agent_answer(result1, result2, result3):
    result_sync = agent.run_sync(
        f"You are given 3 texts {result1} {result2} {result3}, which were created using OCR and 3 different preprocessing steps."
        f"Your goal is to reproduce exactly the text from the image that was scanned using OCR."
        f"If words are missing in the sentence (only if the sentence no longer makes sense), try to find a suitable replacement word as accurately as possible."
        f"Mark any newly added words with *replacement word*."
        f"Your answer should also be in the same language as the texts."
    )
    agent_output = result_sync.output
    return agent_output

# Different preprocessing options
def preprocess_image(image: Image, method="fixed-threshold"):
    gray = np.array(image.convert('L'))

    if method == "fixed-threshold":
        # simple binary threshold
        threshold = 128
        bw = (gray < threshold).astype(np.uint8) * 255
        processed_gray = gray

    elif method == "adaptive-threshold":
        bw = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY, 11, 2
        )
        processed_gray = cv2.equalizeHist(gray)  # improve contrast

    elif method == "sharpen+adaptive":
        # sharpen first
        kernel = np.array([[0, -1, 0],
                           [-1, 5,-1],
                           [0, -1, 0]])
        sharpened = cv2.filter2D(gray, -1, kernel)
        bw = cv2.adaptiveThreshold(
            sharpened, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY, 11, 2
        )
        processed_gray = sharpened

    else:
        bw = gray
        processed_gray = gray

    return Image.fromarray(bw), Image.fromarray(processed_gray)

# Extract text with EasyOCR
def extract_text_from_image(image, method):
    if image is None or image.get("composite") is None:
        return "Upload an image first!", "", "", "", None, None, None
    try:
        image = image["composite"]
        
        pre_bw, pre_gray = preprocess_image(image, method)

        pre_bw_arr = np.array(pre_bw)
        pre_gray_arr = np.array(pre_gray)
        image_arr = np.array(image)

        reader = easyocr.Reader(['de', 'en'], gpu=torch.cuda.is_available())

        result1 = reader.readtext(pre_bw_arr, detail=0, paragraph=True, decoder="beamsearch")
        result2 = reader.readtext(pre_gray_arr, detail=0, paragraph=True, decoder="beamsearch")
        result3 = reader.readtext(image_arr, detail=0, paragraph=True, decoder="beamsearch")
        agent_output = agent_answer(result1, result2, result3)

        return result1, result2, result3, agent_output, pre_bw_arr, pre_gray_arr, image_arr
    except Exception as e:
        return f"An error occurred: {e}", "", "", "", None, None, None

# Gradio UI
with gr.Blocks(title="OCR Demo") as demo:
    gr.Markdown("## OCR Demo\nUpload or edit an image to extract text with EasyOCR.\n")

    with gr.Row():
        inp = gr.ImageEditor(type="pil", label="Upload or Edit Image")
        method = gr.Dropdown(
            ["fixed-threshold", "adaptive-threshold", "sharpen+adaptive"],
            label="Preprocessing Method",
            value="fixed-threshold"
        )

    with gr.Tab("OCR Results"):
        with gr.Tabs():
            with gr.Tab("B&W"):
                bw_text = gr.Textbox(label="OCR (Black & White)")
            with gr.Tab("Grayscale"):
                gray_text = gr.Textbox(label="OCR (Grayscale)")
            with gr.Tab("Original"):
                orig_text = gr.Textbox(label="OCR (Original)")
            with gr.Tab("Agent Answer"):
                agent_text = gr.Textbox(label="Agent")

    with gr.Accordion("Processed Images", open=False):
        bw_img = gr.Image(label="Black & White Image")
        gray_img = gr.Image(label="Grayscale Image")
        or_img = gr.Image(label="Original Image")

    # link function
    inputs = [inp, method]
    outputs = [bw_text, gray_text, orig_text, agent_text, bw_img, gray_img, or_img]

    inp.change(fn=extract_text_from_image, inputs=inputs, outputs=outputs)
    method.change(fn=extract_text_from_image, inputs=inputs, outputs=outputs)

demo.launch()
