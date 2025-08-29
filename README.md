# OCR Demo with EasyOCR and Agent Assistance

This project is a **web-based OCR demo** that leverages **EasyOCR** and an intelligent **agent** to enhance text extraction from images. By applying multiple preprocessing methods and combining their outputs, the agent aims to maximize OCR accuracy.

## Features

- **Interactive Image Upload:** Upload or edit images directly in your browser.
- **Flexible Preprocessing Options:**  
  - **Fixed Threshold**  
  - **Adaptive Threshold**  
  - **Sharpen + Adaptive Threshold**
- **Text Extraction:** Use **EasyOCR** to extract text from images.
- **Agent-Enhanced OCR:** Automatically combines outputs from different preprocessing methods to improve accuracy.
- **Comprehensive Results:** View OCR outputs separately for:  
  - Black & White preprocessed image  
  - Grayscale preprocessed image  
  - Original image  
  - Agent-enhanced final output
- **Processed Image Preview:** Check intermediate preprocessing results in real time.

## Installation

Install the required dependencies:

```bash
pip install -r requirements.txt
```

Create a `.env` file in the project root with your Groq API key:

```env
GROQ_API_KEY=your_groq_api_key_here
```

EasyOCR installation instructions can be found [here](https://github.com/JaidedAI/EasyOCR).

Ensure your `load_models.py` and `agent` object are correctly configured. The agent uses **Groq models** via [ai.pydantic.dev](https://ai.pydantic.dev/), but you can customize it if needed.



## Usage

Launch the Gradio demo:

```bash
python app.py
```

- Open the URL displayed in the console (usually `http://127.0.0.1:7860/`).  
- Upload or edit an image.  
- Select a preprocessing method.  
- View OCR results in the respective tabs.  
- Preview processed images under the **Processed Images** accordion.  

## How It Works

1. **Preprocessing**  
   - Converts the image to grayscale.  
   - Applies the selected preprocessing method (thresholding or sharpening).

2. **OCR Extraction**  
   - EasyOCR extracts text from:  
     - Black & White image  
     - Grayscale image  
     - Original image

3. **Agent Assistance**  
   - Combines all OCR outputs into a single, refined result.  
   - Attempts to reproduce the exact text from the image.  
   - Suggests replacements for missing or unclear words, marking them as `*replacement word*`.

## Notes

- GPU acceleration is used automatically if available (`torch.cuda.is_available()`).  
- Agent output may vary depending on text complexity and language.  
- Currently supported languages: English (`en`) and German (`de`).

## License

MIT License

