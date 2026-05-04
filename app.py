import base64
from io import BytesIO

import gradio as gr
from PIL import Image

from ml.inference import DeepfakeDetector


DETECTOR = DeepfakeDetector()


def _decode_heatmap(heatmap_base64: str):
    if not heatmap_base64:
        return None
    payload = heatmap_base64.split(",", 1)[-1]
    return Image.open(BytesIO(base64.b64decode(payload))).convert("RGB")


def predict_image(image):
    if image is None:
        return "Upload an image to begin.", None

    buffer = BytesIO()
    image.convert("RGB").save(buffer, format="JPEG")
    result = DETECTOR.predict_image(buffer.getvalue())

    if not result.get("success"):
        return f"Prediction failed: {result.get('error', 'Unknown error')}", None

    output_text = (
        f"Prediction: {result['prediction']} | "
        f"Authenticity: {result['authenticity_score']:.2f}% | "
        f"Confidence: {result['confidence']:.2f}%"
    )
    return output_text, _decode_heatmap(result.get("heatmap_base64"))


def demo():
    with gr.Blocks(title='TrustVision Demo') as block:
        gr.Markdown('# TrustVision Image Authenticity Demo')
        with gr.Row():
            image_input = gr.Image(type='pil', label='Upload Image')
            with gr.Column():
                result = gr.Textbox(label='Prediction')
                heatmap = gr.Image(label='Model Attention Heatmap')
        image_input.change(predict_image, inputs=image_input, outputs=[result, heatmap])
    block.launch()


if __name__ == '__main__':
    demo()
