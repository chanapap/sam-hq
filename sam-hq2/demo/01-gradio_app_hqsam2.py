import gradio as gr
import torch
import numpy as np
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor
from PIL import Image as PILImage
import os

# ==============================
# SAM2 HQ Setup
# ==============================
checkpoint = "./checkpoints/sam2.1_hq_hiera_large.pt"
model_cfg = "configs/sam2.1/sam2.1_hq_hiera_l.yaml"
predictor = SAM2ImagePredictor(build_sam2(model_cfg, checkpoint))

hq_token_only = False

# ==============================
# Globals for points/labels
# ==============================
points = []
labels = []

def reset_points_labels():
    """Reset click history"""
    global points, labels
    points, labels = [], []
    return str(points), str(labels)

# ==============================
# SAM2 Inference Function
# ==============================
def run_sam2_segmentation(image, points, labels):
    if len(points) == 0:
        return None, None, None, None  # No points selected

    predictor.set_image(np.array(image))

    input_point = np.array(points)
    input_label = np.array(labels)

    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
        masks, scores, logits = predictor.predict(
            point_coords=input_point,
            point_labels=input_label,
            box=None,
            multimask_output=True,
            hq_token_only=hq_token_only
        )

    overlay_images = []
    for i, mask in enumerate(masks):
        red_mask = np.zeros_like(np.array(image))
        red_mask[:, :, 0] = mask.astype(np.uint8) * 255
        blended = PILImage.blend(PILImage.fromarray(np.array(image)),
                                 PILImage.fromarray(red_mask), alpha=0.5)
        overlay_images.append(blended)

    return overlay_images[0], overlay_images[1], overlay_images[2], masks

# ==============================
# Click Handler
# ==============================
def get_select_coords(img, evt: gr.SelectData, label):
    global points, labels
    pixel_coords = [evt.index[0], evt.index[1]]
    points.append(pixel_coords)
    labels.append(1 if label == "Positive" else 0)

    out1, out2, out3, masks = run_sam2_segmentation(img, points, labels)
    return str(points), str(labels), out1, out2, out3

# ==============================
# Gradio UI
# ==============================
with gr.Blocks() as demo:
    gr.Markdown("# HQ SAM2.1 Point-Based Segmentation")

    with gr.Row():
        input_img = gr.Image(label="Input Image", type="numpy", height=1000)

    with gr.Row():
        label_selector = gr.Radio(["Positive", "Negative"], label="Select Label", value="Positive")
        output_points = gr.Textbox(label="Points", interactive=True, value=str(points))
        output_labels = gr.Textbox(label="Labels", interactive=True, value=str(labels))

    with gr.Row():
        run_button = gr.Button("RUN")
        reset_button = gr.Button("RESET")

    with gr.Row():
        image_output_1 = gr.Image(label="Mask 1")
        image_output_2 = gr.Image(label="Mask 2")
        image_output_3 = gr.Image(label="Mask 3")

    # Bind click events
    input_img.select(get_select_coords, [input_img, label_selector],
                     [output_points, output_labels, image_output_1, image_output_2, image_output_3])

    # Run button (manual trigger)
    run_button.click(fn=lambda img: run_sam2_segmentation(img, points, labels),
                     inputs=[input_img],
                     outputs=[image_output_1, image_output_2, image_output_3, gr.State()])

    # Reset button
    reset_button.click(reset_points_labels, None, [output_points, output_labels])

demo.launch()
