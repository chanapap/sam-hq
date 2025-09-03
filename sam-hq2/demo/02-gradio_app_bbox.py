import gradio as gr
import torch
import numpy as np
from PIL import Image as PILImage
from gradio_image_prompter import ImagePrompter
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor

# ==============================
# SAM2 HQ Setup
# ==============================
checkpoint = "./checkpoints/sam2.1_hq_hiera_large.pt"
model_cfg = "configs/sam2.1/sam2.1_hq_hiera_l.yaml"
predictor = SAM2ImagePredictor(build_sam2(model_cfg, checkpoint))
hq_token_only = False

# ==============================
# Run SAM2 segmentation
# ==============================
def run_sam2(image, prompts):
    if not prompts:
        return image, image, image  # nothing to process

    point_coords, point_labels = None, None
    box = None

    if "point" in prompts:
        point_coords = np.array([pt for _, pt in prompts["point"]])
        point_labels = np.array([lbl for lbl, _ in prompts["point"]])

    if "bbox" in prompts:
        box = np.array(prompts["bbox"][0])

    predictor.set_image(np.array(image))

    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
        masks, scores, _ = predictor.predict(
            point_coords=point_coords,
            point_labels=point_labels,
            box=box,
            multimask_output=True,
            hq_token_only=hq_token_only
        )

    overlay_images = []
    for mask in masks:
        red_mask = np.zeros_like(np.array(image))
        red_mask[:, :, 0] = mask.astype(np.uint8) * 255
        blended = PILImage.blend(PILImage.fromarray(np.array(image)),
                                 PILImage.fromarray(red_mask), alpha=0.5)
        overlay_images.append(blended)

    return overlay_images[0], overlay_images[1], overlay_images[2]

# ==============================
# Process ImagePrompter input on Run
# ==============================
def process_prompts(img_with_prompts, prompts):
    image, img_prompts = img_with_prompts['image'], img_with_prompts['points']
    point_prompts, box_prompts = [], []

    for prompt in img_prompts:
        prompt = [int(p) for p in prompt]
        if prompt[2] == 2 and prompt[5] == 3:  # box
            box_prompts = [[prompt[0], prompt[1], prompt[3], prompt[4]]]
        elif prompt[2] == 1 and prompt[5] == 4:  # positive point
            point_prompts.append((1, (prompt[0], prompt[1])))
        elif prompt[2] == 0 and prompt[5] == 4:  # negative point
            point_prompts.append((0, (prompt[0], prompt[1])))

    if len(point_prompts) > 0:
        prompts['point'] = point_prompts
    elif 'point' in prompts:
        del prompts['point']

    if len(box_prompts) > 0:
        prompts['bbox'] = box_prompts
    elif 'bbox' in prompts:
        del prompts['bbox']

    mask_outputs = run_sam2(image, prompts)
    return *mask_outputs, prompts  # <- include prompts state as last output

# ==============================
# Gradio UI
# ==============================
prompts = gr.State(dict())

with gr.Blocks() as demo:
    gr.Markdown("# SAM2 HQ Segmentation with Points & Bounding Box")

    # State to store prompts
    prompts = gr.State(dict())

    img_prompter = ImagePrompter(
        label="Draw Points (Left=Positive, Right=Negative) or Drag a Box",
        sources="upload"
    )

    run_btn = gr.Button("Run")

    mask1 = gr.Image(label="Mask 1")
    mask2 = gr.Image(label="Mask 2")
    mask3 = gr.Image(label="Mask 3")

    run_btn.click(
        process_prompts,
        inputs=[img_prompter, prompts],
        outputs=[mask1, mask2, mask3, prompts]  # include prompts as output
    )

demo.launch()
