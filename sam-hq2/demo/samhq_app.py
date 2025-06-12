import gradio as gr
from gradio_image_prompter import ImagePrompter
import torch
import numpy as np
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor
import matplotlib.pyplot as plt
from PIL import Image as PILImage
from uuid import uuid4
import os

IMAGE = None
MASKS = None
MASKED_IMAGES = None
INDEX = None


checkpoint = "./checkpoints/sam2.1_hq_hiera_large.pt"
model_cfg = "configs/sam2.1/sam2.1_hq_hiera_l.yaml"
predictor = SAM2ImagePredictor(build_sam2(model_cfg, checkpoint))

# hq_token_only: False means use hq output to correct SAM output. 
#                True means use hq output only. 
#                Default: False
hq_token_only = False 
# To achieve best visualization effect, for images contain multiple objects (like typical coco images), we suggest to set hq_token_only=False
# For images contain single object, we suggest to set hq_token_only = True
# For quantiative evaluation on COCO/YTVOS/DAVIS/UVO/LVIS etc., we set hq_token_only = False

def prompter(prompts):
    # print("prompts", prompts)

    image = np.array(prompts["image"])  # Convert the image to a numpy array
    points = prompts["points"]  # Get the points from prompts
    print("points", points)

    input_box = None
    input_point = None
    input_label = None

    if points[0][2] == 2:
        input_box = [[point[0], point[1], point[3], point[4]] for point in points]
        print("input_box", input_box)

    else:
        input_point = [[point[0], point[1]] for point in points]
        input_label = [point[2] for point in points]  # Assuming all points are foreground
        print("input_point", input_point)
        print("input_label", input_label)

    predictor.set_image(image)
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
        masks, scores, logits = predictor.predict(point_coords=input_point,
                                        point_labels=input_label,
                                        box=input_box,
                                        multimask_output=True, hq_token_only=hq_token_only)

    print("masks.shape", masks.shape)

    # # Prepare individual images with separate overlays
    overlay_images = []
    for i, mask in enumerate(masks):
        print(f"Predicted Mask {i+1}:", mask.shape)
        red_mask = np.zeros_like(image)
        red_mask[:, :, 0] = mask.astype(np.uint8) * 255  # Apply the red channel
        red_mask = PILImage.fromarray(red_mask)

        # Convert the original image to a PIL image
        original_image = PILImage.fromarray(image)

        # Blend the original image with the red mask
        blended_image = PILImage.blend(original_image, red_mask, alpha=0.5)

        # Add the blended image to the list
        overlay_images.append(blended_image)

    global IMAGE, MASKS, MASKED_IMAGES
    IMAGE, MASKS = image, masks
    MASKED_IMAGES = [np.array(img) for img in overlay_images]

    return overlay_images[0], overlay_images[1], overlay_images[2], masks
    # return image, image, image, image



with gr.Blocks() as demo:
    gr.Markdown("# HQ SAM 2 Demo (sam2.1_hq_hiera_large)")

    with gr.Row():
        with gr.Column():
            image_input = gr.State()
            # Input: ImagePrompter for uploaded image
            upload_image_input = ImagePrompter(show_label=False)

            submit_button = gr.Button("Submit")

    with gr.Row():
        with gr.Column():
            # Outputs: Up to 3 overlay images
            image_output_1 = gr.Image(show_label=False)
        with gr.Column():
            image_output_2 = gr.Image(show_label=False)
        with gr.Column():
            image_output_3 = gr.Image(show_label=False)


    # Logic to use uploaded image
    upload_image_input.change(
        fn=lambda img: img, inputs=upload_image_input, outputs=image_input
    )
    # Define the action triggered by the submit button
    submit_button.click(
        fn=prompter,
        inputs=upload_image_input,  # The final image input (whether uploaded or random)
        outputs=[image_output_1, image_output_2, image_output_3, gr.State()],
        show_progress=True,
    )

# Launch the Gradio app
demo.launch()