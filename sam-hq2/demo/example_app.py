import gradio as gr
from gradio_image_prompter import ImagePrompter
import torch
import numpy as np
from sam2.sam2_image_predictor import SAM2ImagePredictor
from uuid import uuid4
import os
from huggingface_hub import upload_folder, login
from PIL import Image as PILImage
from datasets import Dataset, Features, Array2D, Image
import shutil
import random
from datasets import load_dataset

MODEL = "facebook/sam2-hiera-large"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
PREDICTOR = SAM2ImagePredictor.from_pretrained(MODEL, device=DEVICE)

DESTINATION_DS = "amaye15/object-segmentation"


token = os.getenv("TOKEN")
if token:
    login(token)

IMAGE = None
MASKS = None
MASKED_IMAGES = None
INDEX = None


ds_name = ["amaye15/product_labels"]  #  "amaye15/Products-10k", "amaye15/receipts"
choices = ["test", "train"]
max_len = None

ds_stream = load_dataset(random.choice(ds_name), streaming=True)


ds_split = ds_stream[random.choice(choices)]

ds_iter = ds_split.iter(batch_size=1)

for idx, val in enumerate(ds_iter):
    max_len = idx


def prompter(prompts):

    image = np.array(prompts["image"])  # Convert the image to a numpy array
    points = prompts["points"]  # Get the points from prompts

    # Perform inference with multimask_output=True
    with torch.inference_mode():
        PREDICTOR.set_image(image)
        input_point = [[point[0], point[1]] for point in points]
        input_label = [1] * len(points)  # Assuming all points are foreground
        masks, _, _ = PREDICTOR.predict(
            point_coords=input_point, point_labels=input_label, multimask_output=True
        )

    # Prepare individual images with separate overlays
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


def select_mask(
    selected_mask_index,
    mask1,
    mask2,
    mask3,
):
    masks = [mask1, mask2, mask3]
    global INDEX
    INDEX = selected_mask_index
    return masks[selected_mask_index]


def save_selected_mask(image, mask, output_dir="output"):

    output_dir = os.path.join(os.getcwd(), output_dir)

    os.makedirs(output_dir, exist_ok=True)

    folder_id = str(uuid4())

    folder_path = os.path.join(output_dir, folder_id)

    os.makedirs(folder_path, exist_ok=True)

    data_path = os.path.join(folder_path, "data.parquet")

    data = {
        "image": IMAGE,
        "masked_image": MASKED_IMAGES[INDEX],
        "mask": MASKS[INDEX],
    }

    features = Features(
        {
            "image": Image(),
            "masked_image": Image(),
            "mask": Array2D(
                dtype="int64", shape=(MASKS[INDEX].shape[0], MASKS[INDEX].shape[1])
            ),
        }
    )

    ds = Dataset.from_list([data], features=features)
    ds.to_parquet(data_path)

    upload_folder(
        folder_path=output_dir,
        repo_id=DESTINATION_DS,
        repo_type="dataset",
    )

    shutil.rmtree(folder_path)

    iframe_code = """## Success! 🎉🤖✅
You've successfully contributed to the dataset. 
Please note that because new data has been added to the dataset, it may take a couple of minutes to render. 
Check it out here:
[Object Segmentation Dataset](https://huggingface.co/datasets/amaye15/object-segmentation)
"""

    return iframe_code


def get_random_image():
    """Get a random image from the dataset."""
    global max_len
    random_idx = random.choice(range(max_len))
    image_data = list(ds_split.skip(random_idx).take(1))[0]["pixel_values"]
    formatted_image = {
        "image": np.array(image_data),
        "points": [],
    }  # Create the correct format
    return formatted_image


# Define the Gradio Blocks app
with gr.Blocks() as demo:
    gr.Markdown("# Object Segmentation- Image Point Collector and Mask Overlay Tool")
    gr.Markdown(
        """
        This application utilizes **Segment Anything V2 (SAM2)** to allow you to upload an image or select a random image from a dataset and interactively generate segmentation masks based on multiple points you select on the image. 
        ### How It Works:
        1. **Upload or Select an Image**: You can either upload your own image or use a random image from the dataset.
        2. **Point Selection**: Click on the image to indicate points of interest. You can add multiple points, and these will be used collectively to generate segmentation masks using SAM2.
        3. **Mask Generation**: The app will generate up to three different segmentation masks for the selected points, each displayed separately with a red overlay.
        4. **Mask Selection**: Carefully review the generated masks and select the one that best fits your needs. **It's important to choose the correct mask, as your selection will be saved and used for further processing.**
        5. **Save and Contribute**: Save the selected mask along with the image to a dataset, contributing to a shared dataset on Hugging Face.
        **Disclaimer**: All images and masks you work with will be collected and stored in a public dataset. Please ensure that you are comfortable with your selections and the data you provide before saving.
        
        This tool is particularly useful for creating precise object segmentation masks for computer vision tasks, such as training models or generating labeled datasets.
        """
    )

    with gr.Row():
        with gr.Column():
            image_input = gr.State()
            # Input: ImagePrompter for uploaded image
            upload_image_input = ImagePrompter(show_label=False)

            random_image_button = gr.Button("Use Random Image")

            submit_button = gr.Button("Submit")

    with gr.Row():
        with gr.Column():
            # Outputs: Up to 3 overlay images
            image_output_1 = gr.Image(show_label=False)
        with gr.Column():
            image_output_2 = gr.Image(show_label=False)
        with gr.Column():
            image_output_3 = gr.Image(show_label=False)

    # Dropdown for selecting the correct mask
    with gr.Row():
        mask_selector = gr.Radio(
            label="Select the correct mask",
            choices=["Mask 1", "Mask 2", "Mask 3"],
            type="index",
        )
        # selected_mask_output = gr.Image(show_label=False)

    save_button = gr.Button("Save Selected Mask and Image")
    iframe_display = gr.Markdown()

    # Logic for the random image button
    random_image_button.click(
        fn=get_random_image,
        inputs=None,
        outputs=upload_image_input,  # Pass the formatted random image to ImagePrompter
    )

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

    # Define the action triggered by mask selection
    mask_selector.change(
        fn=select_mask,
        inputs=[mask_selector, image_output_1, image_output_2, image_output_3],
        outputs=gr.State(),
    )

    # Define the action triggered by the save button
    save_button.click(
        fn=save_selected_mask,
        inputs=[gr.State(), gr.State()],
        outputs=iframe_display,
        show_progress=True,
    )

# Launch the Gradio app
demo.launch()
