import numpy as np
import torch
import matplotlib.pyplot as plt
import cv2
from sam2.build_sam import build_sam2
from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator
import os

# Function to overlay masks on image
def show_anns(anns, image, borders=True):
    if len(anns) == 0:
        return image

    sorted_anns = sorted(anns, key=(lambda x: x['area']), reverse=True)
    img = np.ones((*sorted_anns[0]['segmentation'].shape, 4), dtype=np.float32)
    img[:, :, 3] = 0  # Transparent background

    for ann in sorted_anns:
        m = ann['segmentation']
        color_mask = np.concatenate([np.random.random(3), [0.5]])
        img[m] = color_mask
        if borders:
            contours, _ = cv2.findContours(m.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            contours = [cv2.approxPolyDP(contour, epsilon=0.01, closed=True) for contour in contours]
            cv2.drawContours(img, contours, -1, (0, 0, 1, 0.4), thickness=1)

    fig, ax = plt.subplots(figsize=(12, 12))
    ax.imshow(image)
    ax.imshow(img)
    ax.axis("off")

    # Return figure so it can be saved
    return fig

def show_anns_on_black_bg(anns, image_shape, borders=True):
    if len(anns) == 0:
        return None

    height, width = image_shape[:2]
    img = np.zeros((height, width, 4), dtype=np.float32)  # Black background (alpha=0)

    sorted_anns = sorted(anns, key=(lambda x: x['area']), reverse=True)

    for ann in sorted_anns:
        m = ann['segmentation']
        color_mask = np.concatenate([np.random.random(3), [0.7]])  # Random RGB + alpha
        img[m] = color_mask
        if borders:
            contours, _ = cv2.findContours(m.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            contours = [cv2.approxPolyDP(contour, epsilon=0.01, closed=True) for contour in contours]
            cv2.drawContours(img, contours, -1, (1, 1, 1, 1), thickness=1)  # White border

    # Create the figure with black background
    fig, ax = plt.subplots(figsize=(12, 12))
    fig.patch.set_facecolor('black')
    ax.set_facecolor('black')
    ax.imshow(img)
    ax.axis("off")

    return fig


# Paths and model setup
input_folder = r"D:\3d-recon\RoomSceneSegmentation\RoomSceneImage-40"
result_path = r"D:\3d-recon\sam-hq\output_automask"
os.makedirs(result_path, exist_ok=True)

checkpoint = "./checkpoints/sam2.1_hq_hiera_large.pt"
model_cfg = "configs/sam2.1/sam2.1_hq_hiera_l.yaml"
mask_generator = SAM2AutomaticMaskGenerator(
    build_sam2(model_cfg, checkpoint),
    pred_iou_thresh=0.7,
    points_per_batch=8,
    stability_score_thresh=0.9,
    hq_token_only=False
)

# Loop through all JPG images
for filename in os.listdir(input_folder):
    if filename.lower().endswith(".jpg"):
        print(f"Processing: {filename}")
        input_path = os.path.join(input_folder, filename)
        output_path = os.path.join(result_path, filename)

        # Load and convert image
        image = cv2.imread(input_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Run mask generation
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
            masks = mask_generator.generate(image)

        # Visualize and save output
        fig = show_anns_on_black_bg(masks, image.shape)

        fig.savefig(output_path, bbox_inches='tight', pad_inches=0)
        plt.close(fig)

        print(f"Saved to: {output_path}")
