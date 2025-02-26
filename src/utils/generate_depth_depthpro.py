import requests
from PIL import Image
import torch
from transformers import DepthProImageProcessorFast, DepthProForDepthEstimation
import sys
import os

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

device = "cpu"
image_processor = DepthProImageProcessorFast.from_pretrained("apple/DepthPro-hf")
model = DepthProForDepthEstimation.from_pretrained("apple/DepthPro-hf").to(device)
model.eval()
folders_root = sys.argv[1]
folders = os.listdir(folders_root)
for folder in folders:
    image_root = os.path.join(folders_root, folder)
    images = sorted(os.listdir(image_root))
    image_paths = [os.path.join(image_root, image) for image in images if image.endswith('.png')]
    print(image_paths)
    for image_path in image_paths:
        image = Image.open(image_path)
        inputs = image_processor(images=image, return_tensors="pt").to(device)

        with torch.no_grad():
            outputs = model(**inputs)

        post_processed_output = image_processor.post_process_depth_estimation(
            outputs, target_sizes=[(image.height, image.width)],
        )

        field_of_view = post_processed_output[0]["field_of_view"]
        focal_length = post_processed_output[0]["focal_length"]
        depth = post_processed_output[0]["predicted_depth"]
        depth = (depth - depth.min()) / depth.max()
        depth = depth * 255.
        depth = depth.detach().cpu().numpy()
        depth = Image.fromarray(depth.astype("uint8"))

        depth.save(image_path[:-4] + "_depthpro.jpg")