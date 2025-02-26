from transformers import pipeline
from PIL import Image


checkpoint = "depth-anything/Depth-Anything-V2-base-hf"
pipe = pipeline("depth-estimation", model=checkpoint)

from PIL import Image
import sys
import os

def generate_depth(image_path):
    image = Image.open(image_path)
    depth_image = pipe(image)["depth"]
    depth_image.save(image_path[:-4] + "_depth_large.jpg")
    return 

folders_root = sys.argv[1]
folders = os.listdir(folders_root)
for folder in folders:
    image_root = os.path.join(folders_root, folder)
    images = sorted(os.listdir(image_root))
    image_paths = [os.path.join(image_root, image) for image in images if image.endswith('.png')]
    print(image_paths)
    for image_path in image_paths:
        generate_depth(image_path)
