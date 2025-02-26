import torch
import cv2
import numpy as np
from torchvision.transforms import Compose, ToTensor, Resize, Normalize
from midas.model_loader import load_model
import sys

def compute_normals(depth):
    """Compute surface normals from a depth map."""
    depth = depth.astype(np.float32)

    # Compute gradients
    dx = cv2.Sobel(depth, cv2.CV_32F, 1, 0, ksize=5)
    dy = cv2.Sobel(depth, cv2.CV_32F, 0, 1, ksize=5)

    # Compute normal components
    dz = np.ones_like(dx)  # Approximate "upward" component
    normals = np.dstack((-dx, -dy, dz))

    # Normalize
    norm = np.linalg.norm(normals, axis=2, keepdims=True)
    normals /= (norm + 1e-6)  # Avoid division by zero

    return (normals + 1) / 2  # Normalize to [0,1] for visualization

def estimate_normals(image_path):
    """Estimate surface normals from a single RGB image."""
    # Load MiDaS model
    model_type = "DPT_Large"  # Options: DPT_Large, DPT_Hybrid, MiDaS_small
    model, transform, device = load_model(model_type)

    # Load and preprocess image
    image = cv2.imread(image_path)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    img = transform(image_rgb).unsqueeze(0).to(device)

    # Estimate depth
    with torch.no_grad():
        depth = model(img)
    
    depth = depth.squeeze().cpu().numpy()

    # Convert depth to normal map
    normal_map = compute_normals(depth)
    
    return normal_map

# Example usage:
image_path = sys.argv[1]
normal_map = estimate_normals(image_path)

cv2.imshow("Estimated Normals", (normal_map * 255).astype(np.uint8))
cv2.waitKey(0)
