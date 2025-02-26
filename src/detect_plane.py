import os
import sys
import torch
import numpy as np
from argparse import ArgumentParser
import cv2

from depth_generation import DepthGeneartor, DepthGeneratorDepthPro, DepthGeneratorDepthAnything
from pcd_generation import PCD_Generator
import open3d as o3d
import time
#instantiate models

def detect_plane(parser, device):
    image_path = parser.image_path
    save_depth = parser.save_depth
    save_pcd = parser.save_pcd
    use_defaults = parser.use_defaults

    print('Initializing Models ... ')
    if parser.depth_method == "depth_pro":
        depth_generator = DepthGeneratorDepthPro(device=device)
 
    else:
       depth_generator = DepthGeneratorDepthAnything(device=device)
    
    pcd_generator = PCD_Generator()
    print('Models initialized !!!')
    time_s = time.time()
    depth_results = depth_generator.run(image_path=image_path,
                                        save_depth=save_depth
                                        )
    rgb_image = cv2.imread(image_path)
    if depth_results["default_intrinsics"] or use_defaults:
        fx, fy = 525.0, 525.0    # Focal length in pixels
    else:
        fx, fy = depth_results["focal_length"], depth_results["focal_length"]

    cx, cy = rgb_image.shape[1] // 2, rgb_image.shape[0] // 2  # Principal point
    intrinsics = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])
    


    depth_image = depth_results['depth']
    pcd = pcd_generator.run(image_path, depth_image=depth_image, intrinsics=intrinsics, save_pcd=save_pcd)

    plane_model, inliers = pcd.segment_plane(distance_threshold=0.1, ransac_n=3, num_iterations=5000)
    A, B, C, D = plane_model
    time_e = time.time()
    print(f"Estimated ground plane equation: {A:.3f}x + {B:.3f}y + {C:.3f}z + {D:.3f} = 0")
    print(f"Time taken: {time_e - time_s} seconds")

    inlier_cloud = pcd.select_by_index(inliers)
    outlier_cloud = pcd.select_by_index(inliers, invert=True)
    inlier_cloud.paint_uniform_color([0.0, 1.0, 0.0])
    o3d.visualization.draw_geometries([inlier_cloud, outlier_cloud])

           

if __name__ == "__main__":
    argparse = ArgumentParser()
    argparse.add_argument("--image_path", required=True, type=str)
    argparse.add_argument("--depth_method", default="depth_pro", type=str)
    argparse.add_argument("--depth_image_path", default=None, type=str)
    argparse.add_argument("--save_depth", default=True, type=bool)
    argparse.add_argument("--save_pcd", default=True, type=bool)
    argparse.add_argument("--use_defaults", default=True, type=bool)

    parser = argparse.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    detect_plane(parser, device)