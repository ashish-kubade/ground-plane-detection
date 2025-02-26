import os
import sys

from networkx import general_random_intersection_graph
import torch
import numpy as np
from argparse import ArgumentParser
import cv2

from depth_generation import DepthGeneartor, DepthGeneratorDepthPro, DepthGeneratorDepthAnything
from pcd_generation import PCD_Generator
from detect_plane import detect_plane
import open3d as o3d
import time
#instantiate models

def run(parser, device):
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
    plane_detection_results = detect_plane(image_path, depth_results, pcd_generator, use_defaults, save_pcd=save_pcd)
    time_e = time.time()

    inlier_cloud = plane_detection_results['inlier_cloud']
    outlier_cloud = plane_detection_results['outlier_cloud']
    plane_model = plane_detection_results['plane_model']

    A, B, C, D = plane_model
    print(f"Estimated ground plane equation: {A:.3f}x + {B:.3f}y + {C:.3f}z + {D:.3f} = 0")
    print('Time taken: ', time_e - time_s)

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

    run(parser, device)