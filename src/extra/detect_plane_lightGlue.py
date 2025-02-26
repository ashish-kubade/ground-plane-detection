import cv2
import numpy as np
import open3d as o3d
import sys
import os

from lightglue import LightGlue, SuperPoint, DISK
from lightglue.utils import load_image, rbd
from lightglue import viz2d
import torch

image_root = sys.argv[1]
images = sorted(os.listdir(image_root))
image_paths = [os.path.join(image_root, image) for image in images if image.endswith('.png')]




def estimate_ground_plane_from_images(image_paths):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # 'mps', 'cpu'

    extractor = SuperPoint(max_num_keypoints=4096).eval().to(device)  # load the extractor
    matcher = LightGlue(features="superpoint").eval().to(device)  # load the matcher
    point_cloud = o3d.geometry.PointCloud()

    for i in range(len(image_paths) - 1):
        image0 = load_image(image_paths[i])
        image1 = load_image(image_paths[i + 1])
        feats0 = extractor.extract(image0.to(device))
        feats1 = extractor.extract(image1.to(device))
        matches01 = matcher({"image0": feats0, "image1": feats1})
        feats0, feats1, matches01 = [
            rbd(x) for x in [feats0, feats1, matches01]
        ]  # remove batch dimension

        kpts0, kpts1, matches = feats0["keypoints"], feats1["keypoints"], matches01["matches"]
        m_kpts0, m_kpts1 = kpts0[matches[..., 0]], kpts1[matches[..., 1]]
        axes = viz2d.plot_images([image0, image1])
        viz2d.plot_matches(m_kpts0, m_kpts1, color="lime", lw=0.2)
        viz2d.add_text(0, f'Stop after {matches01["stop"]} layers', fs=20)
        # Extract matched points
        src_pts = np.float32(m_kpts0.cpu().numpy()).reshape(-1, 1, 2)
        dst_pts = np.float32(m_kpts1.cpu().numpy()).reshape(-1, 1, 2)
        # Find essential matrix and recover pose
        E, _ = cv2.findEssentialMat(src_pts, dst_pts, focal=1.0, pp=(0, 0), method=cv2.RANSAC, prob=0.999, threshold=0.5)
        _, R, t, _ = cv2.recoverPose(E, src_pts, dst_pts)

        # Triangulate points
        proj1 = np.eye(3, 4)
        proj2 = np.hstack((R, t))
        points_4d = cv2.triangulatePoints(proj1, proj2, src_pts, dst_pts)
        points_3d = points_4d[:3] / points_4d[3]

        # Add points to point cloud
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points_3d.T)
        point_cloud += pcd

    # Downsample and segment ground plane
    
    plane_model, inliers = point_cloud.segment_plane(distance_threshold=1, ransac_n=3, num_iterations=5000)

    A, B, C, D = plane_model
    print(f"Estimated ground plane equation: {A:.3f}x + {B:.3f}y + {C:.3f}z + {D:.3f} = 0")

    inlier_cloud = point_cloud.select_by_index(inliers)
    outlier_cloud = point_cloud.select_by_index(inliers, invert=True)
    inlier_cloud.paint_uniform_color([0.0, 1.0, 0.0])
    outlier_cloud.paint_uniform_color([1.0, 0.0, 0.0])

    o3d.visualization.draw_geometries([inlier_cloud, outlier_cloud])

    return plane_model


estimate_ground_plane_from_images(image_paths)