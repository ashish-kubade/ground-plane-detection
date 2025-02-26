import cv2
import numpy as np
import open3d as o3d
import sys
import os

image_root = sys.argv[1]
images = sorted(os.listdir(image_root))
image_paths = [os.path.join(image_root, image) for image in images if image.endswith('.png')]

def estimate_ground_plane_from_images(image_paths):
    sift = cv2.SIFT_create()
    bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)

    point_cloud = o3d.geometry.PointCloud()

    for i in range(len(image_paths) - 1):
        img1 = cv2.imread(image_paths[i])
        img2 = cv2.imread(image_paths[i + 1])

        gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
        gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

        orb = cv2.ORB_create()
        # Detect keypoints and descriptors
        kp1, des1 = sift.detectAndCompute(gray1, None)
        kp2, des2 = sift.detectAndCompute(gray2, None)



        # Match descriptors
        matches = bf.match(des1, des2)
        matches = sorted(matches, key=lambda x: x.distance)

        if len(matches) < 4:
            print(f"Not enough matches found for view {i} and {i + 1}. Skipping...")
            continue
        # Draw matches on images
        matched_image = cv2.drawMatches(img1, kp1, img2, kp2, matches, None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
        cv2.imshow(f"Matches between view {i} and {i + 1}", matched_image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


        # Extract matched points
        src_pts = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
        dst_pts = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)
        print(src_pts.shape, dst_pts.shape)
        # Find essential matrix and recover pose
        E, _ = cv2.findEssentialMat(src_pts, dst_pts, focal=1.0, pp=(0, 0), method=cv2.RANSAC, prob=0.999, threshold=1.0)
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
    
    plane_model, inliers = point_cloud.segment_plane(distance_threshold=0.1, ransac_n=3, num_iterations=5000)

    A, B, C, D = plane_model
    print(f"Estimated ground plane equation: {A:.3f}x + {B:.3f}y + {C:.3f}z + {D:.3f} = 0")

    inlier_cloud = point_cloud.select_by_index(inliers)
    outlier_cloud = point_cloud.select_by_index(inliers, invert=True)
    inlier_cloud.paint_uniform_color([0.0, 1.0, 0.0])
    outlier_cloud.paint_uniform_color([1.0, 0.0, 0.0])

    o3d.visualization.draw_geometries([inlier_cloud, outlier_cloud])

    return plane_model


estimate_ground_plane_from_images(image_paths)