import cv2
import numpy as np
import open3d as o3d
import sys
import os

image_root = sys.argv[1]
images = sorted(os.listdir(image_root))
image_paths = [os.path.join(image_root, image) for image in images]

def detect_lines(image_path):
    image = cv2.imread(image_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150, apertureSize=3)
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=100, minLineLength=50, maxLineGap=10)
    line_image = image.copy()
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            cv2.line(line_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
    cv2.imshow("Detected Lines", line_image)
    out_image_path = image_path[:-4] + "_lines.jpg"
    cv2.imwrite(out_image_path, line_image)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

# Detect lines in each image
folders_root = sys.argv[1]
folders = os.listdir(folders_root)
for folder in folders:
    image_root = os.path.join(folders_root, folder)
    images = sorted(os.listdir(image_root))
    image_paths = [os.path.join(image_root, image) for image in images]
    print(image_paths)
    for image_path in image_paths:
        detect_lines(image_path)