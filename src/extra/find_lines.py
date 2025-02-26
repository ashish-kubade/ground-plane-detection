import cv2
import numpy as np
import sys

def find_long_lines(image_path, min_length=100):
    # Load image
    img = cv2.imread(image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Apply edge detection
    edges = cv2.Canny(gray, 50, 150, apertureSize=3)

    # Detect lines using Hough Transform
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=100, minLineLength=min_length, maxLineGap=10)

    # Draw detected lines
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            cv2.line(img, (x1, y1), (x2, y2), (0, 255, 0), 2)

    # Show result
    cv2.imshow("Long Lines", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


image_path = sys.argv[1]
# Run the function
find_long_lines(image_path, min_length=200)
