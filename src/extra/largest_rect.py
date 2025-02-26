import cv2
import numpy as np
import sys

def find_largest_rectangle(image_path):
    # Load image
    img = cv2.imread(image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Apply edge detection
    edges = cv2.Canny(gray, 50, 150)

    # Find contours
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    largest_rectangle = None
    max_area = 0

    for contour in contours:
        # Approximate the contour to a polygon
        epsilon = 0.02 * cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, epsilon, True)

        # Check if the polygon has 4 sides (rectangle) and is convex
        if len(approx) == 4 and cv2.isContourConvex(approx):
            area = cv2.contourArea(approx)
            if area > max_area:
                max_area = area
                largest_rectangle = approx

    # Draw the largest rectangle
    if largest_rectangle is not None:
        cv2.drawContours(img, [largest_rectangle], -1, (0, 255, 0), 3)

    # Show result
    cv2.imshow("Largest Rectangle", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# Run the function
image_path = sys.argv[1]
find_largest_rectangle(image_path)
