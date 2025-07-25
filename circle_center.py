import cv2
import numpy as np


image_path = '/Users/buraktaskin/Desktop/776307057-JPEG_20230802_104909_8381604715226828084_processed_visualization.jpg'
image = cv2.imread(image_path)
# Convert the image to grayscale for processing
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Apply a median blur to reduce noise and improve circle detection
gray_blurred = cv2.medianBlur(gray, 5)

# Detect circles in the image using the Hough Circle Transform
circles = cv2.HoughCircles(
    gray_blurred, 
    cv2.HOUGH_GRADIENT, 
    dp=1, 
    minDist=100,
    param1=100,
    param2=30,
    minRadius=50,
    maxRadius=200
)


if circles is not None:
    # Round circle parameters to integers and process each detected circle
    circles = np.uint16(np.around(circles))
    for circle in circles[0, :]:
        x, y, r = circle
        width = 2 * r
        height = 2 * r
        print(f"Center Coordinates: x = {x}, y = {y}")
        print(f"Diameter = {2 * r}, Radius = {r}")
        print(f"Width = {width}, Height = {height}")

       
        output = image.copy()
        # Draw the circle's outline
        cv2.circle(output, (x, y), r, (0, 255, 0), 2)        
        # Mark the center of the circle
        cv2.circle(output, (x, y), 2, (0, 0, 255), 3)        
        # Annotate the image with the circle's center coordinates
        cv2.putText(output, f"({x},{y})", (x + 10, y - 10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

        
        cv2.imwrite("output_center_marked.jpg", output)
        break  # Process only the first detected circle

else:
    print("Error: No circles were found.")