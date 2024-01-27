# Path: Finding-lanes/Video.py
import cv2
import numpy as np

def canny(image):
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    # Reduce noise
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    # Detect edges
    canny = cv2.Canny(blur, 50, 150)
    return canny

def region_of_interest(image):
    height = image.shape[0]
    polygons = np.array([
        [(200, height), (1100, height), (550, 250)]
    ])
    # Create a mask
    mask = np.zeros_like(image)
    # Fill the mask
    cv2.fillPoly(mask, polygons, 255)
    # Apply mask
    masked_image = cv2.bitwise_and(image, mask)
    return masked_image

def display_lines(image, lines):
    # Create an array of zeros with the same shape as the image
    line_image = np.zeros_like(image)
    # Check if there are any lines
    if lines is not None:
        # Iterate over every line
        for line in lines:
            # Reshape the line to a 1D array
            x1, y1, x2, y2 = line.reshape(4)
            # Draw the line
            cv2.line(line_image, (x1, y1), (x2, y2), (255, 0, 0), 10)
    return line_image

def make_coordinates(image, line_parameters):
    # Get the slope and intercept of the line
    slope, intercept = line_parameters
    # Get the height of the image
    y1 = image.shape[0]
    # Get the height of the region of interest
    y2 = int(y1 * (3 / 5))
    # Get the x coordinates
    x1 = int((y1 - intercept) / slope)
    x2 = int((y2 - intercept) / slope)
    # Return the coordinates
    return np.array([x1, y1, x2, y2])

def average_slope_intercept(image, lines):
    # Create an array to store the left and right lines
    left_fit = []
    right_fit = []
    # Check if there are any lines
    if lines is not None:
        # Iterate over every line
        for line in lines:
            # Reshape the line to a 1D array
            x1, y1, x2, y2 = line.reshape(4)
            # Get the slope and intercept of the line
            parameters = np.polyfit((x1, x2), (y1, y2), 1)
            # Get the slope
            slope = parameters[0]
            # Get the intercept
            intercept = parameters[1]
            # Check if the slope is negative
            if slope < 0:
                # Add the slope and intercept to the left array
                left_fit.append((slope, intercept))
            else:
                # Add the slope and intercept to the right array
                right_fit.append((slope, intercept))
        # Get the average of the left and right lines
        left_fit_average = np.average(left_fit, axis=0)
        right_fit_average = np.average(right_fit, axis=0)
        # Get the coordinates of the left and right lines
        left_line = make_coordinates(image, left_fit_average)
        right_line = make_coordinates(image, right_fit_average)
        # Return the coordinates
        return np.array([left_line, right_line])
    
# Read the video
cap = cv2.VideoCapture('test2.mp4')
# Iterate over every frame
while(cap.isOpened()):
    # Read the frame
    _, frame = cap.read()
    # Detect edges
    canny_image = canny(frame)
    # Get the region of interest
    cropped_image = region_of_interest(canny_image)
    # Detect lines
    lines = cv2.HoughLinesP(cropped_image, 2, np.pi / 180, 100, np.array([]), minLineLength=40, maxLineGap=5)
    # Get the average of the left and right lines
    averaged_lines = average_slope_intercept(frame, lines)
    # Display the lines
    line_image = display_lines(frame, averaged_lines)
    # Combine the images
    combo_image = cv2.addWeighted(frame, 0.8, line_image, 1, 1)
    # Display the image
    cv2.imshow('result', combo_image)
    # Wait for a key press to exit
    if cv2.waitKey(1) == ord('q'):
        break
# Release the video
cap.release()
# Destroy all windows
cv2.destroyAllWindows()