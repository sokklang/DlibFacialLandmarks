import cv2
import dlib
import imutils

# Load the face detector and facial landmarks predictor
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

# Load the image and specify the desired width or height
input_image_path = 'image.jpg'
desired_width = 600  # Adjust as needed
desired_height = 600  # Adjust as needed

# Load the image and resize it while maintaining aspect ratio
image = cv2.imread(input_image_path)
image = imutils.resize(image, width=desired_width, height=desired_height)

# Convert the image to grayscale for face detection
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Detect faces in the grayscale image
faces = detector(gray)

# Define the ranges to disconnect lines
disconnect_ranges = [(0, 16), (17, 21), (22, 26), (36, 41), (42, 47)]

# Loop through each detected face
for face in faces:
    # Get the facial landmarks for the face
    landmarks = predictor(gray, face)

    # Loop through each landmark point and draw a red circle
    for n in range(0, 68):
        x = landmarks.part(n).x
        y = landmarks.part(n).y
        cv2.circle(image, (x, y), 3, (0, 0, 255), -1)  # Draw a red circle on each landmark point

    # Draw lines connecting the facial landmarks, except for specified ranges
    for (i, j) in disconnect_ranges:
        for k in range(i, j):
            x1, y1 = landmarks.part(k).x, landmarks.part(k).y
            x2, y2 = landmarks.part(k + 1).x, landmarks.part(k + 1).y
            cv2.line(image, (x1, y1), (x2, y2), (0, 255, 0), 1)  # Draw lines connecting landmark points

    # Draw lines around the left eye (indices 36 to 41)
    for i, j in [(36, 37), (37, 38), (38, 39), (39, 40), (40, 41), (41, 36)]:
        x1, y1 = landmarks.part(i).x, landmarks.part(i).y
        x2, y2 = landmarks.part(j).x, landmarks.part(j).y
        cv2.line(image, (x1, y1), (x2, y2), (0, 255, 0), 1)  # Draw lines around the left eye

    # Draw lines around the right eye (indices 42 to 47)
    for i, j in [(42, 43), (43, 44), (44, 45), (45, 46), (46, 47), (47, 42)]:
        x1, y1 = landmarks.part(i).x, landmarks.part(i).y
        x2, y2 = landmarks.part(j).x, landmarks.part(j).y
        cv2.line(image, (x1, y1), (x2, y2), (0, 255, 0), 1)  # Draw lines around the right eye

    # Draw lines around the nose (indices 27 to 35)
    for i, j in [(27,28),(28,29),(29,30),(30,31),(31,32),(32,33),(33,34),(34,35),(30,35)]:
        x1, y1 = landmarks.part(i).x, landmarks.part(i).y
        x2, y2 = landmarks.part(j).x, landmarks.part(j).y
        cv2.line(image, (x1, y1), (x2, y2), (0, 255, 0), 1) 

    # These are landmarks for the outer mouth contour.
    for i,j in [(48, 49), (49, 50), (50, 51), (51, 52), (52, 53), (53, 54), (54, 55), (55, 56), (56, 57), (57, 58), (58, 59), (59, 48)]:
        x1, y1 = landmarks.part(i).x, landmarks.part(i).y
        x2, y2 = landmarks.part(j).x, landmarks.part(j).y
        cv2.line(image, (x1, y1), (x2, y2), (0, 255, 0), 1)

    # These are landmarks for the inner mouth contour, including the lips.
    for i,j in [(60, 61), (61, 62), (62, 63), (63, 64), (64, 65), (65, 66), (66, 67),(60,67)]:
        x1, y1 = landmarks.part(i).x, landmarks.part(i).y
        x2, y2 = landmarks.part(j).x, landmarks.part(j).y
        cv2.line(image, (x1, y1), (x2, y2), (0, 255, 0), 1)         

# Display the resized image with facial landmarks and eye outlines
cv2.imshow('Resized Image with Facial Landmarks and Eye Outlines', image)
cv2.waitKey(0)
cv2.destroyAllWindows()
