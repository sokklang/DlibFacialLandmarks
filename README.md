# Facial Landmarks With Dlib

This Python code uses the dlib library to detect facial landmarks and draw outlines on a given image. It provides a visualization of facial landmarks, including those around the eyes, nose, and mouth.

## Requirements

Make sure you have the following libraries installed:

- OpenCV (`cv2`)
- Dlib
- Imutils
- shape_predictor_68_face_landmarks.dat [link here](https://github.com/davisking/dlib-models/blob/master/shape_predictor_68_face_landmarks.dat.bz2).

You can install these libraries using pip:

```bash
pip install opencv-python
pip install dlib
pip install imutils
```

## USAGE

1. Clone or download this repository to your local machine.

2. Place your image file (e.g., image.jpg) in the same directory as the script.

3. Replace input_image_path in the code with the filename of your image.

4. Adjust the desired width and height if needed:

    ```python
    desired_width = 600  # Adjust as needed
    desired_height = 600  # Adjust as needed
    ```

5. Run the script:

    ```bash
    python main.py
    ```

## Customization

1. You can modify the `disconnect_ranges` variable to specify which landmark point ranges to exclude from line connections.

2. The code includes predefined ranges for the jawline, eyebrows, and other face regions. You can customize these ranges or add new ones to suit your needs.

3. You can change the line colors and thickness by modifying the `cv2.circle` and `cv2.line` function arguments in the code.

## Result

![Local image](image\output_image.jpg)
