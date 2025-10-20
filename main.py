from keras.models import load_model
from PIL import Image, ImageOps
import numpy as np
import cv2
import time

model = load_model("keras_model.h5")

# Labels
class_names = [line.strip() for line in open("labels.txt", "r").readlines()]

# Load the image to display when "six seven" is predicted
sixseven_image = cv2.imread("sixseven.png")

# Open webcam
cap = cv2.VideoCapture(0)

# Frame size for the model
frame_size = (224, 224)

# Timer to control pause
last_trigger_time = 0
pause_seconds = 1

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Convert OpenCV BGR to RGB
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Convert to PIL Image
    image = Image.fromarray(frame_rgb)

    # Resize and crop
    image = ImageOps.fit(image, frame_size, Image.Resampling.LANCZOS)

    # Convert to numpy array
    image_array = np.asarray(image).astype(np.float32)

    # Normalize
    normalized_image_array = (image_array / 127.5) - 1

    # Add batch dimension
    data = np.expand_dims(normalized_image_array, axis=0)

    # Predict
    prediction = model.predict(data)
    index = np.argmax(prediction)
    class_name = class_names[index]

    current_time = time.time()
    if class_name == "six seven" and current_time - last_trigger_time > pause_seconds:
        # Show the sixseven.png image
        cv2.imshow("Six Sevennn", sixseven_image)
        cv2.waitKey(1000)  # Wait 1 second
        last_trigger_time = current_time

    # Optionally show webcam feed
    cv2.imshow("Webcam", frame)

    # Press 'q' to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
