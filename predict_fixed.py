import cv2
import numpy as np
from tensorflow.keras.models import load_model

# Load the model
model = load_model("model.h5")

# Load the digit image
img = cv2.imread("digit.png", cv2.IMREAD_GRAYSCALE)

# Check if image was loaded
if img is None:
    print("❌ Error: digit.png not found.")
    exit()

# Resize to 28x28 as expected by MNIST model
img = cv2.resize(img, (28, 28))

# Invert colors if background is dark
# Check the average pixel value to decide if inversion is needed
if np.mean(img) > 127:
    img = 255 - img  # Invert image (MNIST digits are white on black)

# Normalize and reshape
img = img / 255.0
img = img.reshape(1, 28, 28, 1)

# Predict
prediction = model.predict(img)
predicted_digit = np.argmax(prediction)

print("✅ Predicted Digit:", predicted_digit)
