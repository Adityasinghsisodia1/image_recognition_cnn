import numpy as np
from tensorflow.keras.models import load_model
from PIL import Image

# Load model
model = load_model("model.h5")

# Load and preprocess image
img = Image.open("test_image.png").convert('L')  # Convert to grayscale
img = img.resize((28, 28))
img_array = np.array(img)
img_array = 255 - img_array  # Invert colors if needed
img_array = img_array / 255.0
img_array = img_array.reshape(1, 28, 28, 1)

# Predict
prediction = model.predict(img_array)
predicted_digit = np.argmax(prediction)
print(f"🧠 Predicted Digit: {predicted_digit}")
