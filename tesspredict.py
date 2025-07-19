import cv2
import matplotlib.pyplot as plt

img = cv2.imread("digit.png", cv2.IMREAD_GRAYSCALE)

print("Shape:", img.shape)
print("Pixel Range:", img.min(), "to", img.max())

plt.imshow(img, cmap='gray')
plt.title("digit.png as read by OpenCV")
plt.axis('off')
plt.show()
