import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import time
import sys

# Read image
image = cv2.imread('Quiz1_Grayscale/images/japan.jpg')

# OpenCV grayscale
gray_cv = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Read image as RGB
img = Image.open('Quiz1_Grayscale/images/japan.jpg')
img_array = np.array(img)

height, width, _ = img_array.shape

# NumPy grayscale
def rgb_to_grayscale(img_array):
    gray = (img_array.sum(axis=2) // 3).astype(np.uint8)
    return gray

gray_np = rgb_to_grayscale(img_array)


# Compare execution time
N = 30

start = time.perf_counter()
for _ in range(N):
    cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
cv_time = (time.perf_counter() - start) / N

start = time.perf_counter()
for _ in range(N):
    rgb_to_grayscale(img_array)
np_time = (time.perf_counter() - start) / N

# Compare results
diff = np.abs(gray_cv.astype(int) - gray_np.astype(int))

with open('Quiz1_Grayscale/results/results.txt', 'w') as file:
    sys.stdout = file 
    print(f"Number of repetitions: {N}")
    print(f"OpenCV average time: {cv_time:.6f} s")
    print(f"NumPy average time:  {np_time:.6f} s")
    print(f"Maximum difference:  {np.max(diff)}")
    print(f"Mean difference:     {np.mean(diff):.6f}")
sys.stdout = sys.__stdout__ 

# Display results
plt.figure(figsize=(12, 4))

plt.subplot(1, 3, 1)
plt.imshow(img)
plt.title("Original")

plt.subplot(1, 3, 2)
plt.imshow(gray_cv, cmap='gray')
plt.title("OpenCV")

plt.subplot(1, 3, 3)
plt.imshow(gray_np, cmap='gray')
plt.title("NumPy")

cv2.imwrite('Quiz1_Grayscale/output/gray_opencv.jpg', gray_cv)
cv2.imwrite('Quiz1_Grayscale/output/gray_numpy.jpg', gray_np)
plt.show()