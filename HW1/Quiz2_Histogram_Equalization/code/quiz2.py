import cv2
import numpy as np
import matplotlib.pyplot as plt
import time
import sys

img = cv2.imread('Quiz2_Histogram_Equalization/images/japan.jpg', 0)  

# NumPy Histogram Equalization
def numpy_equalizeHist(img):
    # caculate Histogram 計算每個灰階值出現幾次
    hist = np.bincount(img.ravel(), minlength=256)

    # Cumulative Distribution Function 累積出現次數
    cdf = hist.cumsum()

    # Ignore zero values at the beginning of CDF
    cdf_min = cdf[cdf > 0][0]

    # Histogram Equalization formula
    lut = np.round((cdf - cdf_min) / (img.size - cdf_min) * 255)
    # 確保所有結果都在0 ~ 255
    lut = np.clip(lut, 0, 255).astype(np.uint8)

    return lut[img]

# OpenCV
equ_cv = cv2.equalizeHist(img)

# NumPy
equ_np = numpy_equalizeHist(img)

# Compare execution time
N = 30

start = time.perf_counter()
for _ in range(N):
    cv2.equalizeHist(img)
cv_time = (time.perf_counter() - start) / N

start = time.perf_counter()
for _ in range(N):
    numpy_equalizeHist(img)
np_time = (time.perf_counter() - start) / N


# Compare results
diff = np.abs(equ_cv.astype(int) - equ_np.astype(int))

with open('Quiz2_Histogram_Equalization/results/results.txt', 'w') as file:
    sys.stdout = file 
    print("Number of repetitions:", N)
    print(f"OpenCV average time: {cv_time:.6f} s")
    print(f"NumPy average time:  {np_time:.6f} s")
    print(f"Maximum difference: {np.max(diff)}")
    print(f"Mean difference: {np.mean(diff):.6f}")
sys.stdout = sys.__stdout__ 

# Display images and histograms
plt.figure(figsize=(12, 8))

plt.subplot(2, 3, 1)
plt.imshow(img, cmap='gray')
plt.title("Original")
plt.axis('off')

plt.subplot(2, 3, 2)
plt.imshow(equ_cv, cmap='gray')
plt.title("OpenCV Equalized")
plt.axis('off')

plt.subplot(2, 3, 3)
plt.imshow(equ_np, cmap='gray')
plt.title("NumPy Equalized")
plt.axis('off')

plt.subplot(2, 3, 4)
plt.hist(img.ravel(), bins=256, range=(0, 256))
plt.title("Original Histogram")

plt.subplot(2, 3, 5)
plt.hist(equ_cv.ravel(), bins=256, range=(0, 256))
plt.title("OpenCV Histogram")

plt.subplot(2, 3, 6)
plt.hist(equ_np.ravel(), bins=256, range=(0, 256))
plt.title("NumPy Histogram")

plt.tight_layout()
cv2.imwrite('Quiz2_Histogram_Equalization/output/OpenCV_Histogram.jpg', equ_cv)
cv2.imwrite('Quiz2_Histogram_Equalization/output/NumPy_Histogram.jpg', equ_np)
plt.show()