import cv2
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np 

# %% Reading Image
img = Image.open('motherboard_image.JPEG')
img = img.convert('RGB')
img_np = np.array(img) 

plt.imshow(img)
plt.title('Original Image')
plt.xticks([]), plt.yticks([])
plt.show()

# %% Adding Threshold
img_gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)

ret, thresh = cv2.threshold(img_gray, 120, 255, cv2.THRESH_BINARY_INV)

plt.imshow(thresh, cmap='gray')
plt.title('Threshold Image')
plt.xticks([]), plt.yticks([])
plt.show()

# %% Edge Detection
contours, hierarchy = cv2.findContours(thresh,  cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

largest_contour = max(contours, key=cv2.contourArea)

img_cont = img_np.copy()

cv2.drawContours(img_cont, [largest_contour], -1, (0, 255, 0), 3)

plt.imshow(img_cont, cmap='gray')
plt.title('Contoured Edges')
plt.xticks([]), plt.yticks([])
plt.show()

# %% Mask Creation
mask = np.zeros_like(img_gray)

cv2.drawContours(mask, [largest_contour], -1, 255, thickness=cv2.FILLED)

plt.imshow(mask, cmap='gray')
plt.title('Mask from Contoured Edges')
plt.xticks([]), plt.yticks([])
plt.show()

# %% PCB Extraction
extracted_pcb = cv2.bitwise_and(img_np, img_np, mask=mask)

background = np.zeros_like(img_np)
background[mask == 255] = extracted_pcb[mask == 255]

plt.imshow(background)
plt.title('Extracted PCB')
plt.xticks([]), plt.yticks([])
plt.show()