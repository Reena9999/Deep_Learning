#!/usr/bin/env python
# coding: utf-8

# In[46]:


import PIL


# In[47]:


from PIL import Image
import openpyxl

# Load the image
image_path = r"C:\Users\HP\OneDrive\Documents\Project\Images\1.png"
image = Image.open(image_path)

image = image.convert("L")


# In[48]:


image


# In[50]:


print(image.width, image.height)


# In[51]:


new_image = []
for h in range(image.height):
    new_row = []
    for w in range(image.width):
        pixel_value = image.getpixel((w, h))
        
        print(pixel_value, end=' ')
        new_row.append(pixel_value)
    new_image.append(new_row)
    print("")


# In[24]:


new_image


# In[35]:


def get_threshold_pixel_value(pixel_value, threshold):
    if pixel_value < threshold:
        return 0
    return 255


# In[26]:


new_image = []
for h in range(image.height):
    new_row = []
    for w in range(image.width):
        pixel_value = image.getpixel((w, h))
        
        print(pixel_value, end=' ')
        pixel_value = threshold(pixel_value, 128)
        new_row.append(pixel_value)
        
    new_image.append(new_row)
    print("\n")


# In[27]:


new_image


# In[28]:


import matplotlib.pyplot as plt


# In[29]:


plt.imshow(new_image, cmap='gray')  # 'gray' colormap for grayscale images
plt.axis('off')  # Hide the axis labels and ticks
plt.show()


# In[38]:


def create_threshold_image(image, threshold):
    new_image = []
    for h in range(image.height):
        new_row = []
        for w in range(image.width):
            pixel_value = image.getpixel((w, h))

            #print(pixel_value, end=' ')
            pixel_value = get_threshold_pixel_value(pixel_value, threshold)
            new_row.append(pixel_value)

        new_image.append(new_row)
        #print("\n")

    plt.imshow(new_image, cmap='gray')  # 'gray' colormap for grayscale images
    plt.axis('off')  # Hide the axis labels and ticks
    plt.show()


# In[39]:


create_threshold_image(image, 128)


# In[40]:


for t in range(0, 255, 16):
    create_threshold_image(image, t)


# In[41]:
t


# In[44]:


import cv2
import numpy as np


# In[45]:


# Load the image
image_path = r"C:\Users\HP\OneDrive\Documents\TransformImage\1.png"
image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

# Apply the Sobel operator to get gradients in the horizontal and vertical directions
sobel_x = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=3)
sobel_y = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=3)

# Compute the gradient magnitude
gradient_magnitude = np.sqrt(sobel_x**2 + sobel_y**2)

# Display the original image and the gradient magnitude (optional)
cv2.imshow("Original Image", image)
cv2.imshow("Gradient Magnitude", gradient_magnitude.astype(np.uint8))
cv2.waitKey(0)
cv2.destroyAllWindows()


# In[ ]:




