import cv2
import matplotlib.pyplot as plt
import numpy as np

# Load the image using OpenCV
img = cv2.imread(r'D:\Local Siddharth\fake\easy_50_0110.jpg')

# Check if the image was loaded successfully
if img is None:
    print("Error: Image not found. Please check the file path.")
else:
    # Convert the image from BGR (OpenCV default) to RGB
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Get image properties
    height, width, channels = img.shape
    image_size = img.size
    mean_intensity = np.mean(img)

    print(f"Image Dimensions: {width}x{height}")
    print(f"Number of Channels: {channels}")
    print(f"Image Size (in pixels): {image_size}")
    print(f"Mean Pixel Intensity: {mean_intensity:.2f}")

    # Display the image using matplotlib
    plt.figure(figsize=(10, 5))

    # Subplot 1: Original Image
    plt.subplot(2, 2, 1)
    plt.imshow(img_rgb)
    plt.title("Original Image")
    plt.axis('off')

    # Subplot 2: Image Histogram (Color Channels)
    plt.subplot(2, 2, 2)
    colors = ('r', 'g', 'b')
    for i, color in enumerate(colors):
        hist = cv2.calcHist([img_rgb], [i], None, [256], [0, 256])
        plt.plot(hist, color=color)
    plt.title("Color Histogram")
    plt.xlabel("Pixel Intensity")
    plt.ylabel("Frequency")

    # Subplot 3: Grayscale Version
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    plt.subplot(2, 2, 3)
    plt.imshow(gray_img, cmap='gray')
    plt.title("Grayscale Image")
    plt.axis('off')

    # Subplot 4: Contrast Analysis (using Histogram Equalization)
    equalized_img = cv2.equalizeHist(gray_img)
    plt.subplot(2, 2, 4)
    plt.imshow(equalized_img, cmap='gray')
    plt.title("Contrast Enhanced")
    plt.axis('off')

    # Show all the plots
    plt.tight_layout()
    plt.show()
