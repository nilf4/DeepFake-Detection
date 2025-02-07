import cv2
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import random
import pickle

# Load the trained model from the .pkl file
model_path = r"D:\Local Siddharth\fake_image_classifier.pkl"
with open(model_path, 'rb') as file:
    model = pickle.load(file)

# Updated paths for testing fake and real images
fake_image_dir = r"D:\Local Siddharth\fake"
real_image_dir = r"D:\Local Siddharth\real"

# Image size to resize images
image_size = 224
labels = ['Real', 'Fake']


# Function to load and preprocess an image
def load_and_preprocess_image(image_path):
    img = cv2.imread(image_path)[..., ::-1]  # BGR to RGB
    resized_img = cv2.resize(img, (image_size, image_size))
    normalized_img = resized_img / 255.0  # Normalize the image
    return normalized_img, img


# Function to compute color histograms
def plot_color_histogram(image, ax):
    colors = ('r', 'g', 'b')
    for i, color in enumerate(colors):
        hist = cv2.calcHist([image], [i], None, [256], [0, 256])
        ax.plot(hist, color=color)
    ax.set_title("Color Histogram")
    ax.set_xlim([0, 256])


# Function to display average color
def plot_average_color(image, ax):
    avg_color = image.mean(axis=0).mean(axis=0)
    avg_color_img = np.ones((100, 100, 3), dtype='uint8') * avg_color.astype('uint8')
    ax.imshow(avg_color_img)
    ax.set_title("Average Color")
    ax.axis('off')


# Function to apply Canny edge detection
def plot_edges(image, ax):
    edges = cv2.Canny(cv2.cvtColor(image, cv2.COLOR_RGB2GRAY), 100, 200)
    ax.imshow(edges, cmap='gray')
    ax.set_title("Canny Edge Detection")
    ax.axis('off')


# Function to display grayscale image
def plot_grayscale(image, ax):
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    ax.imshow(gray, cmap='gray')
    ax.set_title("Grayscale Image")
    ax.axis('off')


# Function to make predictions and display results
def predict_and_visualize(fake_dir, real_dir):
    # Randomly select one image from each directory
    fake_img_path = os.path.join(fake_dir, random.choice(os.listdir(fake_dir)))
    real_img_path = os.path.join(real_dir, random.choice(os.listdir(real_dir)))

    # Load and preprocess the images
    fake_img, fake_original = load_and_preprocess_image(fake_img_path)
    real_img, real_original = load_and_preprocess_image(real_img_path)

    # Prepare images for prediction
    fake_img_input = np.expand_dims(fake_img, axis=0)
    real_img_input = np.expand_dims(real_img, axis=0)

    # Predict the probability of being fake (1) or real (0)
    fake_prediction = model.predict(fake_img_input)[0][0]
    real_prediction = model.predict(real_img_input)[0][0]

    # Convert probabilities to percentages
    fake_real_prob = (1 - fake_prediction) * 100
    fake_fake_prob = fake_prediction * 100
    real_real_prob = (1 - real_prediction) * 100
    real_fake_prob = real_prediction * 100

    # Display results for the fake image
    print(f"Prediction for Fake Image: {fake_img_path}")
    print(f"Real: {fake_real_prob:.2f}%, Fake: {fake_fake_prob:.2f}%")
    display_results(fake_original, fake_real_prob, fake_fake_prob, "Fake Image")

    # Display results for the real image
    print(f"\nPrediction for Real Image: {real_img_path}")
    print(f"Real: {real_real_prob:.2f}%, Fake: {real_fake_prob:.2f}%")
    display_results(real_original, real_real_prob, real_fake_prob, "Real Image")


# Function to display results with multiple graphs
def display_results(image, real_prob, fake_prob, title):
    plt.figure(figsize=(16, 10))

    # Show the original image
    plt.subplot(2, 3, 1)
    plt.imshow(image)
    plt.title(f"{title} - Original Image")
    plt.axis('off')

    # Display color histogram
    plt.subplot(2, 3, 2)
    plot_color_histogram(image, plt.gca())

    # Display average color
    plt.subplot(2, 3, 3)
    plot_average_color(image, plt.gca())

    # Display grayscale image
    plt.subplot(2, 3, 4)
    plot_grayscale(image, plt.gca())

    # Display Canny edges
    plt.subplot(2, 3, 5)
    plot_edges(image, plt.gca())

    # Display prediction bar chart
    plt.subplot(2, 3, 6)
    probabilities = [real_prob, fake_prob]
    sns.barplot(x=labels, y=probabilities, palette='coolwarm')
    plt.title(f"Prediction: {labels[np.argmax(probabilities)]}")
    plt.ylabel('Probability (%)')
    plt.ylim(0, 100)

    # Show the plots
    plt.tight_layout()
    plt.show()


# Run the prediction and visualization
predict_and_visualize(fake_image_dir, real_image_dir)
