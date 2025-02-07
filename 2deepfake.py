import cv2
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
from keras.models import Sequential
from keras.layers import Conv2D, MaxPool2D, Flatten, Dense, Dropout
import tensorflow as tf
import pickle

# Updated paths for training fake and real images
fake_image_dir = r"D:\Local Siddharth\fake"
real_image_dir = r"D:\Local Siddharth\real"

# Image size to resize images
image_size = 224
labels = ('real', 'fake')


# Function to get images and labels from the directories
def get_data(fake_dir, real_dir):
    data = []

    # Access fake images
    for img in os.listdir(fake_dir):
        try:
            img_arr = cv2.imread(os.path.join(fake_dir, img))[..., ::-1]  # BGR to RGB
            resized_arr = cv2.resize(img_arr, (image_size, image_size))
            data.append([resized_arr, 1])  # Label 1 for fake
        except Exception as e:
            print(f"Error loading image {img}: {e}")

    # Access real images
    for img in os.listdir(real_dir):
        try:
            img_arr = cv2.imread(os.path.join(real_dir, img))[..., ::-1]  # BGR to RGB
            resized_arr = cv2.resize(img_arr, (image_size, image_size))
            data.append([resized_arr, 0])  # Label 0 for real
        except Exception as e:
            print(f"Error loading image {img}: {e}")

    return np.array(data, dtype='object')


# Load training data
train = get_data(fake_image_dir, real_image_dir)

# Check if data is loaded correctly
print(f"Training data shape: {train.shape}")

# Prepare labels for visualization
l = ['real' if i[1] == 0 else 'fake' for i in train]

# Plotting the distribution of real vs fake images
plt.figure(figsize=(8, 4))
plt.title('Count of Real vs Fake Images', size=16)
sns.countplot(x=l)
plt.show()

# Visualizing random images from the training data
plt.figure(figsize=(8, 6))
plt.imshow(train[1][0])
plt.title(labels[train[1][1]])
plt.show()

plt.figure(figsize=(10, 6))
plt.imshow(train[-1][0])
plt.title(labels[train[-1][1]])
plt.show()

# Splitting the data into features and labels
X_train = []
y_train = []

for feature, label in train:
    X_train.append(feature)
    y_train.append(label)

# Convert to numpy arrays
X_train = np.array(X_train)
y_train = np.array(y_train)

# Normalize the images
X_train = X_train / 255.0

# Check the shapes of the datasets
print(f"X_train shape: {X_train.shape}")
print(f"y_train shape: {y_train.shape}")

# Build a CNN model
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(image_size, image_size, 3)),
    MaxPool2D(pool_size=(2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPool2D(pool_size=(2, 2)),
    Conv2D(128, (3, 3), activation='relu'),
    MaxPool2D(pool_size=(2, 2)),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(1, activation='sigmoid')
])

# Compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Summary of the model
model.summary()

# Train the model
history = model.fit(X_train, y_train, epochs=10, batch_size=32)

# Plot training and validation loss
plt.plot(history.history['loss'], label='Training Loss')
plt.title('Loss during training')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

# Plot training and validation accuracy
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.title('Accuracy during training')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()

# Predict on the training set (you can also split into train/test later)
y_pred = model.predict(X_train)
y_pred = (y_pred > 0.5).astype(int)

# Compute confusion matrix
cm = confusion_matrix(y_train, y_pred)

# Plot the confusion matrix
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap=plt.cm.Blues, xticklabels=labels, yticklabels=labels)
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('True')
plt.show()

# Calculate accuracy
accuracy = np.sum(y_pred.flatten() == y_train) / len(y_train)
print(f"Train Accuracy: {accuracy * 100:.2f}%")

# Save the trained model as a .pkl file
model_path = r"D:\Local Siddharth\fake_image_classifier.pkl"
with open(model_path, 'wb') as file:
    pickle.dump(model, file)

print(f"Model saved at {model_path}")
