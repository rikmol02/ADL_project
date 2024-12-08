import numpy as np 
import pandas as pd 
import tensorflow as tf
import os
import random

# Image related
from PIL import Image

# Performance Plot
from sklearn import metrics 
from sklearn.metrics import f1_score, accuracy_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns

# For the model and its training
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Conv2D, MaxPool2D, Dense, Flatten, Dropout

# Time
import time
import datetime
from tensorflow.keras.applications import ResNet50  # Using ResNet-50 (TensorFlow doesn't directly provide ResNet-18)
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from imblearn.under_sampling import RandomUnderSampler
from collections import Counter
import torch

from skimage.util import random_noise
from imblearn.over_sampling import RandomOverSampler
from collections import Counter

def add_salt_pepper_noise(images, intensity):
    """
    Apply salt-and-pepper noise to a batch of images.
    :param images: Input images as a numpy array (N, H, W, C).
    :param intensity: Noise intensity (fraction of pixels to be noised).
    :return: Noisy images as a numpy array with the same shape as input.
    """
    if images.size == 0:
        raise ValueError("Empty input array for adding salt-and-pepper noise.")
    
    noisy_images = []
    for i, image in enumerate(images):
        noisy_image = random_noise(image, mode='s&p', amount=intensity)
        noisy_image = (noisy_image * 255).astype(np.uint8)  # Convert back to [0, 255]
        noisy_images.append(noisy_image)

        # Optional: Debugging visualization
        if i == 0:
            plt.imshow(noisy_image)
            plt.title(f"Sample Noisy Image (Intensity {intensity})")
            plt.axis('off')
            plt.show()
    
    return np.array(noisy_images)


def adversarial_testing(model, X_test, y_test, target_class, noise_intensities):
    """
    Perform adversarial testing on samples of a target class with salt-and-pepper noise.
    :param model: Trained CNN model.
    :param X_test: Test dataset (numpy array).
    :param y_test: Test labels (numpy array).
    :param target_class: Class to test for adversarial robustness.
    :param noise_intensities: List of noise intensity levels.
    :return: Dictionary mapping noise intensity to average class probabilities.
    """
    # Filter samples of the target class
    target_indices = np.where(y_test == target_class)[0]
    target_samples = X_test[target_indices]

    if target_samples.size == 0:
        print(f"No samples found for target class {target_class}. Skipping adversarial testing.")
        return {}

    # Initialize a dictionary to store average probabilities
    avg_probabilities = {}

    for intensity in noise_intensities:
        print(f"Testing with {intensity * 100}% salt-and-pepper noise...")
        
        try:
            noisy_samples = add_salt_pepper_noise(target_samples, intensity)
        except ValueError as e:
            print(f"Skipping noise intensity {intensity * 100}% due to error: {e}")
            continue

        # Debugging: Check the shape of noisy_samples
        print(f"Shape of noisy_samples for {intensity * 100}% noise: {noisy_samples.shape}")

        if len(noisy_samples) == 0:
            print(f"No noisy samples generated for {intensity * 100}% noise.")
            continue

        # Normalize noisy samples to [0, 1]
        noisy_samples = noisy_samples / 255.0

        # Perform inference
        predictions = model.predict(noisy_samples, verbose=0)
        probabilities = tf.nn.softmax(predictions, axis=1).numpy()
        print(f"Predictions (logits): {predictions}")
        print(f"Probabilities: {probabilities}")

        # Compute average probabilities
        avg_probabilities[intensity] = probabilities.mean(axis=0)
    
    return avg_probabilities




def plot_history(history):
    """
    Plot training and validation accuracy and loss.

    Parameters:
        history (keras.callbacks.History): Training history returned by model.fit().
    """
    # Extract values from the history object
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    
    epochs = range(1, len(acc) + 1)

    # Plot accuracy
    plt.figure(figsize=(14, 6))

    plt.subplot(1, 2, 1)
    plt.plot(epochs, acc, label='Training Accuracy', marker='o')
    plt.plot(epochs, val_acc, label='Validation Accuracy', marker='o')
    plt.title('Training and Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True)

    # Plot loss
    plt.subplot(1, 2, 2)
    plt.plot(epochs, loss, label='Training Loss', marker='o', color='red')
    plt.plot(epochs, val_loss, label='Validation Loss', marker='o', color='orange')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.show()


def apply_oversampling(X_train, y_train, target_classes):
    """
    Oversample the specified classes in the training set.
    """
    # Flatten X_train for oversampling
    X_train_flat = X_train.reshape((X_train.shape[0], -1))
    
    # Ensure there are multiple classes in the data
    unique_classes = np.unique(y_train)
    if len(unique_classes) <= 1:
        raise ValueError(f"The training set must contain more than one class for oversampling. Found: {unique_classes}")
    
    # Perform oversampling
    oversampler = RandomOverSampler(random_state=42)
    X_resampled, y_resampled = oversampler.fit_resample(X_train_flat, y_train)
    
    # Print initial and new class distribution
    print("Before Oversampling:", Counter(y_train))
    print("After Oversampling:", Counter(y_resampled))
    
    # Reshape X_resampled back to the original shape
    X_resampled = X_resampled.reshape((-1, X_train.shape[1], X_train.shape[2], X_train.shape[3]))
    
    return X_resampled, y_resampled

def main():
    # Setting variables for later use
    data = []
    labels = []
    signs = [2, 14, 33, 34]
    classes = len(signs)
    
    # Create a mapping of the original class labels to the range 0 to len(signs)-1
    class_mapping = {sign: idx for idx, sign in enumerate(signs)}
    
    # Retrieving the images and their labels 
    for sign in signs:
        path = os.path.join('./gtsrb-german-traffic-sign/', 'Train', str(sign))
        images = os.listdir(path)
    
        for image_file in images:
            try:
                image = Image.open(path + '/' + image_file)
                image = image.resize((128, 128))
                image = np.array(image)
                
                data.append(image)
                # Map the original class label to the new label
                labels.append(class_mapping[sign])
            except:
                print("Error loading image")
    
    # Converting lists into numpy arrays
    data = np.array(data)
    labels = np.array(labels)
    
    # Normalize image data
    data = data / 255.0  # Normalize pixel values to [0, 1]
        
    # Splitting training and testing dataset
    X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, random_state=42)
    
    # Apply oversampling to the training set for specified classes
    target_classes = [2, 14, 33, 34]
    X_train_balanced, y_train_balanced = apply_oversampling(X_train, y_train, target_classes)
    
    # Convert the labels into one-hot encoding
    y_train_balanced = to_categorical(y_train_balanced, classes)
    y_test = to_categorical(y_test, classes)
    
    print(f"Original X_train shape: {X_train.shape}, Original y_train shape: {y_train.shape}")
    print(f"Oversampled X_train shape: {X_train_balanced.shape}, Oversampled y_train shape: {y_train_balanced.shape}")
    print(f"X_test.shape: {X_test.shape}, y_test.shape: {y_test.shape}")
    
    # Show first image
    plt.imshow(X_train[0])
    plt.show()
    
    # Convert the one-hot encoded labels back to class labels
    y_train_balanced_labels = np.argmax(y_train_balanced, axis=1)
    
    # Show the class distribution in the original training set
    original_class_distribution = Counter(y_train)
    print(f"Original Class Distribution in Training Set: {original_class_distribution}")
    
    # Show the class distribution in the undersampled training set
    oversampled_class_distribution = Counter(y_train_balanced_labels)
    print(f"Oversampled Class Distribution in Training Set: {oversampled_class_distribution}")

    # Plot the class distribution before and after undersampling
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    
    # Plot original class distribution
    axes[0].bar(original_class_distribution.keys(), original_class_distribution.values(), color='lightcoral')
    axes[0].set_title('Original Class Distribution (Training Set)')
    axes[0].set_xlabel('Class Labels')
    axes[0].set_ylabel('Number of Samples')
    
    # Plot undersampled class distribution
    axes[1].bar(oversampled_class_distribution.keys(), oversampled_class_distribution.values(), color='lightgreen')
    axes[1].set_title('Oversampled Class Distribution (Training Set)')
    axes[1].set_xlabel('Class Labels')
    axes[1].set_ylabel('Number of Samples')
    
    plt.tight_layout()
    plt.show()

    start1 = time.time()
    
    def build_cnn():
        model = Sequential()
        model.add(Conv2D(filters=32, kernel_size=(5, 5), activation='relu', input_shape=X_train.shape[1:]))
        model.add(Conv2D(filters=64, kernel_size=(5, 5), activation='relu'))
        model.add(MaxPool2D(pool_size=(2, 2)))
        model.add(Dropout(rate=0.15))
        model.add(Conv2D(filters=128, kernel_size=(3, 3), activation='relu'))
        model.add(MaxPool2D(pool_size=(2, 2)))
        model.add(Dropout(rate=0.20))
        model.add(Flatten())
        model.add(Dense(256, activation='relu'))
        model.add(Dropout(rate=0.25))
        model.add(Dense(classes, activation='softmax'))
    
        # Compile the model
        model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
        return model
    
    # Build and train the CNN model
    model = build_cnn()
    history = model.fit(
        X_train_balanced, y_train_balanced, batch_size=32, epochs=20, validation_data=(X_test, y_test)
    )
    
    # Plot training history
    plot_history(history)
    
    # Importing the test dataset
    y_test = pd.read_csv('./gtsrb-german-traffic-sign/Test.csv')
    
    # Filter the rows where ClassId is in the 'signs' list
    y_test_filtered = y_test[y_test['ClassId'].isin(signs)]
    
    # Extract the labels and image paths for the filtered rows
    test_labels = y_test_filtered["ClassId"].values
    imgs = y_test_filtered["Path"].values
    
    data = []
    
    # Retrieve the images
    with tf.device('/GPU:0'):
        for img in imgs:
            image = Image.open('./gtsrb-german-traffic-sign/' + img)
            image = image.resize((128, 128))
            data.append(np.array(image))
    
    X_test = np.array(data)
    
    X_test = X_test / 255.0  # Normalize test data
    
    # Map the test labels to the same range as the training labels
    test_labels_mapped = [class_mapping[label] for label in test_labels]
    
    # One-hot encode the test labels
    y_test = to_categorical(test_labels_mapped, classes)
    
    print(f"X_test.shape: {X_test.shape}, y_test.shape: {y_test.shape}")
    
    # Convert the one-hot encoded test labels back to class labels
    y_test_labels = np.argmax(y_test, axis=1)
    
    '''
    
    # Normalize and preprocess the test data
    # Filter the rows where ClassId is in the 'signs' list
    # Normalize and preprocess the test data
    # Filter the rows where ClassId is in the 'signs' list
    y_test_filtered_indices = np.isin(y_test, signs)  # Get a boolean mask for target classes
    X_test_filtered = X_test[y_test_filtered_indices]  # Filter corresponding images
    y_test_filtered = y_test[y_test_filtered_indices]  # Filter labels
    
    data = []
    for img in X_test_filtered:
        image = Image.fromarray((img * 255).astype(np.uint8))  # Convert to PIL Image if needed
        image = image.resize((128, 128))
        data.append(np.array(image))
    
    X_test_filtered = np.array(data) / 255.0  # Normalize the test data
    test_labels_mapped = [class_mapping[label] for label in y_test_filtered]  # Map test labels
    '''
        
    # Perform adversarial testing
    noise_intensities = [0, 0.25, 0.5, 0.75]
    target_class = 14  # Example target class index (14 is index 1)
    
    
    #unique_labels = sorted(set(test_labels_mapped))
    
    #if target_class not in unique_labels:
    #    raise ValueError(f"Target class {target_class} not found in test dataset.")
    
    # Map test labels to the appropriate class range
    test_labels_mapped = [class_mapping[label] for label in test_labels if label in class_mapping]
    
    # Debugging: Check available classes
    #print(f"Mapped test labels: {test_labels_mapped}")
    #print(f"Unique classes in test_labels_mapped: {set(test_labels_mapped)}")
    
    # Check if target class exists
    #if target_class not in test_labels_mapped:
        #print(f"Target class {target_class} not found in test data.")
        #print(f"Available test classes: {set(test_labels_mapped)}")
            
    avg_probabilities = adversarial_testing(model, X_test, test_labels, target_class, noise_intensities) #try y_test_labels instead of test_labels
    
    # Visualization
    class_labels = ['Speed', 'Stop', 'Right', 'Left']  # Replace with actual labels
    fig, axes = plt.subplots(1, len(noise_intensities), figsize=(20, 6))

    for i, (intensity, probs) in enumerate(avg_probabilities.items()):
        print('i, intensity, probs')
        print(i, intensity, probs)
        axes[i].bar(class_labels, probs, color='skyblue')
        axes[i].set_title(f'{intensity * 100}% Noise')
        axes[i].set_xlabel('Class')
        axes[i].set_ylabel('Average Probability')
        axes[i].set_ylim(0, 1)
    
    plt.tight_layout()
    plt.show()
    

if __name__ == '__main__':
    np.random.seed(42)
    tf.random.set_seed(42)
    random.seed(42)
    main()