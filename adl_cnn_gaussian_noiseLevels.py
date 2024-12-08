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

from collections import Counter

from imblearn.over_sampling import RandomOverSampler
from collections import Counter


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
    
def add_gaussian_noise(image, mean=0, std_dev=0.1):
    """
    Add Gaussian noise to an image.
    
    Parameters:
        image (numpy.ndarray): Input image array with values in [0, 1].
        mean (float): Mean of the Gaussian distribution.
        std_dev (float): Standard deviation of the Gaussian distribution.
    
    Returns:
        numpy.ndarray: Image with added Gaussian noise.
    """
    # Generate Gaussian noise
    noise = np.random.normal(mean, std_dev, image.shape)
    
    # Add the noise to the image
    noisy_image = image + noise
    
    # Clip the values to keep them in the valid range [0, 1]
    noisy_image = np.clip(noisy_image, 0, 1)
    
    return noisy_image

def create_noisy_dataset(X, y, noise_level=0.1, fraction=0.25):
    """
    Create a noisy dataset by replacing a fraction of the original dataset with noisy images.
    
    Parameters:
        X (numpy.ndarray): Input image data.
        y (numpy.ndarray): Corresponding labels.
        noise_level (float): Standard deviation of Gaussian noise.
        fraction (float): Fraction of the dataset to replace with noisy images.
    
    Returns:
        numpy.ndarray, numpy.ndarray: Dataset with noisy images replacing original images and corresponding labels.
    """
    num_samples = int(fraction * X.shape[0])  # Number of samples to replace with noisy images
    noisy_indices = random.sample(range(X.shape[0]), num_samples)  # Random indices to replace

    # Create noisy samples for the selected indices
    X_noisy = X.copy()
    for idx in noisy_indices:
        X_noisy[idx] = add_gaussian_noise(X[idx], std_dev=noise_level)

    # The labels remain unchanged
    return X_noisy, y



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
    noise_levels = [0.25, 0.5, 0.75]
    accuracies = []
    for n in noise_levels:
        print()
        print(f"Noise level: {n}")
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
        
        # Add p% noisy images to the training set
        X_train_balanced, y_train_balanced = create_noisy_dataset(X_train_balanced, y_train_balanced, noise_level=n, fraction=0.5)
        
        # Convert the labels into one-hot encoding
        y_train_balanced = to_categorical(y_train_balanced, classes)
        
        # Convert the labels into one-hot encoding
        y_test = to_categorical(y_test, classes)
        
        # Print updated dataset shapes
        print(f"Original X_train shape: {X_train.shape}, Original y_train shape: {y_train.shape}")
        print(f"Updated X_train shape: {X_train_balanced.shape}, y_train shape: {y_train_balanced.shape}")
        print(f"Updated X_test shape: {X_test.shape}, y_test shape: {y_test.shape}")
        
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
            # Building the model
            model = Sequential()
            model.add(Conv2D(filters=32, kernel_size=(5, 5), activation='relu', input_shape=X_train.shape[1:]))
            model.add(Conv2D(filters=64, kernel_size=(5, 5), activation='relu'))
            model.add(MaxPool2D(pool_size=(2, 2)))
            model.add(Dropout(rate=0.15))
            model.add(Conv2D(filters=128, kernel_size=(3, 3), activation='relu'))
            #model.add(Conv2D(filters=256, kernel_size=(3, 3), activation='relu'))
            model.add(MaxPool2D(pool_size=(2, 2)))
            model.add(Dropout(rate=0.20))
            model.add(Flatten())
            model.add(Dense(256, activation='relu')) #change back to 512!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
            model.add(Dropout(rate=0.25))
            model.add(Dense(classes, activation='softmax'))
            
            # Compilation of the model
            model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
            
            return model
        
        model = build_cnn()
        # Model display
        #model.summary() 
        
        # Train the model using the balanced data
        history1 = model.fit(X_train_balanced, y_train_balanced, batch_size=32, epochs=20, validation_data=(X_test, y_test))
        end1 = time.time()
        print(f"Time taken to fit model: {(end1 - start1):.2f}")
        
        # plot training and validation accuracy
        #plot_history(history1)
        
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
        
        # Show the class distribution in the test set
        class_distribution = Counter(y_test_labels)
        
        # Print the class distribution
        print(f"Class Distribution in Test Set: {class_distribution}")
        
        # Plot the class distribution
        plt.figure(figsize=(10, 6))
        plt.bar(class_distribution.keys(), class_distribution.values(), color='skyblue')
        plt.xlabel('Class Labels')
        plt.ylabel('Number of Samples')
        plt.title('Class Distribution in Test Set')
        plt.xticks(list(class_distribution.keys()))  # Ensure all class labels are shown on the x-axis
        plt.show()
    
        # Inference using the model
        with tf.device('/GPU:0'):
            pred = np.argmax(model.predict(X_test), axis=-1)
        
        # Accuracy with the test data
        print(accuracy_score(np.argmax(y_test, axis=-1), pred))
        
        # Assuming y_test (true labels in one-hot encoding) and pred (predicted class indices) are defined
        y_true = np.argmax(y_test, axis=-1)  # Convert one-hot encoded true labels to class indices
        y_pred = pred  # Predicted labels are already class indices
        
        # Print the classification report
        print(f"Classification Report:\n{classification_report(y_true, y_pred)}")
        
        # Generate the confusion matrix
        cm = confusion_matrix(y_true, y_pred)
        
        # Plotting the confusion matrix
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=np.arange(classes), yticklabels=np.arange(classes))
        plt.title("Confusion Matrix")
        plt.xlabel("Predicted Labels")
        plt.ylabel("True Labels")
        plt.show()
        
        # Debugging shapes of predictions and labels
        print(f"Predicted labels shape: {y_pred.shape}, True labels shape: {y_true.shape}")
        
        # Test set accuracy
        test_accuracy = accuracy_score(y_true, y_pred)
        print(f"Test Set Accuracy: {test_accuracy:.2f}")
        
        accuracies.append(test_accuracy)
    
    print(f'Accuracies: {accuracies}')
    plt.plot(noise_levels, accuracies, marker='o', linestyle='-', color='blue')
    plt.xlabel('Noise levels (%)')
    plt.ylabel('Test Accuracy')
    plt.title('Accuracy vs. Noise Levels')
    plt.xticks(noise_levels)
    plt.ylim(0, 1)  # Accuracy ranges from 0 to 1
    plt.grid(True)
    plt.show()
    

if __name__ == '__main__':
    np.random.seed(42)
    tf.random.set_seed(42)
    random.seed(42)
    main()
    
#noise levels [0.25, 0.5, 0.75] (fraction: 0.5), Accuracies: [0.9992592592592593, 0.9977777777777778, 0.9985185185185185] (classes: 2, 14, 33, 34)