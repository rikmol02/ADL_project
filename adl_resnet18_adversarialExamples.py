import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import torchvision.models as models
from skimage.util import random_noise
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import RandomOverSampler
from collections import Counter
import time
from tensorflow.keras.utils import to_categorical
import tensorflow as tf
import pandas as pd

# Add salt-and-pepper noise
def add_salt_pepper_noise(images, intensity):
    noisy_images = []
    for image in images:
        noisy_image = random_noise(image, mode='s&p', amount=intensity)
        noisy_images.append((noisy_image * 255).astype(np.uint8))
    return np.array(noisy_images)

# Modify ResNet18
def modify_resnet18(num_classes):
    model = models.resnet18(pretrained=True)
    # Modify the first layer to accept (3, 128, 128) images
    model.conv1 = nn.Conv2d(3, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
    # Modify the final layer to output probabilities with softmax
    model.fc = nn.Sequential(
        nn.Linear(model.fc.in_features, num_classes),
        nn.Softmax(dim=1)  # Softmax for probabilities
    )
    return model

def adversarial_testing(model, X_test, y_test, target_class, noise_intensities, device):
    """
    Perform adversarial testing on samples of a target class with salt-and-pepper noise.
    :param model: Trained ResNet18 model.
    :param X_test: Test dataset (numpy array).
    :param y_test: Test labels (numpy array).
    :param target_class: Class to test for adversarial robustness.
    :param noise_intensities: List of noise intensity levels.
    :param device: Device for PyTorch (CPU/GPU).
    :return: Dictionary mapping noise intensity to average class probabilities.
    """
    # Set model to evaluation mode
    model.eval()
    
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

        # Convert noisy samples to PyTorch tensors
        noisy_samples_tensor = (
            torch.tensor(noisy_samples, dtype=torch.float32)
            .permute(0, 3, 1, 2)
            .to(device)
        )

        # Perform inference
        with torch.no_grad():
            logits = model(noisy_samples_tensor)  # Raw model outputs
            probabilities = torch.nn.functional.softmax(logits, dim=1)  # Convert to probabilities
        
        # Debugging: Print logits and probabilities
        print(f"Logits: {logits}")
        print(f"Probabilities: {probabilities}")

        # Compute average probabilities
        avg_probabilities[intensity] = probabilities.mean(dim=0).cpu().numpy()

    return avg_probabilities


# Main function
def main():
    data = []
    labels = []
    signs = [2, 14, 33, 34]
    classes = len(signs)
    class_mapping = {sign: idx for idx, sign in enumerate(signs)}

    # Load images
    for sign in signs:
        path = os.path.join('./gtsrb-german-traffic-sign/', 'Train', str(sign))
        for img_file in os.listdir(path):
            try:
                img = Image.open(os.path.join(path, img_file)).resize((128, 128))
                data.append(np.array(img))
                labels.append(class_mapping[sign])
            except:
                continue

    data = np.array(data) / 255.0
    labels = np.array(labels)

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, random_state=42)

    # Oversampling
    ros = RandomOverSampler(random_state=42)
    X_train_flat = X_train.reshape((X_train.shape[0], -1))
    X_train_resampled, y_train_resampled = ros.fit_resample(X_train_flat, y_train)
    X_train_resampled = X_train_resampled.reshape((-1, 128, 128, 3))

    # Convert to PyTorch datasets
    train_dataset = TensorDataset(
        torch.tensor(X_train_resampled, dtype=torch.float32).permute(0, 3, 1, 2),
        torch.tensor(y_train_resampled, dtype=torch.long)
    )
    test_dataset = TensorDataset(
        torch.tensor(X_test, dtype=torch.float32).permute(0, 3, 1, 2),
        torch.tensor(y_test, dtype=torch.long)
    )

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    # Initialize model, optimizer, and loss
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = modify_resnet18(len(signs)).to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()

    # Training loop (simplified for brevity)
    def train_model(model, train_dataloader, val_dataloader, criterion, optimizer, device, num_epochs=20):
        since = time.time()
        
        acc_history = []
        loss_history = []
        val_acc_history = []
        val_loss_history = []
        
        best_acc = 0.0
        
        for epoch in range(num_epochs):
            print('Epoch {}/{}'.format(epoch + 1, num_epochs))
            print('-' * 10)
            
            # Training phase
            model.train()  # Set model to training mode
            running_loss = 0.0
            running_corrects = 0
            
            for inputs, labels in train_dataloader:
                inputs = inputs.to(device)
                labels = labels.to(device)
                model.to(device)
                
                optimizer.zero_grad()  # Zero the parameter gradients
                
                # Forward pass
                outputs = model(inputs)  # Raw logits
                loss = criterion(outputs, labels)  # Loss with class indices
                
                # Backward pass and optimization
                loss.backward()
                optimizer.step()
                
                # Statistics
                _, preds = torch.max(outputs, 1)  # Get predicted class indices
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
            
            # Calculate training metrics
            train_loss = running_loss / len(train_dataloader.dataset)
            train_acc = running_corrects.double() / len(train_dataloader.dataset)
            
            print('Train Loss: {:.4f} Train Acc: {:.4f}'.format(train_loss, train_acc))
            
            # Validation phase
            model.eval()  # Set model to evaluation mode
            val_loss = 0.0
            val_corrects = 0
            
            with torch.no_grad():  # No need to compute gradients for validation
                for inputs, labels in val_dataloader:
                    inputs = inputs.to(device)
                    labels = labels.to(device)
                    
                    # Forward pass
                    outputs = model(inputs)  # Raw logits
                    loss = criterion(outputs, labels)  # Loss with class indices
                    
                    # Statistics
                    _, preds = torch.max(outputs, 1)  # Get predicted class indices
                    val_loss += loss.item() * inputs.size(0)
                    val_corrects += torch.sum(preds == labels.data)
            
            # Calculate validation metrics
            val_loss = val_loss / len(val_dataloader.dataset)
            val_acc = val_corrects.double() / len(val_dataloader.dataset)
            
            print('Validation Loss: {:.4f} Validation Acc: {:.4f}'.format(val_loss, val_acc))
            
            # Track best accuracy
            if val_acc > best_acc:
                best_acc = val_acc
            
            acc_history.append(train_acc.item())
            loss_history.append(train_loss)
            val_acc_history.append(val_acc.item())
            val_loss_history.append(val_loss)
            
            print()
    
        time_elapsed = time.time() - since
        print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
        print('Best Validation Acc: {:4f}'.format(best_acc))
        
        return acc_history, loss_history, val_acc_history, val_loss_history
    
    # Train model
    train_acc_hist, train_loss_hist, val_acc_hist, val_loss_hist = train_model(
        model, train_loader, val_loader, criterion, optimizer, device)

    # Adversarial testing
    noise_intensities = [0, 0.25, 0.5, 0.75]
    target_class = 14  # Example target class
    
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
    avg_probs = adversarial_testing(model, X_test, test_labels, target_class, noise_intensities, device)

    # Visualization
    class_labels = ['Speed', 'Stop', 'Right', 'Left']  # Replace with actual class names
    fig, axes = plt.subplots(1, len(noise_intensities), figsize=(20, 6))
    for i, (intensity, probs) in enumerate(avg_probs.items()):
        print('i, intensity, probs')
        print(i, intensity, probs)
        axes[i].bar(class_labels, probs, color='skyblue')
        axes[i].set_title(f'{intensity * 100}% Noise')
        axes[i].set_xlabel('Class')
        axes[i].set_ylabel('Probability')
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()