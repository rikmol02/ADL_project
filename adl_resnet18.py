import torchvision.models as models
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.models import Model
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense
import tensorflow as tf
import time
import numpy as np
import matplotlib.pyplot as plt
import os
from sklearn.model_selection import train_test_split
from collections import Counter
from sklearn.preprocessing import StandardScaler
from sklearn.utils import resample
import pandas as pd
from tensorflow.keras.utils import to_categorical
from imblearn.over_sampling import RandomOverSampler
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import optimizers
from PIL import Image
import torch
import torch.nn as nn
import random
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import seaborn as sns

# Define the plotting function
def plot_training_history(train_acc_hist, train_loss_hist, val_acc_hist, val_loss_hist):
    """
    Plots the training and validation accuracy and loss over epochs.
    
    Parameters:
    - train_acc_hist: List of training accuracy per epoch.
    - train_loss_hist: List of training loss per epoch.
    - val_acc_hist: List of validation accuracy per epoch.
    - val_loss_hist: List of validation loss per epoch.
    """
    epochs = range(1, len(train_acc_hist) + 1)
    
    plt.figure(figsize=(12, 6))
    
    # Accuracy plot
    plt.subplot(1, 2, 1)
    plt.plot(epochs, train_acc_hist, label='Training Accuracy', marker='o')
    plt.plot(epochs, val_acc_hist, label='Validation Accuracy', marker='o')
    plt.title('Accuracy over Epochs')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid()
    
    # Loss plot
    plt.subplot(1, 2, 2)
    plt.plot(epochs, train_loss_hist, label='Training Loss', marker='o')
    plt.plot(epochs, val_loss_hist, label='Validation Loss', marker='o')
    plt.title('Loss over Epochs')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid()
    
    plt.tight_layout()
    plt.show()

def apply_oversampling(X_train, y_train, target_classes):
    """
    Oversample the specified classes in the training set.
    """
    X_train_flat = X_train.reshape((X_train.shape[0], -1))
    
    unique_classes = np.unique(y_train)
    if len(unique_classes) <= 1:
        raise ValueError(f"The training set must contain more than one class for oversampling. Found: {unique_classes}")
    
    oversampler = RandomOverSampler(random_state=42)
    X_resampled, y_resampled = oversampler.fit_resample(X_train_flat, y_train)
    
    print("Before Oversampling:", Counter(y_train))
    print("After Oversampling:", Counter(y_resampled))
    
    X_resampled = X_resampled.reshape((-1, X_train.shape[1], X_train.shape[2], X_train.shape[3]))
    
    return X_resampled, y_resampled

def main():
    data = []
    labels = []
    signs = [2, 14, 33, 34]
    classes = len(signs)
    
    
    class_mapping = {sign: idx for idx, sign in enumerate(signs)}
    
    for sign in signs:
        path = os.path.join('./gtsrb-german-traffic-sign/', 'Train', str(sign))
        images = os.listdir(path)
    
        for image_file in images:
            try:
                image = Image.open(path + '/' + image_file)
                image = image.resize((128, 128))
                image = np.array(image)
                
                data.append(image)
                labels.append(class_mapping[sign])
            except:
                print("Error loading image")
    
    data = np.array(data)
    labels = np.array(labels)
    
    data = data / 255.0  # Normalize pixel values to [0, 1]
        
    X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, random_state=42)
    
    target_classes = [2, 14, 33, 34]
    X_train_balanced, y_train_balanced = apply_oversampling(X_train, y_train, target_classes)
    
    y_train_balanced = to_categorical(y_train_balanced, classes)
    y_test = to_categorical(y_test, classes)
    
    print(f"Original X_train shape: {X_train.shape}, Original y_train shape: {y_train.shape}")
    print(f"Oversampled X_train shape: {X_train_balanced.shape}, Oversampled y_train shape: {y_train_balanced.shape}")
    print(f"X_test.shape: {X_test.shape}, y_test.shape: {y_test.shape}")
    
    #plt.imshow(X_train[0])
    #plt.show()
    
    y_train_balanced_labels = np.argmax(y_train_balanced, axis=1)
    
    original_class_distribution = Counter(y_train)
    print(f"Original Class Distribution in Training Set: {original_class_distribution}")
    
    oversampled_class_distribution = Counter(y_train_balanced_labels)
    print(f"Oversampled Class Distribution in Training Set: {oversampled_class_distribution}")

    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    axes[0].bar(original_class_distribution.keys(), original_class_distribution.values(), color='lightcoral')
    axes[0].set_title('Original Class Distribution (Training Set)')
    axes[0].set_xlabel('Class Labels')
    axes[0].set_ylabel('Number of Samples')
    
    axes[1].bar(oversampled_class_distribution.keys(), oversampled_class_distribution.values(), color='lightgreen')
    axes[1].set_title('Oversampled Class Distribution (Training Set)')
    axes[1].set_xlabel('Class Labels')
    axes[1].set_ylabel('Number of Samples')
    
    plt.tight_layout()
    plt.show()

    start1 = time.time()
    
    def fine_tune_resnet(model, base_model):
        for name, module in list(base_model.named_children())[:3]:  # Freezing the first 3 layers
            for param in module.parameters():
                param.requires_grad = False
        return base_model
    
    model = models.resnet18(pretrained=True)
    model.fc = nn.Linear(model.fc.in_features, classes)
    
    dummy_input = torch.randn(1, 3, 128, 128)  
    output = model(dummy_input)

    # Modify the first convolutional layer to accept 128 input channels
    model.conv1 = nn.Conv2d(3, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)

    #model = fine_tune_resnet(model, resnet18)   
        
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
    
    # Ensure labels are class indices, not one-hot encoded
    y_train_balanced_labels = np.argmax(y_train_balanced, axis=1)
    y_test_labels = np.argmax(y_test, axis=1)
    
    # Convert to PyTorch tensors
    train_dataset = TensorDataset(
        torch.tensor(X_train_balanced, dtype=torch.float32).permute(0, 3, 1, 2),
        torch.tensor(y_train_balanced_labels, dtype=torch.long)
    )
    val_dataset = TensorDataset(
        torch.tensor(X_test, dtype=torch.float32).permute(0, 3, 1, 2),
        torch.tensor(y_test_labels, dtype=torch.long)
    )
    
    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Train model
    train_acc_hist, train_loss_hist, val_acc_hist, val_loss_hist = train_model(
        model, train_loader, val_loader, criterion, optimizer, device)
    
    end1 = time.time()
    print(f"Total training time: {end1 - start1:.2f} seconds")
    
    plot_training_history(train_acc_hist, train_loss_hist, val_acc_hist, val_loss_hist)
    
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
    X_test = X_test / 255.0  # Normalize pixel values to [0, 1]    
    
    # Convert test data to PyTorch tensors
    test_dataset = TensorDataset(
        torch.tensor(X_test, dtype=torch.float32).permute(0, 3, 1, 2),
        torch.tensor(test_labels, dtype=torch.long)
    )
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    
    # Inference using the model
    model.eval()  # Set model to evaluation mode
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
                        
            preds = torch.argmax(outputs, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    # Check the unique classes in all_labels and all_preds
    unique_labels = sorted(set(all_labels))

    all_labels = [class_mapping[label] for label in all_labels]
    
    # Calculate accuracy
    accuracy = accuracy_score(all_labels, all_preds)
    print(f"Test Accuracy: {accuracy * 100:.2f}%")
    
    # Generate confusion matrix
    conf_matrix = confusion_matrix(all_labels, all_preds)
    #print(f'all_labels: {all_labels}')
    #print(f'all_preds: {all_preds}')
    #print(set(all_labels))
    #print(set(all_preds))
    
    # Generate classification report with the correct labels
    report = classification_report(all_labels, all_preds, labels=unique_labels, target_names=[str(label) for label in unique_labels])
    print("Classification Report:")
    print(report)
    
    # Plot confusion matrix
    plt.figure(figsize=(8, 6))
    sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", xticklabels=unique_labels, yticklabels=unique_labels)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.show()
    
if __name__ == '__main__':
    np.random.seed(42)
    tf.random.set_seed(42)
    random.seed(42)
    main()
    
#ResNet-18 (1 epochs) (4 classes, OVERSAMPLED): 99.93 accuracy (time taken to fit model: 475 sec = 8 min.) (classes: [2, 14, 33, 34])
#ResNet-18 (20 epochs) (4 classes, OVERSAMPLED): 100.0 accuracy (time taken to fit model: 9042 sec = 151 min.) (classes: [2, 14, 33, 34])