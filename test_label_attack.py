import torch
import torchvision
from torchvision import transforms
import torch.utils.data as data
import torch.optim as optim
import torch.nn as nn
from torchsummary import summary
import time
import numpy as np
import os
import matplotlib.pyplot as plt


def FGSM(image, label, model, criterion, epsilon=2/255.0):
    # Set requires_grad attribute of tensor. Important for Attack
    image.requires_grad = True
    
    # Forward pass the data through the model
    output, _ = model(image)
    loss = criterion(output, label)
    
    # Zero all existing gradients
    model.zero_grad()
    
    # Backward pass to calculate the gradients
    loss.backward()
    
    # Collect the element-wise sign of the data gradient
    sign_data_grad = image.grad.data.sign()
    
    # Create the perturbed image by adjusting each pixel of the input image
    perturbed_image = image + epsilon * sign_data_grad
    
    # Adding clipping to maintain the range of values
    perturbed_image = torch.clamp(perturbed_image, 0, 1)
    
    return perturbed_image


data_transforms = transforms.Compose([
    transforms.Resize([112, 112]),
    transforms.ToTensor()
    ])


BATCH_SIZE = 256
learning_rate = 0.001
EPOCHS = 30
numClasses = 43

# Define path of training data

train_data_path = "/Users/Nitro/Documents/GitHub/YSP-Traffic-Signal-Project/GTSRB/Train"
model_path = "/Users/Nitro/Documents/GitHub/YSP-Traffic-Signal-Project/GTSRB/Model_2.pth"

train_data = torchvision.datasets.ImageFolder(root = train_data_path, transform = data_transforms)

# Divide data into training and validation (0.8 and 0.2)
ratio = 0.8
n_train_examples = int(len(train_data) * ratio)
n_val_examples = len(train_data) - n_train_examples

train_data, val_data = data.random_split(train_data, [n_train_examples, n_val_examples])

train_loader = data.DataLoader(train_data, shuffle=True, batch_size = BATCH_SIZE)
val_loader = data.DataLoader(val_data, shuffle=True, batch_size = BATCH_SIZE)

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


from class_alexnetTS import AlexnetTS
model = AlexnetTS(numClasses)
model.load_state_dict(torch.load(model_path))
criterion = nn.CrossEntropyLoss()



def calculate_accuracy(y_pred, y):
    top_pred = y_pred.argmax(1, keepdim = True)
    correct = top_pred.eq(y.view_as(top_pred)).sum()
    acc = correct.float() / y.shape[0]
    return acc


def evaluate(model, loader, criterion):
    epoch_loss = 0
    epoch_acc = 0
    
    # Evaluate the model
    model.eval()
    
    
    for (images_batch, labels) in val_loader:
        perturbed_images_batch = []
        for i in range(0, len(images_batch)):
            perturbed_image = FGSM(images_batch[i].unsqueeze(0), labels[i].unsqueeze(0), model, criterion)
            perturbed_images_batch.append(perturbed_image)
        perturbed_images_batch = torch.cat(perturbed_images_batch)
        
        # Run predictions
        output, _ = model(perturbed_images_batch)
        loss = criterion(output, labels)
        
        # Calculate accuracy
        acc = calculate_accuracy(output, labels)
        
        epoch_loss += loss.item()
        epoch_acc += acc.item()
    
    return epoch_loss / len(loader), epoch_acc / len(loader)



val_loss_list = [0]*EPOCHS
val_acc_list = [0]*EPOCHS

for epoch in range(EPOCHS):

    val_start_time = time.monotonic()
    val_loss, val_acc = evaluate(model, val_loader, criterion)
    val_end_time = time.monotonic()

    val_loss_list[epoch] = val_loss
    val_acc_list[epoch] = val_acc
    
    print("Validation: Loss = %.4f, Accuracy = %.4f, Time = %.2f seconds" % (val_loss, val_acc, val_end_time - val_start_time))
    print("")