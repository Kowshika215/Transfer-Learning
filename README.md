# Implementation-of-Transfer-Learning
## Aim
To Implement Transfer Learning for classification using VGG-19 architecture.
## Problem Statement and Dataset
Image classification from scratch requires a huge dataset and long training times. To overcome this, transfer learning can be applied using pre-trained models like VGG-19, which has already learned feature representations from a large dataset (ImageNet).

Problem Statement: Build an image classifier using VGG-19 pre-trained architecture, fine-tuned for a custom dataset (e.g., CIFAR-10, Flowers dataset, or any small image dataset).
Dataset: A dataset consisting of multiple image classes (e.g., train, test, and validation sets). For example, CIFAR-10 (10 classes of small images) or a custom dataset with multiple classes.

## DESIGN STEPS
## STEP 1:
Import the required libraries (PyTorch, torchvision, matplotlib, etc.) and set up the device (CPU/GPU).

## STEP 2:
Load the dataset (train and test). Apply transformations such as resizing, normalization, and augmentation. Create DataLoader objects.

## STEP 3:
Load the pre-trained VGG-19 model from torchvision.models. Modify the final fully connected layer to match the number of classes in the dataset.

## STEP 4:
Define the loss function (CrossEntropyLoss) and the optimizer (Adam).

## STEP 5:
Train the model for the required number of epochs while recording training loss and validation loss.

## STEP 6:
Evaluate the model using a confusion matrix, classification report, and test it on new samples.

## PROGRAM

```python
# Load Pretrained Model and Modify for Transfer Learning
model = models.vgg19(weights = models.VGG19_Weights.DEFAULT)

for param in model.parameters():
  param.requires_grad = False


# Modify the final fully connected layer to match the dataset classes

num_features = model.classifier[-1].in_features
model.classifier[-1] = nn.Linear(num_features,1)


# Include the Loss function and optimizer

criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(),lr=0.001)



# Train the model
def train_model(model, train_loader,test_loader,num_epochs=10):
    train_losses = []
    val_losses = []
    model.train()
    for epoch in range(num_epochs):
        running_loss = 0.0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            outputs = torch.sigmoid(outputs)
            labels = labels.float().unsqueeze(1)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        train_losses.append(running_loss / len(train_loader))

    # Compute validation loss
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for images, labels in test_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                outputs = torch.sigmoid(outputs)
                labels = labels.float().unsqueeze(1)
                loss = criterion(outputs, labels)
                val_loss += loss.item()

        val_losses.append(val_loss / len(test_loader))
        model.train()

        print(f'Epoch [{epoch+1}/{num_epochs}], Train Loss: {train_losses[-1]:.4f}, Validation Loss: {val_losses[-1]:.4f}')

    # Plot training and validation loss
    print("Name: KOWSHIKA R")
    print("Register Number: 212224220049")
    plt.figure(figsize=(8, 6))
    plt.plot(range(1, num_epochs + 1), train_losses, label='Train Loss', marker='o')
    plt.plot(range(1, num_epochs + 1), val_losses, label='Validation Loss', marker='s')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.show()

```

## OUTPUT
### Training Loss, Validation Loss Vs Iteration Plot





### Confusion Matrix



### Classification Report



### New Sample Prediction

![I4](https://github.com/Kowshika215/Transfer-Learning/blob/main/Screenshot%202026-03-15%20202923.png)

## RESULT

The VGG-19 model was successfully trained and optimized to classify defected and non-defected capacitors.
