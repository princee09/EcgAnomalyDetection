import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import timm
from torchvision import datasets, transforms
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
import matplotlib.pyplot as plt

# Data loading
data_transforms = {
    'train': transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ]),
    'val': transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ]),
}

train_dataset = datasets.ImageFolder('ecg-classification-project/data/preprocessed/train', transform=data_transforms['train'])
val_dataset = datasets.ImageFolder('ecg-classification-project/data/preprocessed/val', transform=data_transforms['val'])

test_dataset = datasets.ImageFolder('ecg-classification-project/data/preprocessed/test', transform=data_transforms['val'])

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# Load pretrained ViT model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = timm.create_model('vit_base_patch16_224', pretrained=True, num_classes=4)  # 4 classes
model = model.to(device)

# Define loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

# Training loop
num_epochs = 50
best_val_acc = 0.0
for epoch in range(num_epochs):
    model.train()
    train_loss = 0.0
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()

    # Validation
    model.eval()
    val_correct = 0
    val_total = 0
    y_true_val = []
    y_pred_val = []
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            val_total += labels.size(0)
            val_correct += (predicted == labels).sum().item()
            y_true_val.extend(labels.cpu().numpy())
            y_pred_val.extend(predicted.cpu().numpy())

    val_acc = val_correct / val_total
    print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss/len(train_loader):.4f}, Val Accuracy: {val_acc:.4f}")

    # Save the best model
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        torch.save(model.state_dict(), 'output/Binusai.pth')

print(f"Best Validation Accuracy: {best_val_acc:.4f}")

# Validation Classification Report and Confusion Matrix
val_cm = confusion_matrix(y_true_val, y_pred_val)
print("Validation Classification Report:\n", classification_report(y_true_val, y_pred_val, digits=4))

def plot_confusion_matrix(cm, classes, title='Confusion Matrix'):
    plt.figure(figsize=(8,6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=classes, yticklabels=classes)
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title(title)
    plt.show()

plot_confusion_matrix(val_cm, train_dataset.classes, title="Validation Confusion Matrix")

# Test Metrics
test_correct = 0
test_total = 0
y_true_test = []
y_pred_test = []
model.eval()
with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs, 1)
        test_total += labels.size(0)
        test_correct += (predicted == labels).sum().item()
        y_true_test.extend(labels.cpu().numpy())
        y_pred_test.extend(predicted.cpu().numpy())

test_acc = test_correct / test_total
print(f"Test Accuracy: {test_acc:.4f}")

# Test Classification Report and Confusion Matrix
test_cm = confusion_matrix(y_true_test, y_pred_test)
print("Test Classification Report:\n", classification_report(y_true_test, y_pred_test, digits=4))
plot_confusion_matrix(test_cm, train_dataset.classes, title="Test Confusion Matrix")
