import torch 
import torch.nn as nn 
import torch.optim as optim 
import torchvision.transforms as transforms 
import torchvision.datasets as datasets 
import torchvision.models as models 
import torchvision.utils as utils 
import torchvision.transforms.functional as F 
import torchvision.transforms.functional as TF 
import torchvision.transforms.functional as TFS 
import torchvision.transforms.functional as TFS 
from vit import ViT




def evaluate(model, testloader, device): 
    model.eval()
    correct = 0 
    total = 0 
    with torch.no_grad(): 
        for images, labels in testloader: 
            images = images.to(device)
            labels = labels.to(device)
            outputs, _ = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    return 100 * correct / total

def train_model(model, trainloader, testloader, loss_fn, optimizer, device, epochs): 
    for epoch in range(epochs): 
        for i, (images, labels) in enumerate(trainloader): 
            images = images.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()

            outputs, _ = model(images)
            loss = loss_fn(outputs, labels)
            loss.backward()
            optimizer.step()

            if (i+1) % 100 == 0: 
                print(f"Epoch [{epoch+1}/{epochs}], Step [{i+1}/{len(trainloader)}], Loss: {loss.item():.4f}")
                print(f"Accuracy: {evaluate(model, testloader, device):.2f}%")

            
if __name__ == "__main__": 
    # import cifar and do train test split of dataloaders 

    # Define transforms for training and testing
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # Load CIFAR-10 dataset
    trainset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    testset = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)

    # Create data loaders
    batch_size = 32
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=4)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=4)

    # Get some basic info about the dataset
    classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Using device: {device}")

    config = {
        "hidden_size": 64, 
        "num_heads": 4, 
        "intermediate_size": 4*64, 
        "num_layers": 4, 
        "dropout": 0.1, 
        "num_classes": 10, 
        "image_size": 32, 
        "patch_size": 8, 
        "num_channels": 3
    }

    model = ViT(config).to(device)
    print(f"Model size: {sum(p.numel() for p in model.parameters())}")

    loss = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    epochs = 10
    train_model(model, trainloader, testloader, loss, optimizer, device, epochs)
    print("Training complete")
    print(f"Accuracy: {evaluate(model, testloader, device):.2f}%")
    torch.save(model.state_dict(), "vit.pth")
    print("Model saved")





