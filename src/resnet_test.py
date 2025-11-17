import torch
import torchvision
import torchvision.transforms as transforms
import time

# Model
model = torchvision.models.resnet18(weights="IMAGENET1K_V1")
model.eval()

# Dataset
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

testset = torchvision.datasets.CIFAR10(
    root="./data", train=False, download=True, transform=transform
)
testloader = torch.utils.data.DataLoader(testset, batch_size=32)

# Time entire dataset inference
start = time.time()
with torch.no_grad():
    for images, _ in testloader:
        _ = model(images)
end = time.time()

print("Total time:", end - start, "sec")
