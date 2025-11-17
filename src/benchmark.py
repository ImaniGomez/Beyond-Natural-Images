import time
import torch
import torchvision
import torchvision.transforms as transforms

def benchmark():
    #Load dataset
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])
    testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                           download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=32)

    # Load model
    model = torchvision.models.resnet18(pretrained=True)
    model.eval()

    # Time inference
    start = time.time()
    with torch.no_grad():
        for images, _ in testloader:
            _ = model(images)
    end = time.time()

    print("Total inference time:", end - start)

if __name__ == "__main__":
    benchmark()