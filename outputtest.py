import torch
from torchvision import datasets, transforms
from matplotlib import pyplot as plt
from Main import CNN  # 确保 Main.py 中定义了 CNN 类

# 设置设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 指定模型加载路径并加载模型
model_load_path = '/Users/xingwenbo/Documents/CS4795/model_state_dict.pth'
model = CNN().to(device)
model.load_state_dict(torch.load(model_load_path, map_location=device))
model.eval()

# 定义数据加载器（例如，使用 MNIST 测试集）
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
test_set = datasets.MNIST('./data', download=True, train=False, transform=transform)
test_loader = torch.utils.data.DataLoader(test_set, batch_size=64, shuffle=True)


def visualize_predictions(model, device, data_loader, num_images=5):
    model.eval()
    images, labels = next(iter(data_loader))
    images, labels = images.to(device), labels.to(device)

    with torch.no_grad():
        outputs = model(images)
        _, predicted = torch.max(outputs, 1)

    plt.figure(figsize=(10, 4))
    for i in range(num_images):
        plt.subplot(1, num_images, i+1)
        plt.imshow(images[i].cpu().squeeze(), cmap='gray')
        plt.title(f'Pred: {predicted[i]}')
        plt.axis('off')
    plt.show()

# 使用此函数来可视化模型的预测
visualize_predictions(model, device, test_loader)
