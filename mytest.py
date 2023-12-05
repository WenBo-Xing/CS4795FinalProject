import torch
from torchvision import transforms
from PIL import Image
from matplotlib import pyplot as plt
from Main import CNN  # 确保 Main.py 中定义了 CNN 类

# 加载模型
model_load_path = '/Users/xingwenbo/Documents/CS4795/model_state_dict.pth'
model = CNN()
model.load_state_dict(torch.load(model_load_path))
model.eval()
# 预处理图像的函数
def prepare_image(image_path):
    image = Image.open(image_path).convert('L')
    image = image.resize((28, 28))
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    image = transform(image)
    return image

# 加载和预处理图像
def load_and_prepare_images(image_paths):
    images = []
    for image_path in image_paths:
        image = prepare_image(image_path)
        images.append(image)
    images = torch.stack(images)
    return images

# 图像路径
image_paths = ['/Users/xingwenbo/Documents/CS4795/test1.png', '/Users/xingwenbo/Documents/CS4795/test2.png', '/Users/xingwenbo/Documents/CS4795/test3.png','/Users/xingwenbo/Documents/CS4795/test4.png']
prepared_images = load_and_prepare_images(image_paths)

# 使用模型进行预测
with torch.no_grad():
    outputs = model(prepared_images)
    _, predicted = torch.max(outputs, 1)

# 可视化预测结果
for i, image in enumerate(prepared_images):
    plt.imshow(image.squeeze(), cmap='gray')
    plt.title(f'Predicted: {predicted[i].item()}')
    plt.show()
