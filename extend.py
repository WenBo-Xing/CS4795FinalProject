

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
