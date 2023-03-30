import torch
from torchvision import models, transforms
from PIL import Image
import torchvision.transforms.functional as TF
import os

def get_image_paths(folder_path):
    # 签名：get_image_paths(folder_path: str) -> List[str]
    image_extensions = ['.jpg', '.jpeg', '.png']  # 支持的图片格式
    image_paths = []
    for file_name in os.listdir(folder_path):
        file_path = os.path.join(folder_path, file_name)
        if any(file_path.endswith(ext) for ext in image_extensions):
            image_paths.append(file_path)
    return image_paths


def pgd_attack(image, model, epsilon=0.01, alpha=0.005, num_iter=10):
    # 签名：pgd_attack(image: Tensor, model: Module, epsilon: float, alpha: float, num_iter: int) -> Tensor
    original_label = torch.argmax(model(image))
    perturbed_image = image.clone().detach().requires_grad_(True)

    for i in range(num_iter):
        adv_logits = model(perturbed_image)
        loss = torch.nn.functional.cross_entropy(adv_logits, torch.tensor([original_label]))

        if perturbed_image.grad is not None:
            perturbed_image.grad.zero_()
        loss.backward()

        with torch.no_grad():
            perturbed_image += alpha * perturbed_image.grad.sign()
            perturbed_image = torch.min(torch.max(perturbed_image, image - epsilon), image + epsilon)
            perturbed_image = torch.clamp(perturbed_image, 0, 1)

        perturbed_image.detach_().requires_grad_()

    return perturbed_image


def save_image(tensor, filename,size):
    # 签名：save_image(tensor: Tensor, filename: str) -> None
    image = TF.to_pil_image(tensor)
    image = image.resize(size)
    image.save(filename)


def main():
    # 定义模型和预处理
    model = models.resnet18(pretrained=True)
    model.eval()

    # 记录输入图片的大小
    input_folder = r'C:\Users\Dell\Desktop\picture'
    input_size = Image.open(os.path.join(input_folder, os.listdir(input_folder)[0])).size

    preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
    ])

    # 遍历图像文件夹
    folder_path = r'C:\Users\Dell\Desktop\picture'
    image_paths = get_image_paths(folder_path)

    # 执行攻击并保存攻击后的图像
    output_folder = r"C:\Users\Dell\Desktop\sample"
    epsilon = 0.05
    alpha = 0.01
    num_iter = 10

    for image_path in image_paths:
        # 加载图像
        image = Image.open(image_path)
        input_size = image.size
        preprocess = transforms.Compose([
            transforms.Resize(input_size),
            transforms.ToTensor(),
        ])
        image_tensor = preprocess(image).unsqueeze(0)

        # PGD 攻击
        perturbed_image = pgd_attack(image_tensor, model, epsilon, alpha, num_iter)

        # 保存攻击后的图像
        file_name = os.path.splitext(os.path.basename(image_path))[0]
        new_file_path = os.path.join(output_folder, f'{file_name}_PGD.jpg')
        save_image(perturbed_image.squeeze(), new_file_path, size=input_size)


if __name__ == '__main__':
    main()