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


def fgsm_attack(image, epsilon, data_grad):
    # 签名：fgsm_attack(image: Tensor, epsilon: float, data_grad: Tensor) -> Tensor
    sign_data_grad = data_grad.sign()
    perturbed_image = image + epsilon * sign_data_grad
    perturbed_image = torch.clamp(perturbed_image, 0, 1)
    return perturbed_image


def save_image(tensor, filename, size):
    # 签名：save_image(tensor: Tensor, filename: str, size: Tuple[int, int]) -> None
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
        transforms.Resize(input_size),
        transforms.ToTensor(),
    ])

    # 遍历图像文件夹
    folder_path = r'C:\Users\Dell\Desktop\picture'
    image_paths = get_image_paths(folder_path)

    # 执行攻击并保存攻击后的图像
    output_folder = r"C:\Users\Dell\Desktop\sample"
    epsilon = 0.05

    for image_path in image_paths:
        # 加载图像
        image = Image.open(image_path)
        input_size = image.size
        preprocess = transforms.Compose([
            transforms.Resize(input_size),
            transforms.ToTensor(),
        ])
        image_tensor = preprocess(image).unsqueeze(0)

        # FGSM 攻击
        image_tensor.requires_grad = True
        output = model(image_tensor)
        target = torch.tensor([3])  # 这里假设需要攻击的类别为 3
        loss = torch.nn.functional.nll_loss(output, target)
        model.zero_grad()
        loss.backward()
        data_grad = image_tensor.grad.data
        perturbed_image = fgsm_attack(image_tensor, epsilon, data_grad)

        # 保存攻击后的图像
        file_name = os.path.splitext(os.path.basename(image_path))[0]
        new_file_path = os.path.join(output_folder, f'{file_name}_FGSM.jpg')
        save_image(perturbed_image.squeeze(), new_file_path, size=input_size)


if __name__ == '__main__':
    main()