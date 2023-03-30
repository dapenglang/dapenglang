import torch
from torchvision import models, transforms
from PIL import Image
import torchvision.transforms.functional as TF
import os

def get_image_paths(folder_path):
    # Signature: get_image_paths(folder_path: str) -> List[str]
    image_extensions = ['.jpg', '.jpeg', '.png']  # Supported image formats
    image_paths = []
    for file_name in os.listdir(folder_path):
        file_path = os.path.join(folder_path, file_name)
        if any(file_path.endswith(ext) for ext in image_extensions):
            image_paths.append(file_path)
    return image_paths

def cw_attack(image, target_class, model, c=1e-4, kappa=0, max_iter=1000, lr=0.01):
    # Signature: cw_attack(image: Tensor, target_class: int, model: nn.Module, c: float, kappa: float, max_iter: int, lr: float) -> Tensor
    # Based on implementation from: https://github.com/utkuozbulak/pytorch-cw2
    def f(x):
        output = model(x)
        output = torch.nn.functional.softmax(output, dim=1)
        return output[0][target_class]

    w = image.clone().detach()
    w.requires_grad = True
    optimizer = torch.optim.Adam([w], lr=lr)

    prev_loss = 1e6
    for step in range(max_iter):
        loss1 = torch.max(torch.zeros(1), torch.tensor(kappa) - f(w))
        loss2 = torch.norm((w - image).view(-1), p=float('inf'))
        loss = loss1 + c * loss2

        optimizer.zero_grad()
        loss.backward(retain_graph=True)
        optimizer.step()

        if step % (max_iter // 10) == 0:
            print(f'Step {step}/{max_iter} - Loss: {loss.item()}')

        if loss.item() > prev_loss:
            break
        else:
            prev_loss = loss.item()

    perturbed_image = torch.clamp(w, 0, 1)
    return perturbed_image

def save_image(tensor, filename, size):
    # Signature: save_image(tensor: Tensor, filename: str, size: Tuple[int, int]) -> None
    image = TF.to_pil_image(tensor)
    image = image.resize(size)
    image.save(filename)

def main():
    # Define model and preprocessing
    model = models.resnet18(pretrained=True)
    model.eval()

    # Record input image size
    input_folder = r'C:\Users\Dell\Desktop\picture'
    input_size = Image.open(os.path.join(input_folder, os.listdir(input_folder)[0])).size

    preprocess = transforms.Compose([
        transforms.Resize(input_size),
        transforms.ToTensor(),
    ])

    # Traverse image folder
    folder_path = r'C:\Users\Dell\Desktop\picture'
    image_paths = get_image_paths(folder_path)

    # Perform attack and save perturbed images
    output_folder = r"C:\Users\Dell\Desktop\sample"
    target_class = 3  # Target class to attack
    c = 1e-4  # L2 regularization coefficient
    kappa = 0  # Confidence parameter
    max_iter = 10  # Maximum number of iterations for optimizer
    lr = 0.001  # Learning rate for optimizer

    for image_path in image_paths:
        # Load image
        image = Image.open(image_path)
        input_size = image.size
        preprocess = transforms.Compose([
            transforms.Resize(input_size),
            transforms.ToTensor(),
        ])
        image_tensor = preprocess(image).unsqueeze(0)

        # CW attack
        perturbed_image = cw_attack(image_tensor, target_class, model, c=c, kappa=kappa, max_iter=max_iter, lr=lr)

        # Save perturbed image
        file_name = os.path.splitext(os.path.basename(image_path))[0]
        new_file_path = os.path.join(output_folder, f'{file_name}_CW.jpg')
        save_image(perturbed_image.squeeze(), new_file_path, size=input_size)


if __name__ == '__main__':
    main()