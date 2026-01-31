import torch
import torchvision
import torch.nn as nn
import matplotlib.pyplot as plt
from torchcam.methods import GradCAM
from torchcam.utils import overlay_mask
from PIL import Image
from torchvision.transforms.functional import to_pil_image
from torchinfo import summary


def device_test (): 
    if torch.cuda.is_available():
        device = 'cuda'
    else:
        device = 'cpu'
    print(f"Using device: {device}")
    return device


def model(device):
    weights = torchvision.models.AlexNet_Weights.DEFAULT
    model = torchvision.models.alexnet(weights=weights).to(device)
    summary(model, input_size=(1, 3, 224, 224))
    
    return model, weights

def show_con2d(model):
    print("=========="*8)
    print ("\nConv2d-lager:")
    print("=========="*8)
    for name, module in model.named_modules():
        if isinstance(module, nn.Conv2d):
            print(name, module)
            print("----------"*8)

def load_image(path, device, weights):
    img = Image.open(path).convert('RGB')
    transforms = weights.transforms()
    preprocessed_img = transforms(img)
    tensor_img = preprocessed_img.unsqueeze(0).to(device)
    return img, tensor_img

def show_image (img):
    plt.figure(figsize=(5, 5))
    plt.imshow(img)
    plt.axis('off')
    plt.title(f"Original image")
    plt.tight_layout()
    plt.show()

def predict(model, weights, tensor_img):
    model.eval()
    output = model(tensor_img)
    y_pred = torch.argmax(output, dim = 1)
    class_name = weights.meta["categories"][y_pred.item()]
    print("=========="*5)
    print(f"Predikted klass of picture is: {class_name.upper()}.")
    print("=========="*5)
    return y_pred

def layer_vizualization(model, img, tensor_img,  y_pred, layer_number):
    cam_layer = GradCAM(model=model, target_layer=model.features[layer_number])
    output = model(tensor_img)
    cams_layer = cam_layer(class_idx=y_pred.tolist(), scores=output)
    cam = cams_layer[0]
    result_layer = overlay_mask(img, to_pil_image(cam, mode="F"), alpha=0.5)
    plt.figure(figsize=(5, 5))
    plt.imshow(result_layer)
    plt.axis('off')
    plt.title(f"Layer {layer_number}")
    plt.tight_layout()
    plt.show()
    return y_pred

def layers_vizualization(model, img, tensor_img, y_pred, layer_number):
    fig = plt.figure(figsize=(20,10))
    plt.subplot(1,6,1)
    plt.imshow(img)
    plt.axis('off')
    plt.title('Original image.')
    plt.tight_layout()

    for c, n in enumerate(layer_number):
        cam_layer = GradCAM(model=model, target_layer=model.features[n])
        output = model(tensor_img)
        cams_layer = cam_layer(class_idx=y_pred.tolist(), scores=output)
        cam = cams_layer[0]
        result_layer = overlay_mask(img, to_pil_image(cam, mode="F"), alpha=0.5)
        plt.subplot(1, 6, c+2)

        plt.imshow(result_layer)
        plt.axis('off')
        plt.title(f"Layer {n}.")
        plt.tight_layout()
    plt.show()