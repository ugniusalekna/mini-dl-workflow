import random
import numpy as np
import torch
import matplotlib.pyplot as plt
from PIL import Image
from .fmaps import build_grid, build_fc_composite, update_hooks


def show_image_grid(dataset, class_map, num_images=16, title=''):
    reverse_class_map = {v: k for k, v in class_map.items()}
    indices = random.sample(range(len(dataset)), num_images)

    fig, axes = plt.subplots(4, 4, figsize=(12, 12))
    fig.suptitle(title, fontsize=16)
    for i, ax in enumerate(axes.flat):
        if i < num_images:
            img, label = dataset[indices[i]]
            ax.imshow(img.permute(1, 2, 0))
            ax.set_title(f'Label: {reverse_class_map[label]}')
            ax.axis('off')
    plt.show()


def show_image(image, title=''):
    if isinstance(image, torch.Tensor):
        image = image.permute(1, 2, 0).numpy()
    elif isinstance(image, np.ndarray):
        pass
    elif isinstance(image, Image.Image):
        image = np.array(image)
    else:
        raise TypeError("Unsupported image type")

    plt.imshow(image)
    plt.title(title)
    plt.axis('off')
    plt.show()
    

def visualize_fmap(model, img, layer_name='conv1', device='cpu', use_act=False):
    model.eval()
    activation = {}
    fc_layers = [name for name, _ in model.named_modules() if name.startswith("fc")]
    img_np = img.permute(1, 2, 0).detach().cpu().numpy()
    
    def forward(image):
        nonlocal activation, img_np
        with torch.no_grad():
            _ = model(image.unsqueeze(0).to(device))

        if layer_name.startswith('fc'):
            vis = build_fc_composite(activation, fc_layers, image.shape[-1], image.shape[-2], use_act=use_act)
        else:
            feat = activation[layer_name].squeeze(0)
            vis = plt.cm.viridis(build_grid(feat, use_act=use_act))[:, :, :3]
            
        return img_np, vis

    layer_lst = fc_layers if layer_name.startswith('fc') else [layer_name]
    update_hooks(model, activation, layer_lst)
    
    fig, axs = plt.subplots(1, 2, figsize=(10, 5))
    for ax, im in zip(axs, forward(img)):
       ax.imshow(im)
       ax.axis('off')
    plt.tight_layout()
    plt.show()