import random
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import torch
from torchvision.utils import make_grid
from .fmaps import build_grid, build_fc_composite, update_hooks


def show_image_grid(dataset, num_images=16, title=''):
    indices = random.sample(range(len(dataset)), num_images)
    images, _ = zip(*[dataset[i] for i in indices])

    image_tensor = torch.stack(images)
    grid = make_grid(image_tensor, nrow=int(num_images**0.5), padding=2, normalize=True)

    plt.figure(figsize=(10, 10))
    plt.imshow(grid.permute(1, 2, 0))
    plt.title(title, fontsize=16)
    plt.axis('off')
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
    

def visualize_fmap(net, img, layer_name='conv1', device='cpu', use_act=False):
    net.eval()
    activation = {}
    fc_layers = [name for name, _ in net.named_modules() if name.startswith("fc")]
    img_np = img.permute(1, 2, 0).detach().cpu().numpy()
    
    def forward(image):
        nonlocal activation, img_np
        with torch.no_grad():
            _ = net(image.unsqueeze(0).to(device))

        if layer_name.startswith('fc'):
            vis = build_fc_composite(activation, fc_layers, image.shape[-1], image.shape[-2], use_act=use_act)
        else:
            feat = activation[layer_name].squeeze(0)
            vis = plt.cm.viridis(build_grid(feat, use_act=use_act))[:, :, :3]
            
        return img_np, vis

    layer_lst = fc_layers if layer_name.startswith('fc') else [layer_name]
    update_hooks(net, activation, layer_lst)
    
    fig, axs = plt.subplots(1, 2, figsize=(10, 5))
    for ax, im in zip(axs, forward(img)):
       ax.imshow(im)
       ax.axis('off')
    plt.tight_layout()
    plt.show()