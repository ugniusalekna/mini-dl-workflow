import torch.nn as nn
import torchvision
from torch.utils.tensorboard import SummaryWriter


class Writer(SummaryWriter):
    def __init__(self, log_dir):
        super().__init__(log_dir)
        self.log_dir = log_dir
        self.embeddings = None

    def add_param_hist(self, model, global_step):
        for name, param in model.named_parameters():
            self.add_histogram(f'Weights/{name}', param, global_step)
            if param.grad is not None:
                self.add_histogram(f'Gradients/{name}', param.grad, global_step)

    def add_img_grid(self, tag, img_tensor, global_step):
        grid = torchvision.utils.make_grid(img_tensor[:min(32, img_tensor.size(0))])
        self.add_image(tag, grid, global_step)
        
    def _get_embedding_layer(self, model):
        fc_layers = [m for m in model.modules() if isinstance(m, nn.Linear)]
        if not fc_layers:
            raise ValueError("No fc layers in model.")
        return fc_layers[-1]

    def _capture_embeddings_pre(self, module, input):
        self.embeddings = input[0]
    
    def add_embeddings(self, model, metadata, label_img, global_step):
        embedding_layer = self._get_embedding_layer(model)
        hook_handle = embedding_layer.register_forward_pre_hook(self._capture_embeddings_pre)
        
        device = next(model.parameters()).device
        model(label_img.to(device))
        
        hook_handle.remove()
        
        self.add_embedding(self.embeddings,
                           metadata=metadata.argmax(dim=1).cpu().tolist(),
                           label_img=label_img,
                           global_step=global_step)