import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torch.optim as optim
from data_loader import ImageDataSet, ImageDataLoader, RandomImageDataSet
import os.path as osp

from tools import volume_rendering


def volume_positional_encoding(x, L=10):
    # pe = [x]
    # for i in range(L):
    #     for fn in [torch.sin, torch.cos]:
    #         pe.append(fn(2. ** i * x))
    # return torch.cat(pe, dim=-1)
    #if x is (B, sample, 3)
    s = 2 ** torch.arange(L).float().to(x.device)
    s = s[None, None, :, None]
    x_u = x.unsqueeze(2)
    x_s = x_u * s
    #if x is (B * 32, sample)
    # s = 2 ** torch.arange(L).float().to(x.device)
    # s = s.view(1, L).to(x.device)
    
    # x_s = (x.unsqueeze(-1).expand(-1, -1, L) * s).to(x.device)
    sin_x = torch.sin(x_s)
    cos_x = torch.cos(x_s)
    pe = torch.cat((sin_x, cos_x), dim=-2)
    pe = pe.view(*x.shape[:2], -1)
    return torch.cat((x, pe), dim=-1)          
def positional_encoding(x, L=10):
    # pe = [x]
    # for i in range(L):
    #     for fn in [torch.sin, torch.cos]:
    #         pe.append(fn(2. ** i * x))
    # return torch.cat(pe, dim=-1)
    #if x is (B, sample, 3)
    # s = 2 ** torch.arange(L).float().to(x.device)
    # s = s[None, None, :, None]
    # x_u = x.unsqueeze(2)
    # x_s = x_u * s
    #if x is (B * 32, sample)
    s = 2 ** torch.arange(L).float().to(x.device)
    s = s.view(1, L).to(x.device)
    
    x_s = (x.unsqueeze(-1).expand(-1, -1, L) * s).to(x.device)
    sin_x = torch.sin(x_s)
    cos_x = torch.cos(x_s)
    pe = torch.cat((sin_x, cos_x), dim=-2).reshape(x.shape[0], -1)
    # pe = pe.view(*x.shape[:2], -1)
    return torch.cat((x, pe), dim=1)
class PositionalEncoder(nn.Module):
    def __init__(self, L=10, volume_batch = False):
        super(PositionalEncoder, self).__init__()
        self.L = L
        self.volume_batch = volume_batch
    def forward(self, x):
        if self.volume_batch:
            return volume_positional_encoding(x, self.L)
        else:
            return positional_encoding(x, self.L)
class MLP(nn.Module):
    def __init__(self, num_layers: int = 4, L: int = 10, hidden_size: int = 256):
        super().__init__()
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        #hyper parameters
        self.L = L
        
        input_dims = 2 * (2 * self.L + 1)
        layers = [PositionalEncoder(L=self.L), nn.Linear(input_dims, hidden_size), nn.ReLU()]
        for _ in range(num_layers - 2):
            layers.extend([nn.Linear(hidden_size, hidden_size), nn.ReLU()])
        layers.extend([nn.Linear(hidden_size, 3), nn.Sigmoid()])  # Output RGB
    
        self.layers = nn.Sequential(*layers)

    def forward(self, coords):
        # test = nn.Sequential(PositionalEncoder(L = self.L))(coords)
        return self.layers(coords)
class DeepNeRFModel(nn.Module):
    def __init__(self, num_pre_concat_layers: int = 4, 
                 num_post_concat_layers: int = 4, 
                 coord_freq_L: int = 10, ray_dir_freq_L: int = 4,
                 hidden_size: int = 256):
    
        super().__init__()
        self.num_pre_concat_layers = num_pre_concat_layers
        self.num_post_concat_layers = num_post_concat_layers
        self.hidden_size = hidden_size
        #hyper parameters
        self.coord_freq_L = coord_freq_L
        self.ray_dir_freq_L= ray_dir_freq_L
        coord_input_dims = 3 * (2 * self.coord_freq_L + 1)
        self.coord_input_dims = coord_input_dims
        ray_dir_input_dims = 3 * (2 * self.ray_dir_freq_L + 1)
        self.ray_dir_input_dims = ray_dir_input_dims
        self.x_pe = nn.Sequential(PositionalEncoder(L=self.coord_freq_L, volume_batch=True))
        layers = [nn.Linear(coord_input_dims, hidden_size), 
                  nn.ReLU()]
        for _ in range(self.num_pre_concat_layers - 1):
            layers.extend([nn.Linear(hidden_size, hidden_size), 
                           nn.ReLU()])
        
    
        self.pre_concat_layers = nn.Sequential(*layers)
        
        
        layers = [nn.Linear(hidden_size + coord_input_dims, hidden_size), 
                               nn.ReLU()] + layers[3:]
        self.post_concat_layers = nn.Sequential(*layers)
        
        self.density = nn.Sequential(nn.Linear(hidden_size, 1), 
                                     nn.ReLU())
        self.pre_ray_rgb = nn.Sequential(nn.Linear(hidden_size, 
                                                   hidden_size))
        self.ray_dir_pe = nn.Sequential(PositionalEncoder(L = self.ray_dir_freq_L))
        self.post_ray_rgb = nn.Sequential(nn.Linear(ray_dir_input_dims + hidden_size, 128), 
                                          nn.ReLU(), nn.Linear(128, 3), 
                                          nn.Sigmoid())
    def forward(self, x, rd):
        x_pe = self.x_pe(x)
        pre_concat_x = self.pre_concat_layers(x_pe)
        x_concat = torch.cat((pre_concat_x, x_pe), dim=-1)
        post_concat_x = self.post_concat_layers(x_concat)
        density = self.density(post_concat_x)
        pre_rgb = self.pre_ray_rgb(post_concat_x) #rn it is (B, sample, hidden_size)
        ray_pe = self.ray_dir_pe(rd) # (B, 3) -> (B, 27)
        ray_pe = ray_pe.unsqueeze(1).repeat(1, x_pe.shape[1],  1)
        
        rgb_concat = torch.cat((pre_rgb, ray_pe), dim=-1)
        rgb = self.post_ray_rgb(rgb_concat)
        #if using non sample size batch
        # value = torch.hstack((density, rgb))
            #value = tensor([d, r, g, b])
        return density, rgb
class NeRF(object):
    def __init__(self, layers: int = 4, L: int = 10, learning_rate: float = 1e-2, gpu_id: int = 0, pth: str = ""):
        
        
        device = torch.device("cuda:{}".format(gpu_id) if torch.cuda.is_available() else "cpu")
        print("device: " + "cuda:{}".format(gpu_id) if torch.cuda.is_available() else "cpu")
        self.device = device
        self.model = MLP(num_layers=layers, L = L)
    
        #load model weights/bias if exists
        if pth != "":
            self.model.load_state_dict(torch.load(pth))
        self.criterion = nn.MSELoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        self.ImageDataLoader = ImageDataLoader(sample_size=10000)
        self.psnrs = []
        self.model.to(device)
    
    def get_psnrs(self):
        return self.psnrs
    def psnr(self, mse: torch.Tensor, max_pixel_val: float = 1.0) -> torch.Tensor:
        if mse == 0.:
            return torch.tensor(0).to(self.device)
        inside = torch.tensor((max_pixel_val ** 2) / mse).to(self.device)
        return 10 * torch.log10(inside)
    def get_img_data_loader(self) -> ImageDataLoader:
        return self.ImageDataLoader
    def save_model(self, pth: str = osp.join(osp.dirname(osp.abspath(__file__)), "checkpoints", "nerf.pth")):
        #save model weights
        torch.save(self.model.state_dict(), pth)
        return
    
    def train(self, coords, actual_colors):
        #forward
        coords = coords.to(self.device)
        actual_colors = actual_colors.to(self.device)
        # self.optimizer.zero_grad()
        pred = self.model(coords)
        loss = self.criterion(pred, actual_colors)
        self.optimizer.zero_grad()
        #back propagation
        loss.backward()
        self.optimizer.step()
        
        #Calculate loss
        mse = loss.item()
        
        psnr = self.psnr(mse)
        self.psnrs.append(psnr.cpu().item())
        
        return
    @torch.no_grad()
    def pred(self, coords):
        predict = self.model(coords)
        return predict
    @torch.no_grad()
    def test(self, coords: torch.Tensor, actual_colors: torch.Tensor):
        
        pred = self.pred(coords)
        loss = self.criterion(pred, actual_colors)
        mse = loss.item()
        psnr = self.psnr(mse).cpu().item()
        self.psnrs.append(psnr)
        return pred
class DeepNeRF(NeRF):
    def __init__(self, num_pre_concat_layers: int = 4, 
                 num_post_concat_layers: int = 4, coord_freq_L: int = 10, 
                 ray_dir_freq_L: int = 4, learning_rate: float = 1e-2, 
                 gpu_id: int = 0, pth: str = "", pixel_depth = 32):
        
        device = torch.device("cuda:{}".format(gpu_id) if torch.cuda.is_available() else "cpu")
        self.device = device  
        self.model = DeepNeRFModel(num_pre_concat_layers=num_pre_concat_layers, 
                                   num_post_concat_layers=num_post_concat_layers,
                                   coord_freq_L=coord_freq_L, ray_dir_freq_L=ray_dir_freq_L,
                                   hidden_size=256)
        if pth != "":
            self.model.load_state_dict(torch.load(pth))
        self.criterion = nn.MSELoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        self.ImageDataLoader = ImageDataLoader(sample_size=10000)
        self.psnrs = []
        self.pixel_depth = pixel_depth
        self.model = self.model.to(device)  
    
    def save_model(self, pth: str = osp.join(osp.dirname(osp.abspath(__file__)), "checkpoints", "deep_nerf.pth")):
        #save model weights
        torch.save(self.model.state_dict(), pth)
        return
    @torch.no_grad()
    def pred(self, coords, ray_ds):
        coords = coords.to(self.device)
        ray_ds = ray_ds.to(self.device)
        # pred = self.model(coords, ray_ds)
        sigmas, rgbs = self.model(coords, ray_ds)
        # pred = pred.view(10000, self.pixel_depth, 4)
        # sigmas = pred[:, :, 0].unsqueeze(2)
        # rgbs = pred[:, :, 1:]
        colors = volume_rendering(sigmas, rgbs, step_size=(6.0 - 2.0) / self.pixel_depth)
        return colors
    @torch.no_grad()
    def test(self, coords, ray_ds, actual_colors):
        
        pred = self.pred(coords, ray_ds)
        loss = self.criterion(pred, actual_colors)
        mse = loss.item()
        psnr = self.psnr(mse).cpu().item()
        self.psnrs.append(psnr)
        return pred
    def train(self, coords, ray_ds, actual_colors):
        #forward
        coords = coords.to(self.device)
        ray_ds = ray_ds.to(self.device)
        actual_colors = actual_colors.to(self.device)
        # self.optimizer.zero_grad()
        sigmas, rgbs = self.model(coords, ray_ds)
        # pred = self.model(coords, ray_ds)
        # pred = pred.view(10000, self.pixel_depth, 4)
        # sigmas = pred[:, :, 0].unsqueeze(2)
        # rgbs = pred[:, :, 1:]
        colors = volume_rendering(sigmas, rgbs, step_size=(6.0 - 2.0) / self.pixel_depth)
        filter_test = torch.abs(colors - actual_colors)
        loss = self.criterion(colors, actual_colors).float()
        self.optimizer.zero_grad()
        #back propagation
        loss.backward()
        self.optimizer.step()
        
        #Calculate loss
        mse = loss.item()
        
        psnr = self.psnr(mse)
        self.psnrs.append(psnr.cpu().item())
        
    
    


