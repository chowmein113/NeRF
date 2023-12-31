import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
from tools import *
from tqdm import tqdm
import multiprocessing
import torch.multiprocessing as mp
class TrainImageDataSet(Dataset):
    def __init__(self, image: np.array, num_sample: float):
        self.image = image
        self.num_sample = num_sample
        self.width = image.shape[1]
        self.height = image.shape[0]
        self.coords = np.indices(self.image.shape[:2]).reshape(2, -1).T
    def __len__(self):
        return self.coords.shape[0]
    def __getitem__(self, idx):
        coord = self.coords[idx, ::-1] #(r, c) -> (x, y) format
        norm_coord = self.normalize_coords(coord)
        clr = self.image[coord[1], coord[0]] #back to coords
        return torch.from_numpy(norm_coord).float(), torch.from_numpy(clr).float()


class ImageDataSet(Dataset):
    def __init__(self, image: np.array, num_sample: float):
        """Ordered Image Dataset for prediction and test validation

        Args:
            img (np.array): rgb image as np array
            coords (np.array): N x 2 np array of coordinates in image
            num_sample (float): batch size
        """
        self.image = image
        self.num_sample = num_sample
        self.width = image.shape[1]
        self.height = image.shape[0]
        self.coords = np.indices(self.image.shape[:2]).reshape(2, -1).T
        #get coords in N x 2 in (r, c) format
    def normalize_coords(self, coords: np.array) -> np.array:
        return coords / np.array([self.width - 1, self.height - 1])   
    def __len__(self):
        return self.coords.shape[0] // self.num_sample
    def __getitem__(self, idx):
        end = min((idx + 1) * self.num_sample, self.coords.shape[0])
        batch_coords = self.coords[idx * self.num_sample : end, ::-1] #(r, c) -> (x, y) format
        norm_coords = self.normalize_coords(batch_coords)
        clr = self.image[batch_coords[:, 1], batch_coords[:, 0]] #back to coords in (r, c) format to get pixels
        return torch.from_numpy(norm_coords).float(), torch.from_numpy(clr).float()
class RandomImageDataSet(ImageDataSet):
    def __init__(self, image: np.array, num_sample: float):
        super(RandomImageDataSet, self).__init__(image, num_sample)
        np.random.seed(42)
        np.random.shuffle(self.coords)
    def __len__(self):
        return self.coords.shape[0] // self.num_sample
    def __getitem__(self, idx):
        end = min((idx + 1) * self.num_sample, self.coords.shape[0])
        batch_coords = self.coords[idx * self.num_sample : end, ::-1] #(r, c) -> (x, y) format
        norm_coords = self.normalize_coords(batch_coords)
        clr = self.image[batch_coords[:, 1], batch_coords[:, 0]] #back to coords in (r, c) format to get pixels
        return torch.from_numpy(norm_coords).float(), torch.from_numpy(clr).float()
    # def __getitem__(self, _):
    #     img_coords = np.random.randint(0, [self.width, self.height], size = (self.num_sample, 2))
    #     norm_coords = self.normalize_coords(img_coords)
    #     m1 = np.min(norm_coords[:, 0])
    #     M1 = np.max(norm_coords[:, 0])
    #     m1 = np.min(norm_coords[:, 0])
    #     M1 = np.max(norm_coords[:, 0])
    #     clr = self.image[img_coords[:, 1], img_coords[:, 0]] / 255.0
    #     return torch.from_numpy(norm_coords).float(), torch.from_numpy(clr).float()
    # def __len__(self):
    #     return self.num_sample
    # def __getitem__(self, _):
    #     img_coord = np.random.randint(0, [self.width, self.height])
    #     norm_coord = self.normalize_coords(img_coord)
    #     clr = self.image[img_coord[1], img_coord[0]] / 255.0
    #     return torch.from_numpy(norm_coord).float(), torch.from_numpy(clr).float()
def camera_pixel_to_ray(pixel_pair: np.array, c2w, K):
            """Turns a uv pixel to r0 and rd rays in form
            [[cam_num, cam_num, cam_num]
             [r0, rd, rgb]]"""
            #pixel_pair form : (cam_num, x, y, r, g, b)
            cam_num = pixel_pair[0].int().item()
            uv = pixel_pair[1:3]
            clr = pixel_pair[3:]
            # pixel_coord = (uv - 0.5).int().numpy()
            # clr = torch.tensor(clr) # (x, y) -> (r, c)
            
            
            c2w_mat = c2w[cam_num]
            r0, rd = pixel_to_ray(K, c2w_mat, uv)
            ray_color_pair = torch.vstack([r0, rd, clr]).T
            ray_color_pair = torch.vstack([torch.ones(1, ray_color_pair.size(1)), ray_color_pair])
            #[[cam_num, cam_num, cam_num]
            #[r0, rd, rgb]
            #[r0, rd, rgb]
            #[r0, rd, rgb]]
            return ray_color_pair.numpy()
def worker_camera_pixels_to_rays(pixel_pairs: torch.Tensor, shared_tensor: torch.Tensor, c2w, K, i, chunk_size):
    lst = []
    for pixel_pair in tqdm(pixel_pairs, "Ray generation: "):
        lst.append(camera_pixel_to_ray(pixel_pair, c2w, K))
    result = torch.from_numpy(np.array(lst))
    end = min(shared_tensor.size(0), (i + 1) * chunk_size)
    shared_tensor[i * chunk_size:end, :, :] = result
    return
class NerfDataSet(Dataset):
    def __init__(self, data: np.array, num_samples: int, 
                 num_workers: int, f: float, c2w: np.array, 
                 im_height  = None, im_width = None, shuffle = False):
        self.workers = min(multiprocessing.cpu_count(), num_workers)
        self.num_samples = num_samples
        self.camera_pixel_pairs = None
        self.num_workers = num_workers
        self.c2w = torch.from_numpy(c2w)
        self.data = data
        self.f = f
        self.K = None
        self.ray_color_pairs = None
        if im_height is not None and im_width is not None:
            self.K = intrinsic_K(f, im_height, im_width)
            
        #make pixel camera pairs
        for i in tqdm(range(data.shape[0]), "Adding camera pixel pairs: "):
            self.add_flattened_data(data, i)
            
            
        # self.camera_pixel_pairs = self.normalize_coords(self.camera_pixel_pairs)
        self.camera_pixel_pairs = torch.from_numpy(self.camera_pixel_pairs)
        torch.set_num_threads(1)
        self.camera_pixel_pairs.share_memory_()
        self.c2w.share_memory_()
        assert self.K is not None
        self.K.share_memory_()
        result_tensor = torch.zeros(self.camera_pixel_pairs.size(0), 4, 3)
        result_tensor.share_memory_()
        args = torch.chunk(self.camera_pixel_pairs, self.workers)
        chunk_size = self.camera_pixel_pairs.size(0) // self.workers
        processes = []
        for i, arg in enumerate(args):
            p = multiprocessing.Process(target = worker_camera_pixels_to_rays, args = (arg, result_tensor, self.c2w, self.K, i, chunk_size))
            p.start()
            processes.append(p)
        for p in processes:
            p.join()
        
        shuffle_indices = torch.randperm(result_tensor.size(0))
        self.ray_color_pairs: torch.Tensor = result_tensor[shuffle_indices, :, :]
        # r0_max = self.ray_color_pairs[:, 1:, 0].max()
        # rd_max = self.ray_color_pairs[:, 1:, 1].max()
        # norm_divisor = 
        del self.camera_pixel_pairs
        if shuffle:
            self.shuffle()
        #make rays from these pairs
        # split pairs into chunks
        # multi process and turn into rays
        
        # chunks = torch.chunk(self.camera_pixel_pairs, multiprocessing.cpu_count())
        # clone = self.camera_pixel_pairs.clone()
        
        return
    def shuffle(self):
        indices = torch.randperm(self.ray_color_pairs.shape[0])
        self.ray_color_pairs = self.ray_color_pairs[indices]
    def normalize_coords(self, coords: np.array) -> np.array:
        return coords / np.array([self.width - 1, self.height - 1]) 
    def camera_pixel_to_ray(self, pixel_pair: np.array):
            """Turns a uv pixel to r0 and rd rays in form
            [[cam_num, cam_num, cam_num]
             [r0, rd, rgb]]"""
            #pixel_pair form : (cam_num, x, y, r, g, b)
            cam_num = pixel_pair[0].int().item()
            uv = pixel_pair[1:3]
            clr = pixel_pair[3:]
            # pixel_coord = (uv - 0.5).int().numpy()
            
            # clr = torch.tensor(clr) # (x, y) -> (r, c)
            K = None
            if self.K is None:
                img = self.data[cam_num]
                K = intrinsic_K(self.f, img.shape[0], img.shape[1])
            else:
                K = self.K
            c2w_mat = self.c2w[cam_num]
            r0, rd = pixel_to_ray(K, c2w_mat, uv)
            ray_color_pair = torch.vstack([r0, rd, clr]).T
            ray_color_pair = torch.vstack([torch.ones(1, ray_color_pair.size(1)), ray_color_pair])
            #[[cam_num, cam_num, cam_num]
            #[r0, rd, rgb]
            #[r0, rd, rgb]
            #[r0, rd, rgb]]
            return ray_color_pair     
    def add_flattened_data(self, data: np.array, camera: int):
        image = data[camera]
        coords = np.indices(image.shape[:2]).reshape(2, -1).T #(r, c) form
        clrs = image[coords[:, 0], coords[:, 1]]
        coords = coords[:, ::-1] + 0.5 #(r, c) -> (x, y) or (u, v) format
        cam_num = np.zeros((coords.shape[0], 1))
        cam_num += camera
        coords = np.hstack((cam_num, coords, clrs))
        if self.camera_pixel_pairs is None:
            self.camera_pixel_pairs = coords
        else:
            self.camera_pixel_pairs = np.vstack((self.camera_pixel_pairs, coords))
        return
    def get_rays_by_idx(self, indices):
        return self.ray_color_pairs[indices]
    def sample_rays(self, size: int):
        end = min(self.ray_color_pairs.size(0), size)
        return self.ray_color_pairs[:end]
    def __len__(self):
        return (self.ray_color_pairs.size(0) // self.num_samples) - \
            1 + (1 if (self.ray_color_pairs.size(0) % self.num_samples) > 0 \
                else 0) \
                    if self.ray_color_pairs is not None else 0
    
    def __getitem__(self, idx):
        if self.ray_color_pairs is None:
            return None
        end = min(self.ray_color_pairs.size(0), (idx + 1) * self.num_samples)
        sample = self.ray_color_pairs[idx * self.num_samples: end]
        #TODO: figure out how to grab sample with rays AND color for each pixel while having a good batch load
        return sample
class NerfSingularDataSet(Dataset):
    def __init__(self, data: np.array, num_samples: int, 
                 num_workers: int, f: float, c2w: np.array, 
                 im_height  = None, im_width = None):
        self.num_samples = num_samples
        self.camera_pixel_pairs = None
        self.num_workers = num_workers
        self.c2w = torch.from_numpy(c2w)
        self.data = data
        self.f = f
        self.K = None
        self.ray_color_pairs = []
        if im_height is not None and im_width is not None:
            self.K = intrinsic_K(f, im_height, im_width)
        #make pixel camera pairs
        for i in tqdm(range(data.shape[0]), "Adding camera pixel pairs: "):
            self.add_flattened_data(data, i)
        self.camera_pixel_pairs = torch.from_numpy(self.camera_pixel_pairs)
        #make rays from these pairs
        # split pairs into chunks
        # multi process and turn into rays
        
        # chunks = torch.chunk(self.camera_pixel_pairs, multiprocessing.cpu_count())
        # clone = self.camera_pixel_pairs.clone()
        
        return
        
    def camera_pixel_to_ray(self, pixel_pair: np.array):
            """Turns a uv pixel to r0 and rd rays in form
            [[cam_num, cam_num, cam_num]
             [r0, rd, rgb]]"""
            #pixel_pair form : (cam_num, x, y, r, g, b)
            cam_num = pixel_pair[0].int().item()
            uv = pixel_pair[1:3]
            clr = pixel_pair[3:]
            # pixel_coord = (uv - 0.5).int().numpy()
            # clr = torch.tensor(clr) # (x, y) -> (r, c)
            K = None
            if self.K is None:
                img = self.data[cam_num]
                K = intrinsic_K(self.f, img.shape[0], img.shape[1])
            else:
                K = self.K
            c2w_mat = self.c2w[cam_num]
            r0, rd = pixel_to_ray(K, c2w_mat, uv)
            ray_color_pair = torch.vstack([r0, rd, clr]).T
            ray_color_pair = torch.vstack([cam_num * torch.ones(1, ray_color_pair.size(1)), ray_color_pair])
            #[[cam_num, cam_num, cam_num]
            #[r0, rd, rgb]
            #[r0, rd, rgb]
            #[r0, rd, rgb]]
            return ray_color_pair   
    def add_flattened_data(self, data: np.array, camera: int):
        image = data[camera]
        coords = np.indices(image.shape[:2]).reshape(2, -1).T #(r, c) form
        clrs = image[coords[:, 0], coords[:, 1]]
        coords = coords[:, ::-1] + 0.5 #(r, c) -> (x, y) or (u, v) format
        cam_num = np.zeros((coords.shape[0], 1))
        cam_num += camera
        coords = np.hstack((cam_num, coords, clrs))
        if self.camera_pixel_pairs is None:
            self.camera_pixel_pairs = coords
        else:
            self.camera_pixel_pairs = np.vstack((self.camera_pixel_pairs, coords))
        return
    def sample_rays(self, size: int):
        assert self.camera_pixel_pairs is not None
        randindxs = np.random.randint(0, len(self), size)
        return self.get_rays_by_idx(randindxs)
    def get_rays_by_idx(self, indices):
        res = []
        for idx in indices:
            res.append(self[idx])
        return torch.stack(res, dim=0)
    def __len__(self):
        return self.camera_pixel_pairs.shape[0] if self.camera_pixel_pairs is not None else 0
    
    def __getitem__(self, idx):
        if self.camera_pixel_pairs is None:
            return None
        pair = self.camera_pixel_pairs[idx]
        ray_pixel_pair_tensor = self.camera_pixel_to_ray(pair)
        sample = ray_pixel_pair_tensor
        #TODO: figure out how to grab sample with rays AND color for each pixel while having a good batch load
        return sample
class NerfTestSingularDataSet(Dataset):
    def __init__(self, num_samples: int, 
                 num_workers: int, f: float, c2w: np.array, 
                 im_height  = None, im_width = None):
        self.num_samples = num_samples
        self.camera_pixel_pairs = None
        self.num_workers = num_workers
        self.c2w = torch.from_numpy(c2w)
        self.f = f
        self.K = None
        self.ray_color_pairs = []
        self.im_height = im_height
        self.im_width = im_width
        if im_height is not None and im_width is not None:
            self.K = intrinsic_K(f, im_height, im_width)
        #make pixel camera pairs
        for i in tqdm(range(c2w.shape[0]), "Adding camera pixel pairs: "):
            self.add_flattened_data(i)
        self.camera_pixel_pairs = torch.from_numpy(self.camera_pixel_pairs)
        #make rays from these pairs
        # split pairs into chunks
        # multi process and turn into rays
        
        # chunks = torch.chunk(self.camera_pixel_pairs, multiprocessing.cpu_count())
        # clone = self.camera_pixel_pairs.clone()
        
        return
        
    def camera_pixel_to_ray(self, pixel_pair: np.array):
            """Turns a uv pixel to r0 and rd rays in form
            [[cam_num, cam_num]
             [r0, rd]]"""
            #pixel_pair form : (cam_num, x, y)
            cam_num = pixel_pair[0].int().item()
            uv = pixel_pair[1:]
            # pixel_coord = (uv - 0.5).int().numpy()
            # clr = torch.tensor(clr) # (x, y) -> (r, c)
            K = None
            if self.K is None:
                img = self.data[cam_num]
                K = intrinsic_K(self.f, img.shape[0], img.shape[1])
            else:
                K = self.K
            c2w_mat = self.c2w[cam_num]
            r0, rd = pixel_to_ray(K, c2w_mat, uv)
            uv = np.append(uv, 1)
            uv = torch.from_numpy(uv)
            ray_color_pair = torch.vstack([r0, rd, uv]).T
            ray_color_pair = torch.vstack([cam_num * torch.ones(1, ray_color_pair.size(1)), ray_color_pair])
            #[[cam_num, cam_num, cam_num]
            #[[r0,      rd,   uv]
            # [r0,      rd,   uv]
            # [r0,      rd,   1]]
            return ray_color_pair   
    def add_flattened_data(self, camera: int):
        # image = data[camera]
        coords = np.indices((self.im_height, self.im_width)).reshape(2, -1).T #(r, c) form
        coords = coords[:, ::-1] + 0.5 #(r, c) -> (x, y) or (u, v) format
        cam_num = np.zeros((coords.shape[0], 1))
        cam_num += camera
        coords = np.hstack((cam_num, coords))
        if self.camera_pixel_pairs is None:
            self.camera_pixel_pairs = coords
        else:
            self.camera_pixel_pairs = np.vstack((self.camera_pixel_pairs, coords))
        return
    def sample_rays(self, size: int):
        assert self.camera_pixel_pairs is not None
        randindxs = np.random.randint(0, len(self), size)
        return self.get_rays_by_idx(randindxs)
    def get_rays_by_idx(self, indices):
        res = []
        for idx in indices:
            res.append(self[idx])
        return torch.stack(res, dim=0)
    def __len__(self):
        return self.camera_pixel_pairs.shape[0] if self.camera_pixel_pairs is not None else 0
    
    def __getitem__(self, idx):
        if self.camera_pixel_pairs is None:
            return None
        pair = self.camera_pixel_pairs[idx]
        ray_pixel_pair_tensor = self.camera_pixel_to_ray(pair)
        sample = ray_pixel_pair_tensor
        #TODO: figure out how to grab sample with rays AND color for each pixel while having a good batch load
        return sample 
class ImageDataLoader(object):
    def __init__(self, sample_size: int = 10):
        self.sample_size = sample_size
        self.dataloaders = []
    
    
    def add_img(self, img: np.array):
        img_dataset = RandomImageDataSet(img, self.sample_size)
        img_loader = DataLoader(img_dataset, batch_size=1, shuffle=True)
        self.dataloaders.append(img_loader)
    def add_dataset(self, dataset: Dataset):
        img_loader = DataLoader(dataset, batch_size=1, shuffle=False)
        self.dataloaders.append(img_loader)
    def __len__(self):
        return len(self.dataloaders)
    def get_data_loader(self, idx):
        return self.dataloaders[idx]
    