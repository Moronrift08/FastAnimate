import os, sys
import math
import json
import importlib
from pathlib import Path

import cv2
import random
import numpy as np
from PIL import Image
import webdataset as wds
import pytorch_lightning as pl

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torchvision import transforms

from src.utils.train_util import instantiate_from_config
from src.utils.camera_util import (
    FOV_to_intrinsics, 
    center_looking_at_camera_pose, 
    get_circular_camera_poses,
)

from ..models.geometry.camera.perspective_camera import PerspectiveCamera
from ..models.geometry.render.neural_render import NeuralRender
import trimesh


def rotatedx(original_vertices):
        # 原始的vertices
    vertices = original_vertices

    # 定义旋转矩阵
    theta = np.radians(90)
    rotation_matrix = np.array([[1, 0, 0],
                                [0, np.cos(theta), -np.sin(theta)],
                                [0, np.sin(theta), np.cos(theta)]])

    # 对vertices进行旋转操作
    rotated_vertices = np.dot(rotation_matrix, vertices.T).T

    return rotated_vertices

def rotatedz(original_vertices):
        # 原始的vertices
    vertices = original_vertices

    # 定义旋转矩阵
    theta = np.radians(90)
    rotation_matrix = np.array([[np.cos(theta), -np.sin(theta), 0],
                                [np.sin(theta), np.cos(theta), 0],
                                [0, 0, 1]])
    # 对vertices进行旋转操作
    rotated_vertices = np.dot(rotation_matrix, vertices.T).T

    return rotated_vertices

def FOV_to_intrinsics(fov, device='cpu'):
    """
    Creates a 3x3 camera intrinsics matrix from the camera field of view, specified in degrees.
    Note the intrinsics are returned as normalized by image size, rather than in pixel units.
    Assumes principal point is at image center.
    """
    focal_length = 0.5 / np.tan(np.deg2rad(fov) * 0.5)
    intrinsics = torch.tensor([[focal_length, 0, 0.5], [0, focal_length, 0.5], [0, 0, 1]], device=device)
    return intrinsics

def spherical_camera_pose(azimuths: np.ndarray, elevations: np.ndarray, radius=2.5):
    azimuths = np.deg2rad(azimuths)
    elevations = np.deg2rad(elevations)

    xs = radius * np.cos(elevations) * np.cos(azimuths)
    ys = radius * np.cos(elevations) * np.sin(azimuths)
    zs = radius * np.sin(elevations)

    cam_locations = np.stack([xs, ys, zs], axis=-1)
    cam_locations = torch.from_numpy(cam_locations).float()

    c2ws = center_looking_at_camera_pose(cam_locations)
    return c2ws


class DataModuleFromConfig(pl.LightningDataModule):
    def __init__(
        self, 
        batch_size=8, 
        num_workers=4, 
        train=None, 
        validation=None, 
        test=None, 
        **kwargs,
    ):
        super().__init__()

        self.batch_size = batch_size
        self.num_workers = num_workers

        self.dataset_configs = dict()
        if train is not None:
            self.dataset_configs['train'] = train
        if validation is not None:
            self.dataset_configs['validation'] = validation
        if test is not None:
            self.dataset_configs['test'] = test
    
    def setup(self, stage):

        if stage in ['fit']:
            self.datasets = dict((k, instantiate_from_config(self.dataset_configs[k])) for k in self.dataset_configs)
        else:
            raise NotImplementedError

    def train_dataloader(self):

        sampler = DistributedSampler(self.datasets['train'])
        return wds.WebLoader(self.datasets['train'], batch_size=self.batch_size, num_workers=self.num_workers, shuffle=False, sampler=sampler)

    def val_dataloader(self):

        sampler = DistributedSampler(self.datasets['validation'])
        return wds.WebLoader(self.datasets['validation'], batch_size=1, num_workers=self.num_workers, shuffle=False, sampler=sampler)

    def test_dataloader(self):

        return wds.WebLoader(self.datasets['test'], batch_size=self.batch_size, num_workers=self.num_workers, shuffle=False)


class ObjaverseData(Dataset):
    def __init__(self,
        root_dir='objaverse/',
        meta_fname='valid_paths.json',
        input_image_dir='rendering_random_32views',
        target_image_dir='rendering_random_32views',
        input_view_num=6,
        target_view_num=4,
        total_view_n=32,
        fov=50,
        camera_rotation=True,
        validation=False,
    ):
        self.root = root_dir
        if 'CustomHumans' in root_dir:
            self.paths = []
            self.root_dir = Path(root_dir)
            folders = [f.path for f in os.scandir(root_dir) if f.is_dir()]
            self.paths = sorted(folders)[:100]

            self.paths = self.paths

        else:
            self.paths = []
            root_dir_1 = '/hpc2hdd/home/gzhang292/project2/THuman2.1'
            self.root_dir_1 = Path(root_dir_1)
            folders_1 = [f.path for f in os.scandir(root_dir_1) if f.is_dir()]
            self.paths_1 = sorted(folders_1)[:1000]

            root_dir_2 = '/hpc2hdd/home/gzhang292/project2/2K2K/generated/1M/1M'
            self.root_dir_2 = Path(root_dir_2)
            folders_2 = [f.path for f in os.scandir(root_dir_2) if f.is_dir()]
            self.paths_2 = sorted(folders_2)[:1000]

            self.paths = self.paths + self.paths_1 + self.paths_2

        print('============= length of dataset %d =============' % len(self.paths))

        self.fov = 30
        self.input_view_num = input_view_num
        self.target_view_num = target_view_num


        # self.camera = PerspectiveCamera(fovy=30, device='cpu')
        # self.renderer = NeuralRender(device='cpu', camera_model=self.camera)


        # self.input_image_dir = input_image_dir
        # self.target_image_dir = target_image_dir

        # self.input_view_num = input_view_num
        # self.target_view_num = target_view_num
        # self.total_view_n = total_view_n
        # self.fov = fov
        # self.camera_rotation = camera_rotation

        # with open(os.path.join(root_dir, meta_fname)) as f:
        #     filtered_dict = json.load(f)
        # paths = filtered_dict['good_objs']
        # self.paths = paths
        
        # self.depth_scale = 6.0
            
        # total_objects = len(self.paths)
        # print('============= length of dataset %d =============' % len(self.paths))



    def __len__(self):
        return len(self.paths)

    def load_im(self, path, color):
        '''
        replace background pixel with random color in rendering
        '''
        pil_img = Image.open(path)

        image = np.asarray(pil_img, dtype=np.float32) / 255.
        alpha = image[:, :, 3:]
        image = image[:, :, :3] * alpha + color * (1 - alpha)

        image = torch.from_numpy(image).permute(2, 0, 1).contiguous().float()
        alpha = torch.from_numpy(alpha).permute(2, 0, 1).contiguous().float()
        return image, alpha
    
    def __getitem__(self, index):


        azimuths = random.sample([i for i in range(-180, 180, 5)], self.input_view_num + self.target_view_num) #np.array([30, 90, 150, 210, 270, 330]).astype(float)
        elevations = random.sample([i for i in range(-20, 40, 1)], self.input_view_num + self.target_view_num) #np.array([20, -10, 20, -10, 20, -10]).astype(float)
        
        if 'CustomHumans' in self.root:
            azimuths = np.array([-150, -120, -90, -60, -30, 0, 30, 60, 90, 120])
            elevations = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0])

        azimuths = np.array(azimuths).astype(float)
        elevations = np.array(elevations).astype(float)

        radius = 2.0

        c2ws = spherical_camera_pose(azimuths, elevations, radius)

        K = FOV_to_intrinsics(self.fov)
        Ks = K.unsqueeze(0).repeat(self.input_view_num + self.target_view_num, 1, 1).float()

        # camera_mv_bx4x4 = torch.linalg.inv(c2ws)

        folder_path = self.paths[index]

        if '2K2K' in folder_path:
            for filename in os.listdir(folder_path):
                if filename.endswith(".obj"):
                    file_path = os.path.join(folder_path, filename)
                    mesh = trimesh.load(file_path) # 保存为OBJ文件 mesh.export('output.obj') 
                elif filename.endswith(".png"):
                    file_path = os.path.join(folder_path, filename)
                    tex = torch.from_numpy(cv2.cvtColor(cv2.imread(file_path), cv2.COLOR_BGR2RGB).astype(np.float32)/255.0)
                    
            mesh_v_nx3 = mesh.vertices
            mesh_v_nx3 = mesh_v_nx3 - mesh_v_nx3.mean(axis=0)
            large_ratio = 1/(mesh_v_nx3.max(axis=0) - mesh_v_nx3.min(axis=0))
            mesh_v_nx3 = mesh_v_nx3*large_ratio.min()
            mesh_v_nx3 = rotatedx(mesh_v_nx3)
            mesh_v_nx3 = rotatedx(mesh_v_nx3)

        elif 'THuman2.1' in folder_path: 
            for filename in os.listdir(folder_path):
                if filename.endswith("_rotated.obj"):
                    file_path = os.path.join(folder_path, filename)
                    mesh = trimesh.load(file_path) # 保存为OBJ文件 mesh.export('output.obj') 
                elif filename.endswith("_0.jpeg"):
                    file_path = os.path.join(folder_path, filename)
                    tex = torch.from_numpy(cv2.cvtColor(cv2.imread(file_path), cv2.COLOR_BGR2RGB).astype(np.float32)/255.0)
                    
            mesh_v_nx3 = mesh.vertices
            mesh_v_nx3 = mesh_v_nx3 - mesh_v_nx3.mean(axis=0)
            large_ratio = 1/(mesh_v_nx3.max(axis=0) - mesh_v_nx3.min(axis=0))
            mesh_v_nx3 = mesh_v_nx3*large_ratio.min()
            mesh_v_nx3 = rotatedx(mesh_v_nx3)
            mesh_v_nx3 = rotatedz(mesh_v_nx3)

        elif 'CustomHumans' in folder_path: 
            for filename in os.listdir(folder_path):
                if filename.endswith(".obj"):
                    file_path = os.path.join(folder_path, filename)
                    mesh = trimesh.load(file_path) # 保存为OBJ文件 mesh.export('output.obj') 
                elif filename.endswith(".png"):
                    file_path = os.path.join(folder_path, filename)
                    tex = torch.from_numpy(cv2.cvtColor(cv2.imread(file_path), cv2.COLOR_BGR2RGB).astype(np.float32)/255.0)
                    
            mesh_v_nx3 = mesh.vertices
            mesh_v_nx3 = mesh_v_nx3 - mesh_v_nx3.mean(axis=0)
            large_ratio = 1/(mesh_v_nx3.max(axis=0) - mesh_v_nx3.min(axis=0))
            mesh_v_nx3 = mesh_v_nx3*large_ratio.min()
            mesh_v_nx3 = rotatedx(mesh_v_nx3)
            mesh_v_nx3 = rotatedz(mesh_v_nx3)




        # indices = np.where(mesh_v_nx3[:,1]>0)
        # random_indices_rows = np.random.choice(indices[0].shape[0], size=50000, replace=False)
        # indices = (indices[0][random_indices_rows])

        # mesh_v_nx3 = mesh_v_nx3[indices]
        # mesh_v_nx3 = mesh_v_nx3 - mesh_v_nx3.mean(axis=0)


        # random_rows = np.random.choice(mesh.faces.shape[0], size=100000, replace=False)
        # mesh_faces = mesh.faces[random_rows]


        # new_faces = []
        # indices = torch.from_numpy(indices).int()
        # for row in mesh_faces:
        #     for r in row:
        #         if r in indices:
        #             new_faces.append(row)
        #             continue


        # new_faces = np.stack(new_faces)

        # offset = (mesh_v_nx3.max(axis=0) - mesh_v_nx3.min(axis=0))*0.25

        # mesh_v_nx3 = mesh_v_nx3 - offset
        # large_ratio = 1.2/(mesh_v_nx3.max(axis=0) - mesh_v_nx3.min(axis=0))
        # mesh_v_nx3 = mesh_v_nx3*large_ratio


        vtx_pos = torch.from_numpy(mesh_v_nx3).float()
        pos_idx = torch.from_numpy(mesh.faces).int()

        vtx_uv = torch.tensor(np.array(mesh.visual.uv), dtype=torch.float32)
        vtx_uv[:, 1] = 1 - vtx_uv[:, 1]

        uv_idx = pos_idx

        # mesh_v_nx3 = torch.from_numpy(mesh_v_nx3).float()
        # mesh_f_fx3 = torch.from_numpy(mesh_f_fx3).int()

        # 计算填充的数量
        # pad_width = 25000 - vtx_pos.shape[0]
        pad_width = vtx_pos.shape[0]

        # 使用numpy的pad函数进行填充
        padded_vtx_pos = np.pad(vtx_pos, ((0, pad_width), (0, 0)), 'constant', constant_values=0)
        padded_vtx_uv = np.pad(vtx_uv, ((0, pad_width), (0, 0)), 'constant', constant_values=0)

        # out = self.renderer.render_mesh(
        #             mesh_v_nx3.unsqueeze(dim=0),
        #             mesh_f_fx3,
        #             camera_mv_bx4x4,
        #             mesh_v_nx3.unsqueeze(dim=0),
        #             resolution=320,
        #             device='cpu',
        #             hierarchical_mask=False
        #         )

        data = {
            "v":padded_vtx_pos,
            "f":pos_idx,
            "vt":padded_vtx_uv,
            "ft":uv_idx,
            "v_len":np.array([vtx_pos.shape[0]]),
            'tex': tex,
            

            # 'input_images': images[:self.input_view_num],           # (6, 3, H, W)
            # 'input_alphas': alphas[:self.input_view_num],           # (6, 1, H, W) 
            # 'input_depths': depths[:self.input_view_num],           # (6, 1, H, W)
            # 'input_normals': normals[:self.input_view_num],         # (6, 3, H, W)
            'input_c2ws': c2ws[:self.input_view_num],               # (6, 4, 4)
            'input_Ks': Ks[:self.input_view_num],                   # (6, 3, 3)

            # lrm generator input and supervision
            # 'target_images': images[self.input_view_num:],          # (V, 3, H, W)
            # 'target_alphas': alphas[self.input_view_num:],          # (V, 1, H, W)
            # 'target_depths': depths[self.input_view_num:],          # (V, 1, H, W)
            # 'target_normals': normals[self.input_view_num:],        # (V, 3, H, W)
            'target_c2ws': c2ws[self.input_view_num:],              # (V, 4, 4)
            'target_Ks': Ks[self.input_view_num:],                  # (V, 3, 3)
        }
        return data


        # print("")
        while True:
            input_image_path = os.path.join(self.root_dir, self.input_image_dir, self.paths[index])
            target_image_path = os.path.join(self.root_dir, self.target_image_dir, self.paths[index])

            indices = np.random.choice(range(self.total_view_n), self.input_view_num + self.target_view_num, replace=False)
            input_indices = indices[:self.input_view_num]
            target_indices = indices[self.input_view_num:]

            '''background color, default: white'''
            bg_white = [1., 1., 1.]
            bg_black = [0., 0., 0.]

            image_list = []
            alpha_list = []
            depth_list = []
            normal_list = []
            pose_list = []

            try:
                input_cameras = np.load(os.path.join(input_image_path, 'cameras.npz'))['cam_poses']
                for idx in input_indices:
                    image, alpha = self.load_im(os.path.join(input_image_path, '%03d.png' % idx), bg_white)
                    normal, _ = self.load_im(os.path.join(input_image_path, '%03d_normal.png' % idx), bg_black)
                    depth = cv2.imread(os.path.join(input_image_path, '%03d_depth.png' % idx), cv2.IMREAD_UNCHANGED) / 255.0 * self.depth_scale
                    depth = torch.from_numpy(depth).unsqueeze(0)
                    pose = input_cameras[idx]
                    pose = np.concatenate([pose, np.array([[0, 0, 0, 1]])], axis=0)

                    image_list.append(image)
                    alpha_list.append(alpha)
                    depth_list.append(depth)
                    normal_list.append(normal)
                    pose_list.append(pose)

                target_cameras = np.load(os.path.join(target_image_path, 'cameras.npz'))['cam_poses']
                for idx in target_indices:
                    image, alpha = self.load_im(os.path.join(target_image_path, '%03d.png' % idx), bg_white)
                    normal, _ = self.load_im(os.path.join(target_image_path, '%03d_normal.png' % idx), bg_black)
                    depth = cv2.imread(os.path.join(target_image_path, '%03d_depth.png' % idx), cv2.IMREAD_UNCHANGED) / 255.0 * self.depth_scale
                    depth = torch.from_numpy(depth).unsqueeze(0)
                    pose = target_cameras[idx]
                    pose = np.concatenate([pose, np.array([[0, 0, 0, 1]])], axis=0)

                    image_list.append(image)
                    alpha_list.append(alpha)
                    depth_list.append(depth)
                    normal_list.append(normal)
                    pose_list.append(pose)

            except Exception as e:
                print(e)
                index = np.random.randint(0, len(self.paths))
                continue

            break
        
        images = torch.stack(image_list, dim=0).float()                 # (6+V, 3, H, W)
        alphas = torch.stack(alpha_list, dim=0).float()                 # (6+V, 1, H, W)
        depths = torch.stack(depth_list, dim=0).float()                 # (6+V, 1, H, W)
        normals = torch.stack(normal_list, dim=0).float()               # (6+V, 3, H, W)
        w2cs = torch.from_numpy(np.stack(pose_list, axis=0)).float()    # (6+V, 4, 4)
        c2ws = torch.linalg.inv(w2cs).float()

        normals = normals * 2.0 - 1.0
        normals = F.normalize(normals, dim=1)
        normals = (normals + 1.0) / 2.0
        normals = torch.lerp(torch.zeros_like(normals), normals, alphas)

        # random rotation along z axis
        if self.camera_rotation:
            degree = np.random.uniform(0, math.pi * 2)
            rot = torch.tensor([
                [np.cos(degree), -np.sin(degree), 0, 0],
                [np.sin(degree), np.cos(degree), 0, 0],
                [0, 0, 1, 0],
                [0, 0, 0, 1],
            ]).unsqueeze(0).float()
            c2ws = torch.matmul(rot, c2ws)

            # rotate normals
            N, _, H, W = normals.shape
            normals = normals * 2.0 - 1.0
            normals = torch.matmul(rot[:, :3, :3], normals.view(N, 3, -1)).view(N, 3, H, W)
            normals = F.normalize(normals, dim=1)
            normals = (normals + 1.0) / 2.0
            normals = torch.lerp(torch.zeros_like(normals), normals, alphas)

        # random scaling
        if np.random.rand() < 0.5:
            scale = np.random.uniform(0.7, 1.1)
            c2ws[:, :3, 3] *= scale
            depths *= scale

        # instrinsics of perspective cameras
        K = FOV_to_intrinsics(self.fov)
        Ks = K.unsqueeze(0).repeat(self.input_view_num + self.target_view_num, 1, 1).float()

        data = {
            'input_images': images[:self.input_view_num],           # (6, 3, H, W)
            'input_alphas': alphas[:self.input_view_num],           # (6, 1, H, W) 
            'input_depths': depths[:self.input_view_num],           # (6, 1, H, W)
            'input_normals': normals[:self.input_view_num],         # (6, 3, H, W)
            'input_c2ws': c2ws[:self.input_view_num],               # (6, 4, 4)
            'input_Ks': Ks[:self.input_view_num],                   # (6, 3, 3)

            # lrm generator input and supervision
            'target_images': images[self.input_view_num:],          # (V, 3, H, W)
            'target_alphas': alphas[self.input_view_num:],          # (V, 1, H, W)
            'target_depths': depths[self.input_view_num:],          # (V, 1, H, W)
            'target_normals': normals[self.input_view_num:],        # (V, 3, H, W)
            'target_c2ws': c2ws[self.input_view_num:],              # (V, 4, 4)
            'target_Ks': Ks[self.input_view_num:],                  # (V, 3, 3)
        }
        return data


class ValidationData(Dataset):
    def __init__(self,
        root_dir='objaverse/',
        input_view_num=6,
        input_image_size=320,
        fov=30,
    ):
        self.root_dir = Path(root_dir)
        self.input_view_num = input_view_num
        self.input_image_size = input_image_size
        self.fov = fov

        folders = [f.path for f in os.scandir(root_dir) if f.is_dir()]
        self.paths = sorted(folders)

        # self.paths = sorted(os.listdir(self.root_dir))
        print('============= length of dataset %d =============' % len(self.paths))

        cam_distance = 4.0
        azimuths = np.array([30, 90, 150, 210, 270, 330])
        elevations = np.array([20, -10, 20, -10, 20, -10])
        azimuths = np.deg2rad(azimuths)
        elevations = np.deg2rad(elevations)

        x = cam_distance * np.cos(elevations) * np.cos(azimuths)
        y = cam_distance * np.cos(elevations) * np.sin(azimuths)
        z = cam_distance * np.sin(elevations)

        cam_locations = np.stack([x, y, z], axis=-1)
        cam_locations = torch.from_numpy(cam_locations).float()
        c2ws = center_looking_at_camera_pose(cam_locations)
        self.c2ws = c2ws.float()
        self.Ks = FOV_to_intrinsics(self.fov).unsqueeze(0).repeat(6, 1, 1).float()

        render_c2ws = get_circular_camera_poses(M=8, radius=cam_distance, elevation=20.0)
        render_Ks = FOV_to_intrinsics(self.fov).unsqueeze(0).repeat(render_c2ws.shape[0], 1, 1)
        self.render_c2ws = render_c2ws.float()
        self.render_Ks = render_Ks.float()

    def __len__(self):
        return len(self.paths)

    def load_im(self, path, color):
        '''
        replace background pixel with random color in rendering
        '''
        pil_img = Image.open(path)
        pil_img = pil_img.resize((self.input_image_size, self.input_image_size), resample=Image.BICUBIC)

        image = np.asarray(pil_img, dtype=np.float32) / 255.
        if image.shape[-1] == 4:
            alpha = image[:, :, 3:]
            image = image[:, :, :3] * alpha + color * (1 - alpha)
        else:
            alpha = np.ones_like(image[:, :, :1])

        image = torch.from_numpy(image).permute(2, 0, 1).contiguous().float()
        alpha = torch.from_numpy(alpha).permute(2, 0, 1).contiguous().float()
        return image, alpha
    
    def __getitem__(self, index):
        # load data
        input_image_path = os.path.join(self.root_dir, self.paths[index])

        '''background color, default: white'''
        bkg_color = [1.0, 1.0, 1.0]

        image_list = []
        alpha_list = []

        for idx in range(self.input_view_num):
            image, alpha = self.load_im(os.path.join(input_image_path, f'{idx:03d}.png'), bkg_color)
            image_list.append(image)
            alpha_list.append(alpha)
        
        images = torch.stack(image_list, dim=0).float()
        alphas = torch.stack(alpha_list, dim=0).float()

        data = {
            'input_images': images,
            'input_alphas': alphas,
            'input_c2ws': self.c2ws,
            'input_Ks': self.Ks,

            'render_c2ws': self.render_c2ws,
            'render_Ks': self.render_Ks,
        }
        return data
