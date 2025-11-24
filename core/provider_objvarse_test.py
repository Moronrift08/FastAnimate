import os
import cv2
import random
import numpy as np
import tyro
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from torch.utils.data import Dataset

import kiui

import sys
sys.path.append('/hpc2hdd/home/jshu704/Animation/LBS_LGM_nanjie')

import core

from core.options import Options
IMAGENET_DEFAULT_MEAN = (0.485, 0.456, 0.406)
IMAGENET_DEFAULT_STD = (0.229, 0.224, 0.225)
import PIL.Image
import pyspng
import json
from kiui.cam import orbit_camera
import pickle
import numpy as np
from mvdream.pipeline_mvdream import MVDreamPipeline
import rembg
import openmesh as om
import math
import cv2

from vision.ssd.mb_tiny_fd import create_mb_tiny_fd, create_mb_tiny_fd_predictor
from vision.ssd.config.fd_config import define_img_size
define_img_size(640)  # must put define_img_size() before 'import create_mb_tiny_fd, create_mb_tiny_fd_predictor'
import torchvision

from pathlib import Path
import trimesh

from src.utils.camera_util import (
    FOV_to_intrinsics, 
    center_looking_at_camera_pose, 
    get_circular_camera_poses,
)
# Thuman
# from core.lib_thuman.dataset.mesh_util import SMPLX
# from core.lib_thuman.common.render_utils import face_vertices
# import core.lib_thuman.smplx as smplx
from scipy.spatial import cKDTree

# Xhuman
from core.lib.dataset.mesh_util import SMPLX
from core.lib.common.render_utils import face_vertices
import core.lib.smplx as smplx


def rotatedx(original_vertices, angle=90):
    # 原始的vertices
    vertices = original_vertices

    # 定义旋转矩阵
    theta = np.radians(angle)
    rotation_matrix = np.array([[1, 0, 0],
                                [0, np.cos(theta), -np.sin(theta)],
                                [0, np.sin(theta), np.cos(theta)]])

    # 对vertices进行旋转操作
    rotated_vertices = np.dot(rotation_matrix, vertices.T).T

    return rotated_vertices

def rotatedy(original_vertices, angle=90):
    vertices = original_vertices

    # 定义旋转矩阵
    theta = np.radians(angle)
    rotation_matrix = np.array([[np.cos(theta), 0, np.sin(theta)],
                                [0, 1, 0],
                                [-np.sin(theta), 0, np.cos(theta)]])

    # 对vertices进行旋转操作
    rotated_vertices = np.dot(rotation_matrix, vertices.T).T

    return rotated_vertices

def rotatedz(original_vertices, angle=90):
        # 原始的vertices
    vertices = original_vertices

    # 定义旋转矩阵
    theta = np.radians(angle)
    rotation_matrix = np.array([[np.cos(theta), -np.sin(theta), 0],
                                [np.sin(theta), np.cos(theta), 0],
                                [0, 0, 1]])
    # 对vertices进行旋转操作
    rotated_vertices = np.dot(rotation_matrix, vertices.T).T

    return rotated_vertices


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

def new_mesh(mesh_ori, ratio):

    try:
        bb_max = mesh_ori.vertices.max(axis=0)
        bb_min = mesh_ori.vertices.min(axis=0)
    except:
        mesh_ori = mesh_ori.geometry[[i for i in mesh_ori.geometry][0]]
        bb_max = mesh_ori.vertices.max(axis=0)
        bb_min = mesh_ori.vertices.min(axis=0)
    centers = (
        (bb_min[0] + bb_max[0]) / 2,
        (bb_min[1] + bb_max[1]) / 2,
        (bb_min[2] + bb_max[2]) / 2
    )
    total_size = (bb_max - bb_min).max()
    scale = total_size / (0.5 + ratio)
    translation = (
        -centers[0],
        -centers[1],
        -centers[2]
    )
    scales_inv = (
        2/scale, 2/scale, 2/scale
    )
    mesh_ori.vertices = mesh_ori.vertices+translation
    mesh_ori.vertices = mesh_ori.vertices*scales_inv

    return mesh_ori, translation, scales_inv


def fetchPly_rotate_mesh(positions, data2):
    # angel_x = 90  # 旋转角度
    radian = data2[0] #angel_x * np.pi / 180  # 旋转弧度
    Rotation_Matrix_1 = [  # 绕x轴三维旋转矩阵 
        [1, 0, 0],
        [0, math.cos(radian), -math.sin(radian)],
        [0, math.sin(radian), math.cos(radian)]]
    
    Rotation_Matrix_1 = np.array(Rotation_Matrix_1)


    # angel_y = -90  # 旋转角度
    radian = -1*(data2[1]+np.pi/2) #angel_y * np.pi / 180  # 旋转弧度
    Rotation_Matrix_2 = [  # 绕y轴三维旋转矩阵
        [math.cos(radian), 0, math.sin(radian)],
        [0, 1, 0],
        [-math.sin(radian), 0, math.cos(radian)]]
    
    Rotation_Matrix_2 = np.array(Rotation_Matrix_2)

    # # 构造旋转矩阵
    # rotation_axis = np.cross(old_global_orient, new_global_orient)
    # rotation_angle = np.arccos(np.dot(old_global_orient, new_global_orient.T))
    # rotation_matrix = axangle2mat(rotation_axis[0], rotation_angle)

    p = np.dot(Rotation_Matrix_2, positions.T) # 计算
    p = np.dot(Rotation_Matrix_1, p) # 计算
    positions = p.T
    positions = positions.astype(np.float32)


    return positions


class ObjaverseDataset(Dataset):

    def _warn(self):
        raise NotImplementedError('this dataset is just an example and cannot be used directly, you should modify it to your own setting! (search keyword TODO)')

    def __init__(self, opt: Options, training=True):

        self.opt = opt
        self.training = training
        self.items = []

        # Thuman
        # root_path = '/hpc2hdd/home/jshu704/Animation/data/data_thuman_demo/Thuman2/origin'

        # Xhuman
        root_path = '/hpc2hdd/home/jshu704/Animation/data/data_fig3'

        root_dirs = sorted(os.listdir(root_path))

        # Xhuman
        for root_dir in root_dirs:
            origin_path = os.path.join(root_path, root_dir, 'origin')
            origin_scan_path = os.path.join(origin_path, 'scan')
            if os.path.isdir(origin_scan_path):
                self.items.append({
                    'origin_scan': origin_scan_path
                })

        # Thuman
        # for root_dir in root_dirs:
        #     origin_path = os.path.join(root_path, root_dir)
        #     origin_scan_path = os.path.join(origin_path, 'scan')
        #     if os.path.isdir(origin_scan_path):
        #         self.items.append({
        #             'origin_scan': origin_scan_path
        #         })

        model_init_params = dict(
            gender='male',
            model_type='smplx',
            model_path=SMPLX().model_dir,
            create_global_orient=False,
            create_body_pose=False,
            create_betas=False,
            create_left_hand_pose=False,
            create_right_hand_pose=False,
            create_expression=False,
            create_jaw_pose=False,
            create_leye_pose=False,
            create_reye_pose=False,
            create_transl=False,
            num_pca_comps=12
        )

        def get_smpl_model(model_type, gender):
            return smplx.create(**model_init_params)

        smpl_type='smplx'
        smpl_gender='male'

        self.smpl_model_old = get_smpl_model(smpl_type, smpl_gender)

    def prepare_default_rays(self, input_size, view_num=6, device='cpu', elevation=0):
        
        from kiui.cam import orbit_camera
        from core.utils import get_rays
        
        cam_poses = np.stack([
            orbit_camera(elevation, 0, radius=self.opt.cam_radius),
            orbit_camera(elevation, 180, radius=self.opt.cam_radius),
            orbit_camera(elevation, 90, radius=self.opt.cam_radius),
            orbit_camera(elevation, 270, radius=self.opt.cam_radius),
            orbit_camera(elevation, 45, radius=self.opt.cam_radius),
            orbit_camera(elevation, 315, radius=self.opt.cam_radius),

        ], axis=0) # [4, 4, 4]
        cam_poses = cam_poses[0:view_num]
        
        cam_poses = torch.from_numpy(cam_poses)

        rays_embeddings = []
        for i in range(cam_poses.shape[0]):
            rays_o, rays_d = get_rays(cam_poses[i], input_size, input_size, self.opt.fovy) # [h, w, 3]
            rays_plucker = torch.cat([torch.cross(rays_o, rays_d, dim=-1), rays_d], dim=-1) # [h, w, 6]
            rays_embeddings.append(rays_plucker)

        rays_embeddings = torch.stack(rays_embeddings, dim=0).permute(0, 3, 1, 2).contiguous().to(device) # [V, 6, h, w]
        
        return rays_embeddings


    def _open_file(self, fname):
        if self._type == 'dir':
            return open(os.path.join(self._path, fname), 'rb')
        if self._type == 'zip':
            return self._get_zipfile().open(fname, 'r')
        return None

    @staticmethod
    def _file_ext(fname):
        return os.path.splitext(fname)[1].lower()

    def _load_raw_image(self, raw_idx):
        fname = self._image_fnames[raw_idx]
        with self._open_file(fname) as f:
            # if pyspng is not None and self._file_ext(fname) == '.png':
            if self._file_ext(fname) == '.png':
                image = pyspng.load(f.read())
            else:
                image = np.array(PIL.Image.open(f))

            assert image.shape[-1]==4
            image = image[..., :3] * (image[..., -1:] == 255) + (255. - image[..., -1:])
            image = PIL.Image.fromarray(image.astype('uint8'), 'RGB')
            image = image.resize((self.img_size, self.img_size))
            image = np.array(image)
        if image.ndim == 2:
            image = image[:, :, np.newaxis] # HW => HWC
        image = image.transpose(2, 0, 1) # HWC => CHW
        return image
    
    def __len__(self):
        return len(self.items)

    def _load_raw_labels(self):
        fname = 'extrinsics_smpl.json'

        if fname not in self._all_fnames:
            return None
        with self._open_file(fname) as f:
            labels = json.load(f)['labels']
        if labels is None:
            return None
        labels = dict(labels)
        labels = [labels[fname.replace('\\', '/')] for fname in self._image_fnames]
        labels = np.array(labels)
        labels = labels.astype({1: np.int64, 2: np.float32}[labels.ndim])
        return labels
    
    def _get_raw_labels(self):
        if self._raw_labels is None:
            self._raw_labels = self._load_raw_labels() if self._use_labels else None
            if self._raw_labels is None:
                self._raw_labels = np.zeros([self._raw_shape[0], 0], dtype=np.float32)
            assert isinstance(self._raw_labels, np.ndarray)
            assert self._raw_labels.shape[0] == self._raw_shape[0]
            assert self._raw_labels.dtype in [np.float32, np.int64]
            if self._raw_labels.dtype == np.int64:
                assert self._raw_labels.ndim == 1
                assert np.all(self._raw_labels >= 0)
        return self._raw_labels

    def get_label(self, idx):
        label = self._get_raw_labels()[self._raw_idx[idx]]
        if label.dtype == np.int64:
            onehot = np.zeros(self.label_shape, dtype=np.float32)
            onehot[label] = 1
            label = onehot
        return label.copy()

    def _load_image_and_mask(self, raw_idx):
        fname = self._image_fnames[raw_idx]

        with self._open_file(fname) as f:
            if self._file_ext(fname) == '.png':
                ori_img = pyspng.load(f.read())
            else:
                ori_img = np.array(PIL.Image.open(f))

        assert ori_img.shape[-1] == 4
        img = ori_img[:, :, :3]
        mask = ori_img[:, :, 3:4]

        image = PIL.Image.fromarray(img.astype('uint8'), 'RGB')
        resize_img = np.array(image.resize((self.img_size, self.img_size)))
        mask = cv2.resize(mask, (self.img_size, self.img_size), interpolation=cv2.INTER_NEAREST)  ########

        img = resize_img.transpose(2, 0, 1)
        #background = np.zeros_like(img)
        if self.white_bg:
            background = np.ones_like(img) * 255
        else:
            background = np.zeros_like(img)

        img = img * (mask > 0).astype(np.float) + background * (1 - (mask > 0).astype(np.float))
        return np.ascontiguousarray(img),  np.ascontiguousarray(mask)

    def det_face(self, img, img_path):
        image = cv2.cvtColor(img[:,:,:3], cv2.COLOR_BGR2RGB)
        boxes, labels, probs = self.predictor.predict(image, 2000 / 2, 0.96)

        if probs.shape[0] == 1:

            try:
                boxes[0][0]= boxes[0][0]-10
                boxes[0][1]= boxes[0][1]-10
                boxes[0][2]= boxes[0][2]+10
                boxes[0][3]= boxes[0][3]+10

                boxes = torch.clamp(boxes, min=0, max=1023)
                faces = torch.cat([torch.ones([1,1])*2,boxes], dim=-1)#.int().numpy()
                # cv2.imwrite('./gray.png',image)
                # cv2.imwrite('./gray_head.png',image[faces[0][1]:faces[0][3], faces[0][0]:faces[0][2]])
            except:
                print('wrong')
                faces = torch.zeros([1,5])#.int().numpy()

        elif probs.shape[0] == 0:
            faces = torch.zeros([1,5])#.int().numpy()
        else:

            print("multi")

            try:
                boxes[0][0]= boxes[0][0]-10
                boxes[0][1]= boxes[0][1]-10
                boxes[0][2]= boxes[0][2]+10
                boxes[0][3]= boxes[0][3]+10

                boxes = torch.clamp(boxes, min=0, max=1023)
                faces = torch.cat([torch.ones([1,1])*2,boxes], dim=-1)#.int().numpy()
                # cv2.imwrite('./gray.png',image)
                # cv2.imwrite('./gray_head.png',image[faces[0][1]:faces[0][3], faces[0][0]:faces[0][2]])
            except:
                print('wrong')
                faces = torch.zeros([1,5])#.int().numpy()

        return faces



    def __getitem__(self, idx):
         # 读入到scan目录 /hpc2hdd/home/jshu704/Animation/data/data_test_pair/000/origin/scan
        scan_path = self.items[idx]['origin_scan']
        gt_path = scan_path.replace('origin', 'gt')
        A_path = scan_path.replace('origin', 'A')

        # Thuman
        # gt_path = scan_path
        # A_path = scan_path


        # all_files = os.listdir(scan_path)
        # for f in all_files:
        #     if f.endswith('.obj'):
        #         origin_scan_path = os.path.join(scan_path, f)
        #     # Thuman
        #     elif f.endswith('jpeg'):
        #         origin_jpg_path = os.path.join(scan_path, f)

        # Thuman

        # origin_jpg_path = origin_scan_path.replace('mesh-', 'atlas-').replace('.obj', '.jpg')

        # temp_smpl_ply_path = os.path.dirname(origin_scan_path.replace('scan', 'smpl'))
        # origin_smpl_ply_path = os.path.join(temp_smpl_ply_path, 'mesh_smplx.obj')
        # origin_smpl_pkl_path = origin_smpl_ply_path.replace('mesh_smplx.obj', 'smplx_param.pkl')
        
        # gt_scan_path = origin_scan_path
        # gt_jpg_path = origin_jpg_path
        # # gt_smpl_ply_path = origin_smpl_ply_path
        # # gt_smpl_pkl_path = origin_smpl_pkl_path
        # gt_smpl_ply_path = '/hpc2hdd/home/jshu704/Animation/data/data_thuman_demo/THuman_target_pose/001/mesh_smplx.obj'
        # gt_smpl_pkl_path = '/hpc2hdd/home/jshu704/Animation/data/data_thuman_demo/THuman_target_pose/001/smplx_param.pkl'

        # A_scan_path = origin_scan_path
        # A_jpg_path = origin_jpg_path
        # # A_smpl_ply_path = origin_smpl_ply_path
        # # A_smpl_pkl_path = origin_smpl_pkl_path
        # A_smpl_ply_path = '/hpc2hdd/home/jshu704/Animation/data/data_thuman_demo/THuman_target_pose/001/mesh_smplx.obj'
        # A_smpl_pkl_path = '/hpc2hdd/home/jshu704/Animation/data/data_thuman_demo/THuman_target_pose/001/smplx_param.pkl'


        # # origin_scan = trimesh.load(origin_scan_path)
        # origin_tex = torch.from_numpy(cv2.cvtColor(cv2.imread(origin_jpg_path), cv2.COLOR_BGR2RGB).astype(np.float32)/255.0)
        # origin_smpl = trimesh.load(origin_smpl_ply_path, force='mesh')
        
        # A_gt_scan = trimesh.load(A_scan_path)
        # A_gt_tex = torch.from_numpy(cv2.cvtColor(cv2.imread(A_jpg_path), cv2.COLOR_BGR2RGB).astype(np.float32)/255.0)
        # A_origin_smpl = trimesh.load(A_smpl_ply_path, force='mesh')

        # Posed_gt_scan = trimesh.load(gt_scan_path)
        # posed_gt_tex = torch.from_numpy(cv2.cvtColor(cv2.imread(gt_jpg_path), cv2.COLOR_BGR2RGB).astype(np.float32)/255.0)
        # target_smpl = trimesh.load(gt_smpl_ply_path, force='mesh')

        # A_scan = trimesh.load(A_scan_path)
        # A_tex = torch.from_numpy(cv2.cvtColor(cv2.imread(A_jpg_path), cv2.COLOR_BGR2RGB).astype(np.float32)/255.0)
        # A_smpl = trimesh.load(A_smpl_ply_path, force='mesh')


        # X-human

        all_files = os.listdir(scan_path)
        for f in all_files:
            if f.endswith('.obj'):
                origin_scan_path = os.path.join(scan_path, f)

        origin_jpg_path = origin_scan_path.replace('mesh-', 'atlas-').replace('.obj', '.jpg')
        
        origin_smpl_ply_path = origin_scan_path.replace('scan', 'smpl').replace('.obj', '_smplx.ply')
        origin_smpl_pkl_path = origin_scan_path.replace('scan', 'smpl').replace('.obj', '_smplx.pkl')


        all_files = os.listdir(gt_path)
        for f in all_files:
            if f.endswith('.obj'):
                gt_scan_path = os.path.join(gt_path, f)

        gt_jpg_path = gt_scan_path.replace('mesh-', 'atlas-').replace('.obj', '.jpg')
        gt_smpl_ply_path = gt_scan_path.replace('scan', 'smpl').replace('.obj', '_smplx.ply')
        gt_smpl_pkl_path = gt_scan_path.replace('scan', 'smpl').replace('.obj', '_smplx.pkl')

        all_files = os.listdir(A_path)
        for f in all_files:
            if f.endswith('.obj'):
                A_scan_path = os.path.join(A_path, f)

        A_jpg_path = A_scan_path.replace('mesh-', 'atlas-').replace('.obj', '.jpg')
        A_smpl_ply_path = A_scan_path.replace('scan', 'smpl').replace('.obj', '_smplx.ply')
        A_smpl_pkl_path = A_scan_path.replace('scan', 'smpl').replace('.obj', '_smplx.pkl')


        # origin_scan = trimesh.load(origin_scan_path)
        origin_tex = torch.from_numpy(cv2.cvtColor(cv2.imread(origin_jpg_path), cv2.COLOR_BGR2RGB).astype(np.float32)/255.0)
        origin_smpl = trimesh.load(origin_smpl_ply_path, force='mesh')
        
        A_gt_scan = trimesh.load(A_scan_path)
        A_gt_tex = torch.from_numpy(cv2.cvtColor(cv2.imread(A_jpg_path), cv2.COLOR_BGR2RGB).astype(np.float32)/255.0)
        A_origin_smpl = trimesh.load(A_smpl_ply_path, force='mesh')

        Posed_gt_scan = trimesh.load(gt_scan_path)
        posed_gt_tex = torch.from_numpy(cv2.cvtColor(cv2.imread(gt_jpg_path), cv2.COLOR_BGR2RGB).astype(np.float32)/255.0)
        target_smpl = trimesh.load(gt_smpl_ply_path, force='mesh')

        A_scan = trimesh.load(A_scan_path)
        A_tex = torch.from_numpy(cv2.cvtColor(cv2.imread(A_jpg_path), cv2.COLOR_BGR2RGB).astype(np.float32)/255.0)
        A_smpl = trimesh.load(A_smpl_ply_path, force='mesh')

        # Load the Apose and target pose SMPL pkl for animation between Stage1 and Stage2 (Nanjie)
        fitted_path = A_smpl_pkl_path
        param_apose = np.load(fitted_path, allow_pickle=True)
        for key in param_apose.keys():
            param_apose[key] = torch.as_tensor(param_apose[key])

        fitted_path = gt_smpl_pkl_path
        param_targetpose = np.load(fitted_path, allow_pickle=True)
        for key in param_targetpose.keys():
            param_targetpose[key] = torch.as_tensor(param_targetpose[key])

        ####################LBS####################
        ###########################################
        target_path = A_smpl_pkl_path
        target_param = np.load(target_path, allow_pickle=True)
        for key in target_param.keys():
            target_param[key] = torch.as_tensor(target_param[key])

        fitted_path = origin_smpl_pkl_path
        param = np.load(fitted_path, allow_pickle=True)
        for key in param.keys():
            param[key] = torch.as_tensor(param[key])

        obj_file_path = origin_scan_path
        origin_scan = trimesh.load(obj_file_path)

        jpg_file_path = origin_jpg_path
        tex = torch.from_numpy(cv2.cvtColor(cv2.imread(jpg_file_path), cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0)

        transl = param['transl']
        origin_scan.vertices = origin_scan.vertices - transl.numpy()


        # Thuman2
        # scale = param['scale']
        # translation = param['translation']
        # origin_scan.vertices = (origin_scan.vertices - translation.numpy())/scale.numpy()

        model_forward_params = dict(
            betas=param['betas'],
            global_orient=param['global_orient'],
            body_pose=param['body_pose'],
            left_hand_pose=param['left_hand_pose'],
            right_hand_pose=param['right_hand_pose'],
            jaw_pose=param['jaw_pose'],
            leye_pose=param['leye_pose'],
            reye_pose=param['reye_pose'],
            expression=param['expression'],
            return_verts=True,
            return_joint_transformation=True,
            return_vertex_transformation=True,
        )
        smpl_out_tpose = self.smpl_model_old(**model_forward_params)
        
        trimesh.Trimesh(smpl_out_tpose.vertices[0], self.smpl_model_old.faces)
        econ_obj = origin_scan

        smpl_verts = smpl_out_tpose.vertices.detach()[0]
        smpl_tree = cKDTree(smpl_verts.cpu().numpy())
        dist, idxx = smpl_tree.query(econ_obj.vertices, k=3)

        # t-pose for ECON
        econ_verts = torch.tensor(econ_obj.vertices).float()
        rot_mat_t = smpl_out_tpose.vertex_transformation.detach()[0][idxx[:, 0]]
        homo_coord = torch.ones_like(econ_verts)[..., :1]
        econ_cano_verts = torch.inverse(rot_mat_t) @ torch.cat([econ_verts, homo_coord],
                                                    dim=1).unsqueeze(-1)
        econ_cano_verts = econ_cano_verts[:, :3, 0].cpu()

        model_forward_params_re= dict(
            betas=target_param['betas'],
            global_orient=target_param['global_orient'],
            body_pose=target_param['body_pose'],
            left_hand_pose=target_param['left_hand_pose'],
            right_hand_pose=target_param['right_hand_pose'],
            jaw_pose=target_param['jaw_pose'],
            leye_pose=target_param['leye_pose'],
            reye_pose=target_param['reye_pose'],
            expression=target_param['expression'],
            return_verts=True,
            return_joint_transformation=True,
            return_vertex_transformation=True,
        )
        smpl_out_repose = self.smpl_model_old(**model_forward_params_re)


        A_origin_smpl = trimesh.Trimesh(smpl_out_repose.vertices[0], self.smpl_model_old.faces) # .export(path+'/cur.obj')

        # da-pose for ECON
        rot_mat_da = smpl_out_repose.vertex_transformation.detach()[0][idxx[:, 0]]
        econ_da_verts = rot_mat_da @ torch.cat([econ_cano_verts, homo_coord], dim=1).unsqueeze(-1)
        econ_da_verts = econ_da_verts[:, :3, 0].cpu()

        origin_scan.vertices = econ_da_verts

        # A_out_path = os.path.join('/hpc2hdd/home/jshu704/Animation/LBS_LGM_nanjie/workspace_demo', f'{idx:03d}')
        # origin_scan.export(A_out_path+'/A_scan.obj')
        
        origin_scan.vertices = rotatedx(origin_scan.vertices)
        origin_scan.vertices = rotatedz(origin_scan.vertices)
        A_gt_scan.vertices = rotatedx(A_gt_scan.vertices)
        A_gt_scan.vertices = rotatedz(A_gt_scan.vertices)
        A_origin_smpl.vertices = rotatedx(A_origin_smpl.vertices)
        A_origin_smpl.vertices = rotatedz(A_origin_smpl.vertices)

        # Nanjie
        Posed_gt_scan.vertices = rotatedx(Posed_gt_scan.vertices)
        Posed_gt_scan.vertices = rotatedz(Posed_gt_scan.vertices)

        target_smpl.vertices = rotatedx(target_smpl.vertices)
        target_smpl.vertices = rotatedz(target_smpl.vertices)

        ratio = random.uniform(0, 0.1)
        origin_scan, transl_scan, scale_scan = new_mesh(origin_scan, ratio)
        A_origin_smpl, _, _ = new_mesh(A_origin_smpl, ratio)
        A_gt_scan, _, _ = new_mesh(A_gt_scan, ratio)
        #Nanjie - Normalize
        Posed_gt_scan, _, _ = new_mesh(Posed_gt_scan, ratio)
        target_smpl,_,_ = new_mesh(target_smpl, ratio)

        origin_vtx_pos = torch.from_numpy(origin_scan.vertices).float()
        origin_pos_idx = torch.from_numpy(origin_scan.faces).int()
        gt_vtx_pos = torch.from_numpy(A_gt_scan.vertices).float()
        gt_pos_idx = torch.from_numpy(A_gt_scan.faces).int()
        #Nanjie - vertices & faces
        posed_gt_vtx_pos = torch.from_numpy(Posed_gt_scan.vertices).float()
        posed_gt_pos_idx = torch.from_numpy(Posed_gt_scan.faces).int()

        origin_vtx_uv = torch.tensor(np.array(origin_scan.visual.uv), dtype=torch.float32)
        origin_vtx_uv[:, 1] = 1 - origin_vtx_uv[:, 1]
        gt_vtx_uv = torch.tensor(np.array(A_gt_scan.visual.uv), dtype=torch.float32)
        gt_vtx_uv[:, 1] = 1 - gt_vtx_uv[:, 1]
        #Nanjie - uv
        posed_gt_vtx_uv = torch.tensor(np.array(Posed_gt_scan.visual.uv), dtype=torch.float32)
        posed_gt_vtx_uv[:, 1] = 1 - posed_gt_vtx_uv[:, 1]

        origin_uv_idx = origin_pos_idx
        gt_uv_idx = gt_pos_idx
        #Nanjie - uv
        posed_gt_uv_idx = posed_gt_pos_idx

        origin_pad_width = origin_vtx_pos.shape[0]
        gt_pad_width = gt_vtx_pos.shape[0]
        #Nanjie - padding length
        posed_gt_pad_width = posed_gt_vtx_pos.shape[0]

        origin_padded_vtx_pos = np.pad(origin_vtx_pos, ((0, origin_pad_width), (0, 0)), 'constant', constant_values=0)
        origin_padded_vtx_uv = np.pad(origin_vtx_uv, ((0, origin_pad_width), (0, 0)), 'constant', constant_values=0)
        gt_padded_vtx_pos = np.pad(gt_vtx_pos, ((0, gt_pad_width), (0, 0)), 'constant', constant_values=0)
        gt_padded_vtx_uv = np.pad(gt_vtx_uv, ((0, gt_pad_width), (0, 0)), 'constant', constant_values=0)
        #Nanjie - padd vtx & pad uv
        posed_gt_padded_vtx_pos = np.pad(posed_gt_vtx_pos, ((0, posed_gt_pad_width), (0, 0)), 'constant', constant_values=0)
        posed_gt_padded_vtx_uv = np.pad(posed_gt_vtx_uv, ((0, posed_gt_pad_width), (0, 0)), 'constant', constant_values=0)

        azimuths = random.sample([i for i in np.arange(0, 360, 0.01)], 6) 
        elevations = random.sample([i for i in np.arange(-20, 20, 0.01)], 6) 
        first = 0
        azimuths = np.concatenate([np.array([first, (first+180)%360, (first+90)%360, (first+270)%360, (first+45)%360, (first+315)%360]), np.array(azimuths)], axis=0)
        elevations = np.concatenate([np.array([0, 0, 0, 0, 0, 0]), np.array(elevations)], axis=0)

        azimuths = np.array(azimuths).astype(float)
        elevations = np.array(elevations).astype(float)

        radius = [1.5 for i in range(1 + 11)]
        radius = np.array(radius).astype(float)

        poses = []
        for i in range(1 + 11):
            pose = orbit_camera(elevations[i], azimuths[i], radius[i])
            poses.append(pose)

        poses = np.array(poses).astype(float)

        c2ws = spherical_camera_pose(azimuths, elevations, radius)

        data = {
            "scan_v":origin_padded_vtx_pos,
            "scan_f":origin_pos_idx,
            "gt_v":gt_padded_vtx_pos,
            "gt_f":gt_pos_idx,
            "scan_vt":origin_padded_vtx_uv,
            "scan_ft":origin_uv_idx,
            "gt_vt":gt_padded_vtx_uv,
            "gt_ft":gt_uv_idx,
            "origin_smpl_v":torch.from_numpy(A_origin_smpl.vertices).float(),
            "origin_smpl_f":torch.from_numpy(A_origin_smpl.faces).int(),
            "origin_tex":origin_tex,
            "gt_tex":A_gt_tex,
            "origin_v_len":np.array([origin_vtx_pos.shape[0]]),
            "gt_v_len":np.array([gt_vtx_pos.shape[0]]),
            'input_c2ws_gs': poses[:1],               # (6, 4, 4)
            'target_c2ws_gs': poses[1:],               # (6, 4, 4)
            'input_c2ws': c2ws[:1],               # (6, 4, 4)
            'target_c2ws': c2ws[1:],              # (V, 4, 4)

            # Nanjie 
            'param_apose' : param_apose, # Apose parameter
            'param_targetpose': param_targetpose, # Targetpose parameter
            # Attributes from posed scan (used in stage2)
            'posed_scan_v': posed_gt_padded_vtx_pos,
            'posed_scan_f': posed_gt_pos_idx,
            'posed_scan_vt': posed_gt_padded_vtx_uv,
            'posed_scan_ft': posed_gt_uv_idx,
            'posed_v_len': np.array([posed_gt_vtx_pos.shape[0]]),
            'posed_gt_tex': posed_gt_tex,

            "target_smpl_v":torch.from_numpy(target_smpl.vertices).float(),
            "target_smpl_f":torch.from_numpy(target_smpl.faces).int(),
            "ratio": ratio
        }
        return data


    # Scan normalize -> nomralized gaussian

    # pred_smpl <-> nomralized gaussian