import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

import kiui
from kiui.lpips import LPIPS

from core.unet import UNet
from core.options import Options
from core.gs import GaussianRenderer
from mvdream.pipeline_mvdream import MVDreamPipeline
import cv2
import copy
from core.visual_encoder.model_utils import GaussianUpsampler, unproject_depth
import trimesh
import kaolin as kal
import torchvision
from vision.ssd.mb_tiny_fd import create_mb_tiny_fd, create_mb_tiny_fd_predictor
import random
from torchmetrics.image import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity

from src.models.geometry.camera.perspective_camera import PerspectiveCamera
from src.models.geometry.render.neural_render import NeuralRender

import torchvision.transforms.functional as TF
IMAGENET_DEFAULT_MEAN = (0.485, 0.456, 0.406)
IMAGENET_DEFAULT_STD = (0.229, 0.224, 0.225)
from core.utils import get_rays, grid_distortion, orbit_camera_jitter
import pickle
import os
import tinyobjloader
from core.mesh import *

from kaolin.ops.mesh import check_sign
from kaolin import _C

# NOTE: Nanjie
from scipy.spatial import cKDTree
from core.lib.dataset.mesh_util import SMPLX


def new_verts(verts, ratio):
    # Compute bounding box max and min
    bb_max = verts.max(dim=0).values  # PyTorch equivalent of numpy max(axis=0)
    bb_min = verts.min(dim=0).values  # PyTorch equivalent of numpy min(axis=0)
    
    # Compute centers
    centers = (
        (bb_min[0] + bb_max[0]) / 2,
        (bb_min[1] + bb_max[1]) / 2,
        (bb_min[2] + bb_max[2]) / 2
    )
    
    # Compute total size and scale
    total_size = (bb_max - bb_min).max().item()
    scale = total_size / (0.5 + ratio)
    
    # Compute translation
    translation = torch.tensor([
        -centers[0],
        -centers[1],
        -centers[2]
    ], device=verts.device).float()
    
    # Compute inverse scale
    scales_inv = torch.tensor([
        2 / scale, 2 / scale, 2 / scale
    ], device=verts.device).float()
    
    # Apply translation and scaling
    verts = verts + translation
    verts = verts * scales_inv

    return verts, translation, scales_inv

class MLP(nn.Module):
    """Super basic but super useful MLP class.
    """
    def __init__(self, 
        input_dim, 
        output_dim, 
        activation = torch.relu,
        bias = True,
        layer = nn.Linear,
        num_layers = 4, 
        hidden_dim = 128, 
        skip       = [2]
    ):
        """Initialize the MLP.

        Args:
            input_dim (int): Input dimension of the MLP.
            output_dim (int): Output dimension of the MLP.
            activation (function): The activation function to use.
            bias (bool): If True, use bias.
            layer (nn.Module): The MLP layer module to use.
            num_layers (int): The number of hidden layers in the MLP.
            hidden_dim (int): The hidden dimension of the MLP.
            skip (List[int]): List of layer indices where the input dimension is concatenated.

        Returns:
            (void): Initializes the class.
        """
        super().__init__()
        
        self.input_dim = input_dim
        self.output_dim = output_dim        
        self.activation = activation
        self.bias = bias
        self.layer = layer
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim
        self.skip = skip
        if self.skip is None:
            self.skip = []
        
        self.make()

    def make(self):
        """Builds the actual MLP.
        """
        layers = []
        for i in range(self.num_layers):
            if i == 0: 
                layers.append(self.layer(self.input_dim, self.hidden_dim, bias=self.bias))
            elif i in self.skip:
                layers.append(self.layer(self.hidden_dim+self.input_dim, self.hidden_dim, bias=self.bias))
            else:
                layers.append(self.layer(self.hidden_dim, self.hidden_dim, bias=self.bias))
        self.layers = nn.ModuleList(layers)
        self.lout = self.layer(self.hidden_dim, self.output_dim, bias=self.bias)

    def forward(self, x, return_h=False, sigmoid=False):
        """Run the MLP!

        Args:
            x (torch.FloatTensor): Some tensor of shape [batch, ..., input_dim]
            return_h (bool): If True, also returns the last hidden layer.

        Returns:
            (torch.FloatTensor, (optional) torch.FloatTensor):
                - The output tensor of shape [batch, ..., output_dim]
                - The last hidden layer of shape [batch, ..., hidden_dim]
        """
        N = x.shape[0]

        for i, l in enumerate(self.layers):
            if i == 0:
                h = self.activation(l(x))
            elif i in self.skip:
                h = torch.cat([x, h], dim=-1)
                h = self.activation(l(h))
            else:
                h = self.activation(l(h))
        
        out = self.lout(h)
        
        if sigmoid:
            out = torch.sigmoid(out) 

        if return_h:
            return out, h
        else:
            return out


class _UnbatchedTriangleDistanceCuda(torch.autograd.Function):
    @staticmethod
    def forward(ctx, points, face_vertices):
        num_points = points.shape[0]
        num_faces = face_vertices.shape[0]
        min_dist = torch.zeros((num_points), device=points.device, dtype=points.dtype)
        min_dist_idx = torch.zeros((num_points), device=points.device, dtype=torch.long)
        dist_type = torch.zeros((num_points), device=points.device, dtype=torch.int32)
        _C.metrics.unbatched_triangle_distance_forward_cuda(
            points, face_vertices, min_dist, min_dist_idx, dist_type)
        ctx.save_for_backward(points.contiguous(), face_vertices.contiguous(),
                              min_dist_idx, dist_type)
        ctx.mark_non_differentiable(min_dist_idx, dist_type)
        return min_dist, min_dist_idx, dist_type

    @staticmethod
    def backward(ctx, grad_dist, grad_face_idx, grad_dist_type):
        points, face_vertices, face_idx, dist_type = ctx.saved_tensors
        grad_dist = grad_dist.contiguous()
        grad_points = torch.zeros_like(points)
        grad_face_vertices = torch.zeros_like(face_vertices)
        _C.metrics.unbatched_triangle_distance_backward_cuda(
            grad_dist, points, face_vertices, face_idx, dist_type,
            grad_points, grad_face_vertices)
        return grad_points, grad_face_vertices


def closest_point_fast(
    V : torch.Tensor, 
    F : torch.Tensor,
    points : torch.Tensor):

    """Returns the closest texture for a set of points.

        V (torch.FloatTensor): mesh vertices of shape [V, 3] 
        F (torch.LongTensor): mesh face indices of shape [F, 3]
        points (torch.FloatTensor): sample locations of shape [N, 3]

    Returns:
        (torch.FloatTensor): signed distances of shape [N, 1]
        (torch.FloatTensor): projected points of shape [N, 3]
        (torch.FloatTensor): face indices of shape [N, ]
    """

    face_vertices =  V[F]
    sign = check_sign(V.unsqueeze(0), F, points).squeeze(0)

    if points.is_cuda:
        cur_dist, cur_face_idx, cur_dist_type = _UnbatchedTriangleDistanceCuda.apply(
                points[0], face_vertices)
    else:
        cur_dist, cur_face_idx, cur_dist_type = _unbatched_naive_point_to_mesh_distance(
                points, face_vertices)

    # hit_point = _find_closest_point(points, face_vertices, cur_face_idx, cur_dist_type)

    dist = torch.where (sign, -torch.sqrt(cur_dist), torch.sqrt(cur_dist))


    return dist[...,None], None, cur_face_idx

def load_obj(
    fname : str, 
    load_materials : bool = False):
    """Load .obj file using TinyOBJ and extract info.
    This is more robust since it can triangulate polygon meshes 
    with up to 255 sides per face.
    
    Args:
        fname (str): path to Wavefront .obj file
    """

    assert os.path.exists(fname), \
        'Invalid file path and/or format, must be an existing Wavefront .obj'
    
    reader = tinyobjloader.ObjReader()
    config = tinyobjloader.ObjReaderConfig()
    config.triangulate = True # Ensure we don't have any polygons

    reader.ParseFromFile(fname, config)

    # Get vertices
    attrib = reader.GetAttrib()
    vertices = torch.FloatTensor(attrib.vertices).reshape(-1, 3)

    # Get triangle face indices
    shapes = reader.GetShapes()
    faces = []
    for shape in shapes:
        faces += [idx.vertex_index for idx in shape.mesh.indices]
    faces = torch.LongTensor(faces).reshape(-1, 3)
    
    mats = {}

    if load_materials:
        # Load per-faced texture coordinate indices
        texf = []
        matf = []
        for shape in shapes:
            texf += [idx.texcoord_index for idx in shape.mesh.indices]
            matf.extend(shape.mesh.material_ids)
        # texf stores [tex_idx0, tex_idx1, tex_idx2, mat_idx]
        texf = torch.LongTensor(texf).reshape(-1, 3)
        matf = torch.LongTensor(matf).reshape(-1, 1)
        texf = torch.cat([texf, matf], dim=-1)

        # Load texcoords
        texv = torch.FloatTensor(attrib.texcoords).reshape(-1, 2)
        
        # Load texture maps
        parent_path = os.path.dirname(fname) 
        materials = reader.GetMaterials()
        for i, material in enumerate(materials):
            mats[i] = {}
            diffuse = getattr(material, 'diffuse')
            if diffuse != '':
                mats[i]['diffuse'] = torch.FloatTensor(diffuse)

            for texopt in texopts:
                mat_path = getattr(material, texopt)
                if mat_path != '':
                    img = load_mat(os.path.join(parent_path, mat_path))
                    mats[i][texopt] = img
                    #mats[i][texopt.split('_')[0]] = img
        return vertices, faces, texv, texf, mats

    return vertices, faces

class SMPL_query(nn.Module):

    def __init__(self, smpl_F, can_V):
        super().__init__()
        self.smpl_F = smpl_F.cuda() #[num_faces, 3]
        self.uv = can_V.unsqueeze(0).cuda() #[1, num_vertices, 3]

    def interpolate(self, coords, smpl_V):

        """Query local features using the feature codebook, or the given input_code.
        Args:
            coords (torch.FloatTensor): coords of shape [batch, num_samples, 3]
            smpl_V (torch.FloatTensor): SMPL vertices of shape [batch, num_vertices, 3]
        Returns:
            (torch.FloatTensor): interpolated features of shape [batch, num_samples, feature_dim]
        """
        b_size = coords.shape[0]

        import trimesh
        h = trimesh.Trimesh(vertices=smpl_V[0].cpu().detach().numpy(), 
        faces=self.smpl_F.cpu().detach().numpy()) 
        h.export('/hpc2hdd/home/gzhang292/project1/LGM_final_stage1_singleview_basic/1.obj')


        sdf, hitpt, fid, weights = batched_closest_point_fast(smpl_V, self.smpl_F,
                                                              coords) # [B, Ns, 1], [B, Ns, 3], [B, Ns, 1], [B, Ns, 3]
        
        normal = torch.nn.functional.normalize( hitpt - coords, eps=1e-6, dim=2) # [B x Ns x 3]


        # return normal, sdf
        hitface = self.smpl_F[fid] # [B, Ns, 3]

        inputs_feat = self.uv.repeat(b_size, 1, 1).unsqueeze(2).expand(-1, -1, hitface.shape[-1], -1) 
            
        indices = hitface.unsqueeze(-1).expand(-1, -1, -1, inputs_feat.shape[-1])
        nearest_feats = torch.gather(input=inputs_feat, index=indices, dim=1) # [B, Ns, 3, 3]

        out_coord = torch.sum(nearest_feats * weights[...,None], dim=2) # K-weighted sum by: [B x Ns x 3]
        
        #coords_feats = torch.cat([out_coord, sdf, normal, coords[...,2:3]], dim=-1) # [B, Ns, 8]
        z = coords[...,2:3]
        return out_coord, sdf, normal, z


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

def rotatedx(original_vertices, angle=-90):
    vertices = original_vertices
    theta = np.radians(angle)
    rotation_matrix = np.array([[1, 0, 0],
                                [0, np.cos(theta), -np.sin(theta)],
                                [0, np.sin(theta), np.cos(theta)]])

    rotated_vertices = np.dot(rotation_matrix, vertices.T).T

    return rotated_vertices

def rotatedy(original_vertices, angle=-90):
    vertices = original_vertices

    theta = np.radians(angle)
    rotation_matrix = np.array([[np.cos(theta), 0, np.sin(theta)],
                                [0, 1, 0],
                                [-np.sin(theta), 0, np.cos(theta)]])

    rotated_vertices = np.dot(rotation_matrix, vertices.T).T

    return rotated_vertices

def rotatedz(original_vertices, angle=-90):
    vertices = original_vertices


    theta = np.radians(angle)
    rotation_matrix = np.array([[np.cos(theta), -np.sin(theta), 0],
                                [np.sin(theta), np.cos(theta), 0],
                                [0, 0, 1]])
    rotated_vertices = np.dot(rotation_matrix, vertices.T).T

    return rotated_vertices



class RGBMetrics(nn.Module):
    """Computation of error metrics between predicted and ground truth images

    Input:
        pred: predicted image [B, C, H, W]
        gt: ground truth image [B, C, H, W]

    Returns:
        PSNR
        SSIM
        LPIPS
    """

    def __init__(self, **kwargs):
        super().__init__()

        self.psnr = PeakSignalNoiseRatio(data_range=1.0)
        self.ssim = StructuralSimilarityIndexMeasure(data_range=1.0, kernel_size=11)
        self.lpips = LearnedPerceptualImagePatchSimilarity()

    @torch.no_grad()
    def forward(self, pred, gt):
        self.device = pred.device
        self.psnr.to(self.device)
        self.ssim.to(self.device)
        self.lpips.to(self.device)

        psnr_score = self.psnr(pred, gt)
        ssim_score = self.ssim(pred, gt)
        lpips_score = self.lpips(pred, gt)

        # return ssim_score
        return (psnr_score, ssim_score, lpips_score)
        # return (psnr_score, ssim_score, lpips_score)
def zero_module(module):
    """
    Zero out the parameters of a module and return it.
    """
    for p in module.parameters():
        p.detach().zero_()
    return module

#import smplx
import core.lib.smplx as smplx
def get_smpl_model(model_init_params):
    return smplx.create(**model_init_params)


class LGM_1(nn.Module):
    def __init__(
        self,
        opt: Options,
    ):
        super().__init__()

        self.opt = opt

        # unet
        self.unet = UNet(
            9, 14, 
            down_channels=self.opt.down_channels,
            down_attention=self.opt.down_attention,
            mid_attention=self.opt.mid_attention,
            up_channels=self.opt.up_channels,
            up_attention=self.opt.up_attention,
        )

        # last conv
        self.conv = nn.Conv2d(14, 14, kernel_size=1) # NOTE: maybe remove it if train again

        # Gaussian Renderer
        self.gs = GaussianRenderer(opt)

        # activations...
        self.pos_act = lambda x: x.clamp(-1, 1)
        self.scale_act = lambda x: 0.1 * F.softplus(x)
        self.opacity_act = lambda x: torch.sigmoid(x)
        self.rot_act = F.normalize
        self.rgb_act = lambda x: 0.5 * torch.tanh(x) + 0.5 # NOTE: may use sigmoid if train again

        # LPIPS loss
        if self.opt.lambda_lpips > 0:
            self.lpips_loss = LPIPS(net='vgg')
            self.lpips_loss.requires_grad_(False)

        # gs renderer
        self.tan_half_fov = np.tan(0.5 * np.deg2rad(opt.fovy))
        self.proj_matrix = torch.zeros(4, 4, dtype=torch.float32, device='cpu')
        self.proj_matrix[0, 0] = 1 / self.tan_half_fov
        self.proj_matrix[1, 1] = 1 / self.tan_half_fov
        self.proj_matrix[2, 2] = (opt.zfar + opt.znear) / (opt.zfar - opt.znear)
        self.proj_matrix[3, 2] = - (opt.zfar * opt.znear) / (opt.zfar - opt.znear)
        self.proj_matrix[2, 3] = 1

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
                    num_pca_comps=10
                )

        self.smpl_model = get_smpl_model(model_init_params)


    def state_dict(self, **kwargs):
        # remove lpips_loss
        state_dict = super().state_dict(**kwargs)
        for k in list(state_dict.keys()):
            if 'lpips_loss' in k:
                del state_dict[k]
        return state_dict

    def prepare_default_rays(self, device, elevation=0):
        
        from kiui.cam import orbit_camera
        from core.utils import get_rays

        cam_poses = np.stack([
            orbit_camera(elevation, 0, radius=self.opt.cam_radius),
            orbit_camera(elevation, 90, radius=self.opt.cam_radius),
            orbit_camera(elevation, 180, radius=self.opt.cam_radius),
            orbit_camera(elevation, 270, radius=self.opt.cam_radius),
        ], axis=0) # [4, 4, 4]
        cam_poses = torch.from_numpy(cam_poses)

        rays_embeddings = []
        for i in range(cam_poses.shape[0]):
            rays_o, rays_d = get_rays(cam_poses[i], self.opt.input_size, self.opt.input_size, self.opt.fovy) # [h, w, 3]
            rays_plucker = torch.cat([torch.cross(rays_o, rays_d, dim=-1), rays_d], dim=-1) # [h, w, 6]
            rays_embeddings.append(rays_plucker)
        rays_embeddings = torch.stack(rays_embeddings, dim=0).permute(0, 3, 1, 2).contiguous().to(device) # [V, 6, h, w]  这里可能有问题！！！！！！！  前后左右的顺序应该是0，
        
        return rays_embeddings

        
    def depth_to_xyz(self, output_c2ws, x, final_res):
        camera_feature = output_c2ws.flatten(-2, -1)
        c2w = camera_feature[:, 0:1, :16]
        batch_size = x.shape[0]
        input_views = 1

        c2w = c2w.reshape(batch_size, input_views, 4, 4)

        input_c2ws = output_c2ws[:, 0:input_views]

        ray_o = c2w[:, :, :3, 3]
        depth_offset = torch.norm(ray_o, dim=-1, p=2, keepdim=True)

        output_fxfycxcy = torch.tensor([[[1.0946174630511205, 1.0946174630511205, 0.5, 0.5]]]).to(c2w.device).repeat(c2w.shape[0],c2w.shape[1],1)
        input_fxfycxcy = output_fxfycxcy[:, 0:input_views]

        output_depth = -x[..., 2]
        output_depth = (output_depth.reshape(batch_size, input_views, -1) + depth_offset[:,0:1]).reshape(batch_size, -1, 1)
        output_depth = output_depth.reshape(batch_size, input_views, final_res, final_res)
        output_xyz_z = unproject_depth(output_depth, input_fxfycxcy, input_c2ws)

        output_depth = -x[..., 1]
        output_depth = (output_depth.reshape(batch_size, input_views, -1) + depth_offset[:,0:1]).reshape(batch_size, -1, 1)
        output_depth = output_depth.reshape(batch_size, input_views, final_res, final_res)
        output_xyz_y = unproject_depth(output_depth, input_fxfycxcy, input_c2ws)

        output_depth = -x[..., 0]
        output_depth = (output_depth.reshape(batch_size, input_views, -1) + depth_offset[:,0:1]).reshape(batch_size, -1, 1)
        output_depth = output_depth.reshape(batch_size, input_views, final_res, final_res)
        output_xyz_x = unproject_depth(output_depth, input_fxfycxcy, input_c2ws)

        xyz = (output_xyz_x+output_xyz_y+output_xyz_z)/3
        # xyz = output_xyz_z
        return torch.cat([xyz,x[...,3:]], dim=-1)

    def config_conv_layer_smpl(self):
        self.upsample = torch.nn.Upsample(
            size=None, 
            scale_factor=2, 
            mode='bilinear', 
            align_corners=None, 
            recompute_scale_factor=None
        )

        self.rgb_metrics = RGBMetrics()
        self.conv_extra_smpl0 = copy.deepcopy(self.conv)
        del self.conv


    def forward_gaussians(self, images, smpl_out_conv, smpl_mid_block, data=None, front_normals=None, training=False):
        # images: [B, 4, 9, H, W]
        # return: Gaussians: [B, dim_t]

        B, V, C, H, W = images.shape     # [1, 8, 9, 512, 512]
        images = images.view(B*V, C, H, W)
        x, _ = self.unet(images, True) 
        
        x_up = x[0]

        x_new0 = self.conv_extra_smpl0(x_up)

        x_new = x_new0 
        x_new = x_new.view( -1, 14, self.opt.splat_size, self.opt.splat_size)
        view = 8
        x = x_new.reshape(B, view, 14, self.opt.splat_size, self.opt.splat_size)
        x = x.permute(0, 1, 3, 4, 2).reshape(B, -1, 14)

        pos = self.pos_act(x[..., 0:3]) # [B, N, 3]   
        opacity = self.opacity_act(x[..., 3:4])
        scale = self.scale_act(x[..., 4:7])
        rotation = self.rot_act(x[..., 7:11])
        rgbs = self.rgb_act(x[..., 11:])

        gaussians = torch.cat([pos, opacity, scale, rotation, rgbs], dim=-1) # [B, N, 14]

        return gaussians

    def forward_gaussians_normals_smpl(self, images):
        B, V, C, H, W = images.shape
        images = images.view(B*V, C, H, W)

        x, smpl_mid_block = self.unet_normals_smpl(images) 

        smpl_out_conv = x
        x = x.reshape(B, 4, 14, self.opt.splat_size//2, self.opt.splat_size//2)

        x = x.permute(0, 1, 3, 4, 2).reshape(B, -1, 14)
        
        pos = self.pos_act(x[..., 0:3]) # [B, N, 3]
        opacity = self.opacity_act(x[..., 3:4])
        scale = self.scale_act(x[..., 4:7])
        rotation = self.rot_act(x[..., 7:11])
        rgbs = self.rgb_act(x[..., 11:])

        gaussians = torch.cat([pos, opacity, scale, rotation, rgbs], dim=-1) # [B, N, 14]
        
        return gaussians, smpl_out_conv, smpl_mid_block


    def sparsify(self, t_vertices):
        random_indices = np.random.permutation(65536)
        indices = torch.from_numpy(random_indices[:20000])

        t_vertices_sample = t_vertices[:, indices, :]

        return t_vertices_sample


    # NOTE: Gaussian animation functions - Nanjie
    def gaussian_animation(self, Apose_gaussians, Apose_param, target_param, data=None):
        """
        Input:
            tpose_gaussians: - Refined Gaussians from LGM [b, 14, n]
            Apose_vertices:  - T-pose SMPL vertices [b, 10475, 3]
            Apose_parameters
            target_parameters

        Return:
            transposed_gaussians: Reposed Gaussian Avatar [b, 14, n]
        """        
        bsz = Apose_gaussians.shape[0]

        Tpose_gaussian_bsz = []
        Posed_gaussian_bsz = []
        for item in range(bsz):
            
            Apose_gaussians = self.gs.prune(Apose_gaussians)
            
            gs_xyz = Apose_gaussians[item, :, :3] #[524288,3]

            # Transform to T-pose:
            model_forward_params_A = dict(
                betas=Apose_param['betas'][item],
                global_orient=Apose_param['global_orient'][item],
                body_pose=Apose_param['body_pose'][item],
                left_hand_pose=Apose_param['left_hand_pose'][item],
                right_hand_pose=Apose_param['right_hand_pose'][item],
                jaw_pose=Apose_param['jaw_pose'][item],
                leye_pose=Apose_param['leye_pose'][item],
                reye_pose=Apose_param['reye_pose'][item],
                expression=Apose_param['expression'][item],
                return_verts=True,
                return_joint_transformation=False,
                return_vertex_transformation=False)

            smpl_out_tpose = self.smpl_model(**model_forward_params_A)

            Apose_mesh = trimesh.Trimesh(smpl_out_tpose.vertices.squeeze(0).float().detach().cpu().numpy(), self.smpl_model.faces)
            
            # To obtain the translation and scales
            Apose_mesh, translation, scales_inv = new_mesh(Apose_mesh, 0)
        
            model_forward_params_A = dict(
                betas=Apose_param['betas'][item],
                global_orient=Apose_param['global_orient'][item],
                body_pose=Apose_param['body_pose'][item],
                left_hand_pose=Apose_param['left_hand_pose'][item],
                right_hand_pose=Apose_param['right_hand_pose'][item],
                jaw_pose=Apose_param['jaw_pose'][item],
                leye_pose=Apose_param['leye_pose'][item],
                reye_pose=Apose_param['reye_pose'][item],
                expression=Apose_param['expression'][item],
                return_verts=True,
                return_joint_transformation=True,
                return_vertex_transformation=True,
                manual_scale = scales_inv,
                manual_transl = translation) 
            smpl_out_tpose = self.smpl_model(**model_forward_params_A)
            
            # Find nearest smpl vertex
            Apose_vertices = torch.from_numpy(Apose_mesh.vertices).float().to(Apose_gaussians.device)
            smpl_tree = cKDTree(Apose_vertices.cpu().numpy()) #[10xxx,3]
            _, idx = smpl_tree.query(gs_xyz.detach().cpu().numpy(), k=3) #[116308,3]
            
            # t-pose for gaussians
            rot_mat_t = (smpl_out_tpose.vertex_transformation.detach()[0][idx[:, 0]]).float()
            homo_coord = torch.ones_like(gs_xyz)[..., :1]
            tpose_gs_xyz = torch.inverse(rot_mat_t) @ torch.cat([gs_xyz, homo_coord],dim=1).unsqueeze(-1)
            tpose_gs_xyz = tpose_gs_xyz[:, :3, 0]
            
            tpose_gaussians = torch.cat([tpose_gs_xyz, Apose_gaussians[item, :, 3:]], dim = -1)
            # self.gs.save_ply(self.tpose_gaussians.detach().cpu().unsqueeze(0), '/hpc2hdd/home/gzhang292/nanjie/project3/LBS_LGM_nanjie/vis/tpose.ply')
            
            # Repose
            model_forward_params_re = dict(
                betas=Apose_param['betas'][item],
                global_orient=target_param['global_orient'][item],
                body_pose=target_param['body_pose'][item],
                left_hand_pose=target_param['left_hand_pose'][item],
                right_hand_pose=target_param['right_hand_pose'][item],
                jaw_pose=target_param['jaw_pose'][item],
                leye_pose=target_param['leye_pose'][item],
                reye_pose=target_param['reye_pose'][item],
                expression=target_param['expression'][item],
                return_verts=False,
                return_joint_transformation=True,
                return_vertex_transformation=True,
                manual_scale = scales_inv,
                manual_transl = translation)
            
            smpl_out_repose = self.smpl_model(**model_forward_params_re)
            
            rot_mat_da = (smpl_out_repose.vertex_transformation.detach()[0][idx[:, 0]]).float()
            rot_gs_xyz = rot_mat_da @ torch.cat([tpose_gs_xyz, homo_coord], dim=-1).unsqueeze(-1)
            rot_gs_xyz = rot_gs_xyz[:, :3, 0]

            # concat the gaussians 
            rot_gs_xyz,_,_ = new_verts(rot_gs_xyz, 0)
            transposed_gaussians = torch.cat([rot_gs_xyz, tpose_gaussians[:, 3:]], dim = -1)

            Tpose_gaussian_bsz.append(tpose_gaussians.unsqueeze(0))
            Posed_gaussian_bsz.append(transposed_gaussians.unsqueeze(0))

        tpose_gaussians = torch.cat(Tpose_gaussian_bsz, dim = 0)
        transposed_gaussians = torch.cat(Posed_gaussian_bsz, dim = 0)
        
        return tpose_gaussians, transposed_gaussians #[1, n, 14]


    def get_transform_params_torch(smpl, params, rot_mats=None, correct_Rs=None):
        """
        obtain the transformation parameters for linear blend skinning
        """
        v_template = smpl['v_template']

        # add shape blend shapes
        shapedirs = smpl['shapedirs']
        betas = params['shapes']
        # v_shaped = v_template[None] + torch.sum(shapedirs[None] * betas[:,None], axis=-1).float()
        v_shaped = v_template[None] + torch.sum(shapedirs[None][...,:betas.shape[-1]] * betas[:,None], axis=-1).float()

        if rot_mats is None:
            # add pose blend shapes
            poses = params['poses'].reshape(-1, 3)
            # bs x 24 x 3 x 3
            rot_mats = batch_rodrigues_torch(poses).view(params['poses'].shape[0], -1, 3, 3)

            if correct_Rs is not None:
                rot_mats_no_root = rot_mats[:, 1:]
                rot_mats_no_root = torch.matmul(rot_mats_no_root.reshape(-1, 3, 3), correct_Rs.reshape(-1, 3, 3)).reshape(-1, rot_mats.shape[1]-1, 3, 3)
                rot_mats = torch.cat([rot_mats[:, 0:1], rot_mats_no_root], dim=1)

        # obtain the joints
        joints = torch.matmul(smpl['J_regressor'][None], v_shaped) # [bs, 24 ,3]

        # obtain the rigid transformation
        parents = smpl['kintree_table'][0]
        A = get_rigid_transformation_torch(rot_mats, joints, parents)

        # apply global transformation
        R = params['R'] 
        Th = params['Th'] 
        return A, R, Th, joints


    def forward(self, data, step_ratio=1, lpips_weight_additon=0, training=False):
        # data: output of the dataloader
        # return: template gassians
        results = {}
        loss = 0
        before_act = None
        smpl_mid_block = None

        images = data['final_input']
        gaussians_stage1 = self.forward_gaussians(images, before_act, smpl_mid_block, data, front_normals=None, training=training) # [B, N, 14]
        results['A_gaussians'] = gaussians_stage1

        return results, data


class LGM_2(nn.Module):
    def __init__(
        self,
        opt: Options,
    ):
        super().__init__()

        self.opt = opt

        # unet
        self.unet = UNet(
            9, 14, 
            down_channels=self.opt.down_channels,
            down_attention=self.opt.down_attention,
            mid_attention=self.opt.mid_attention,
            up_channels=self.opt.up_channels,
            up_attention=self.opt.up_attention,
        )

        # last conv
        self.conv = nn.Conv2d(14, 14, kernel_size=1) # NOTE: maybe remove it if train again

        # Gaussian Renderer
        self.gs = GaussianRenderer(opt)

        # activations...
        self.pos_act = lambda x: x.clamp(-1, 1)
        self.scale_act = lambda x: 0.1 * F.softplus(x)
        self.opacity_act = lambda x: torch.sigmoid(x)
        self.rot_act = F.normalize
        self.rgb_act = lambda x: 0.5 * torch.tanh(x) + 0.5 # NOTE: may use sigmoid if train again

        # LPIPS loss
        if self.opt.lambda_lpips > 0:
            self.lpips_loss = LPIPS(net='vgg')
            self.lpips_loss.requires_grad_(False)

        # gs renderer
        self.tan_half_fov = np.tan(0.5 * np.deg2rad(opt.fovy))
        self.proj_matrix = torch.zeros(4, 4, dtype=torch.float32, device='cpu')
        self.proj_matrix[0, 0] = 1 / self.tan_half_fov
        self.proj_matrix[1, 1] = 1 / self.tan_half_fov
        self.proj_matrix[2, 2] = (opt.zfar + opt.znear) / (opt.zfar - opt.znear)
        self.proj_matrix[3, 2] = - (opt.zfar * opt.znear) / (opt.zfar - opt.znear)
        self.proj_matrix[2, 3] = 1

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
                    num_pca_comps=10
                )

        self.smpl_model = get_smpl_model(model_init_params)


    def state_dict(self, **kwargs):
        # remove lpips_loss
        state_dict = super().state_dict(**kwargs)
        for k in list(state_dict.keys()):
            if 'lpips_loss' in k:
                del state_dict[k]
        return state_dict

    def prepare_default_rays(self, device, elevation=0):
        
        from kiui.cam import orbit_camera
        from core.utils import get_rays

        cam_poses = np.stack([
            orbit_camera(elevation, 0, radius=self.opt.cam_radius),
            orbit_camera(elevation, 90, radius=self.opt.cam_radius),
            orbit_camera(elevation, 180, radius=self.opt.cam_radius),
            orbit_camera(elevation, 270, radius=self.opt.cam_radius),
        ], axis=0) # [4, 4, 4]
        cam_poses = torch.from_numpy(cam_poses)

        rays_embeddings = []
        for i in range(cam_poses.shape[0]):
            rays_o, rays_d = get_rays(cam_poses[i], self.opt.input_size, self.opt.input_size, self.opt.fovy) # [h, w, 3]
            rays_plucker = torch.cat([torch.cross(rays_o, rays_d, dim=-1), rays_d], dim=-1) # [h, w, 6]
            rays_embeddings.append(rays_plucker)
        rays_embeddings = torch.stack(rays_embeddings, dim=0).permute(0, 3, 1, 2).contiguous().to(device) # [V, 6, h, w]  这里可能有问题！！！！！！！  前后左右的顺序应该是0，
        
        return rays_embeddings

        
    def depth_to_xyz(self, output_c2ws, x, final_res):
        camera_feature = output_c2ws.flatten(-2, -1)
        c2w = camera_feature[:, 0:1, :16]
        batch_size = x.shape[0]
        input_views = 1

        c2w = c2w.reshape(batch_size, input_views, 4, 4)

        input_c2ws = output_c2ws[:, 0:input_views]

        ray_o = c2w[:, :, :3, 3]
        depth_offset = torch.norm(ray_o, dim=-1, p=2, keepdim=True)

        output_fxfycxcy = torch.tensor([[[1.0946174630511205, 1.0946174630511205, 0.5, 0.5]]]).to(c2w.device).repeat(c2w.shape[0],c2w.shape[1],1)
        input_fxfycxcy = output_fxfycxcy[:, 0:input_views]

        output_depth = -x[..., 2]
        output_depth = (output_depth.reshape(batch_size, input_views, -1) + depth_offset[:,0:1]).reshape(batch_size, -1, 1)
        output_depth = output_depth.reshape(batch_size, input_views, final_res, final_res)
        output_xyz_z = unproject_depth(output_depth, input_fxfycxcy, input_c2ws)

        output_depth = -x[..., 1]
        output_depth = (output_depth.reshape(batch_size, input_views, -1) + depth_offset[:,0:1]).reshape(batch_size, -1, 1)
        output_depth = output_depth.reshape(batch_size, input_views, final_res, final_res)
        output_xyz_y = unproject_depth(output_depth, input_fxfycxcy, input_c2ws)

        output_depth = -x[..., 0]
        output_depth = (output_depth.reshape(batch_size, input_views, -1) + depth_offset[:,0:1]).reshape(batch_size, -1, 1)
        output_depth = output_depth.reshape(batch_size, input_views, final_res, final_res)
        output_xyz_x = unproject_depth(output_depth, input_fxfycxcy, input_c2ws)

        xyz = (output_xyz_x+output_xyz_y+output_xyz_z)/3
        # xyz = output_xyz_z
        return torch.cat([xyz,x[...,3:]], dim=-1)

    def config_conv_layer_smpl(self):
        self.upsample = torch.nn.Upsample(
            size=None, 
            scale_factor=2, 
            mode='bilinear', 
            align_corners=None, 
            recompute_scale_factor=None
        )

        self.rgb_metrics = RGBMetrics()
        self.conv_extra_smpl0 = copy.deepcopy(self.conv)
        del self.conv


    def forward_gaussians(self, images, smpl_out_conv, smpl_mid_block, data=None, front_normals=None, training=False):
        # images: [B, 4, 9, H, W]
        # return: Gaussians: [B, dim_t]

        B, V, C, H, W = images.shape     # [1, 8, 9, 512, 512]
        images = images.view(B*V, C, H, W)
        x, _ = self.unet(images, True) 
        
        x_up = x[0]

        x_new0 = self.conv_extra_smpl0(x_up)

        x_new = x_new0 
        x_new = x_new.view( -1, 14, self.opt.splat_size, self.opt.splat_size)
        view = 8
        x = x_new.reshape(B, view, 14, self.opt.splat_size, self.opt.splat_size)
        x = x.permute(0, 1, 3, 4, 2).reshape(B, -1, 14)

        pos = self.pos_act(x[..., 0:3]) # [B, N, 3]   
        opacity = self.opacity_act(x[..., 3:4])
        scale = self.scale_act(x[..., 4:7])
        rotation = self.rot_act(x[..., 7:11])
        rgbs = self.rgb_act(x[..., 11:])

        gaussians = torch.cat([pos, opacity, scale, rotation, rgbs], dim=-1) # [B, N, 14]

        return gaussians, gaussians

    def forward_gaussians_normals_smpl(self, images):
        B, V, C, H, W = images.shape
        images = images.view(B*V, C, H, W)

        x, smpl_mid_block = self.unet_normals_smpl(images) 

        smpl_out_conv = x
        x = x.reshape(B, 4, 14, self.opt.splat_size//2, self.opt.splat_size//2)

        x = x.permute(0, 1, 3, 4, 2).reshape(B, -1, 14)
        
        pos = self.pos_act(x[..., 0:3]) # [B, N, 3]
        opacity = self.opacity_act(x[..., 3:4])
        scale = self.scale_act(x[..., 4:7])
        rotation = self.rot_act(x[..., 7:11])
        rgbs = self.rgb_act(x[..., 11:])

        gaussians = torch.cat([pos, opacity, scale, rotation, rgbs], dim=-1) # [B, N, 14]
        
        return gaussians, smpl_out_conv, smpl_mid_block


    def sparsify(self, t_vertices):
        random_indices = np.random.permutation(65536)

        # randomly select data for training (not used actually)
        indices = torch.from_numpy(random_indices[:20000])  

        t_vertices_sample = t_vertices[:, indices, :]

        return t_vertices_sample
    

    def prepare_render_images(self, batch):

        origin_camera = PerspectiveCamera(fovy=self.opt.fovy, device=batch['origin_v_len'].device)
        origin_renderer = NeuralRender(device=batch['origin_v_len'].device, camera_model=origin_camera)
        c2ws = torch.cat([batch['input_c2ws'],batch['target_c2ws']],dim=1)

   
        batch_size = c2ws.size(0)
        origin_image_list = []
        origin_alpha_list = []
        origin_depth_list = []
        origin_normal_list = []
        origin_bg_white = torch.ones(3, dtype=torch.float32, device=c2ws.device)
        origin_bg_black = torch.zeros(3, dtype=torch.float32, device=c2ws.device)

        gt_camera = PerspectiveCamera(fovy=self.opt.fovy, device=batch['gt_v_len'].device)
        gt_renderer = NeuralRender(device=batch['gt_v_len'].device, camera_model=gt_camera)
        gt_image_list = []
        gt_alpha_list = []
        gt_depth_list = []
        gt_normal_list = []
        
        posed_gt_image_list = []
        posed_gt_alpha_list = []
        posed_gt_depth_list = []
        posed_gt_normal_list = []

        normal_list_smpl = []
        alpha_list_smpl = []

        normal_list_smpl_target = []
        alpha_list_smpl_target = []

        gt_bg_white = torch.ones(3, dtype=torch.float32, device=c2ws.device)
        gt_bg_black = torch.zeros(3, dtype=torch.float32, device=c2ws.device)

        for idx in range(batch_size):
            ### 1.Scan (from input) rendering

            origin_mesh_v_nx3 = batch['scan_v'][idx, :batch['origin_v_len'][idx]]
            origin_mesh_f_fx3 = batch['scan_f'][idx]
            origin_camera_mv_bx4x4 = torch.linalg.inv(c2ws[idx])

            origin_vtx_uv = batch['scan_vt'][idx, :batch['origin_v_len'][idx]]
            origin_uv_idx = batch['scan_ft'][idx]
            origin_tex = batch['origin_tex'][idx]

            try:
                origin_out = origin_renderer.render_mesh(
                            origin_mesh_v_nx3.unsqueeze(dim=0),
                            origin_mesh_f_fx3,
                            origin_camera_mv_bx4x4,
                            origin_mesh_v_nx3.unsqueeze(dim=0),
                            resolution=2048,
                            device=batch['origin_v_len'].device,
                            hierarchical_mask=False,
                            uv=origin_vtx_uv,
                            uv_idx=origin_uv_idx,
                            tex=origin_tex,
                        )
            except:
                origin_out = origin_renderer.render_mesh(
                            origin_mesh_v_nx3.unsqueeze(dim=0),
                            origin_mesh_f_fx3,
                            origin_camera_mv_bx4x4,
                            origin_mesh_v_nx3.unsqueeze(dim=0),
                            resolution=2048,
                            device=batch['origin_v_len'].device,
                            hierarchical_mask=False,
                            uv=origin_vtx_uv,
                            uv_idx=origin_uv_idx,
                            tex=origin_tex,
                        )
            origin_mesh_feature, origin_antialias_mask, origin_hard_mask, origin_rast, origin_v_pos_clip, origin_mask_pyramid, origin_depth, origin_normal, origin_image = origin_out
        
            origin_alpha = origin_hard_mask

            origin_normal = origin_normal * origin_alpha + origin_bg_white * (1-origin_alpha)
            origin_image = origin_image * origin_alpha + origin_bg_white * (1-origin_alpha)

            origin_image = origin_image.permute(0, 3, 1, 2).contiguous().float()
            origin_normal = origin_normal.permute(0, 3, 1, 2).contiguous().float()
            origin_alpha = origin_alpha.permute(0, 3, 1, 2).contiguous().float()
            origin_depth = origin_depth.permute(0, 3, 1, 2).contiguous().float()


            origin_image = F.interpolate(origin_image, (self.opt.output_size, self.opt.output_size), mode='bilinear', align_corners=False)
            origin_normal = F.interpolate(origin_normal, (self.opt.output_size, self.opt.output_size), mode='bilinear', align_corners=False)
            origin_alpha = F.interpolate(origin_alpha, (self.opt.output_size, self.opt.output_size), mode='bilinear', align_corners=False)
            origin_depth = F.interpolate(origin_depth, (self.opt.output_size, self.opt.output_size), mode='bilinear', align_corners=False)

            ### 2.A-pose Scan (from input) rendering
            gt_mesh_v_nx3 = batch['gt_v'][idx, :batch['gt_v_len'][idx]]
            gt_mesh_f_fx3 = batch['gt_f'][idx]
            gt_camera_mv_bx4x4 = torch.linalg.inv(c2ws[idx])

            gt_vtx_uv = batch['gt_vt'][idx, :batch['gt_v_len'][idx]]
            gt_uv_idx = batch['gt_ft'][idx]
            gt_tex = batch['gt_tex'][idx]

            try:
                gt_out = gt_renderer.render_mesh(
                            gt_mesh_v_nx3.unsqueeze(dim=0),
                            gt_mesh_f_fx3,
                            gt_camera_mv_bx4x4,
                            gt_mesh_v_nx3.unsqueeze(dim=0),
                            resolution=2048,
                            device=batch['gt_v_len'].device,
                            hierarchical_mask=False,
                            uv=gt_vtx_uv,
                            uv_idx=gt_uv_idx,
                            tex=gt_tex,
                        )
            except:
                gt_out = gt_renderer.render_mesh(
                            gt_mesh_v_nx3.unsqueeze(dim=0),
                            gt_mesh_f_fx3,
                            gt_camera_mv_bx4x4,
                            gt_mesh_v_nx3.unsqueeze(dim=0),
                            resolution=2048,
                            device=batch['gt_v_len'].device,
                            hierarchical_mask=False,
                            uv=gt_vtx_uv,
                            uv_idx=gt_uv_idx,
                            tex=gt_tex,
                        )
            gt_mesh_feature, gt_antialias_mask, gt_hard_mask, gt_rast, gt_v_pos_clip, gt_mask_pyramid, gt_depth, gt_normal, gt_image = gt_out
        
            gt_alpha = gt_hard_mask

            gt_normal = gt_normal * gt_alpha + gt_bg_white * (1-gt_alpha)
            gt_image = gt_image * gt_alpha + gt_bg_white * (1-gt_alpha)

            gt_image = gt_image.permute(0, 3, 1, 2).contiguous().float()
            gt_normal = gt_normal.permute(0, 3, 1, 2).contiguous().float()
            gt_alpha = gt_alpha.permute(0, 3, 1, 2).contiguous().float()
            gt_depth = gt_depth.permute(0, 3, 1, 2).contiguous().float()


            gt_image = F.interpolate(gt_image, (self.opt.output_size, self.opt.output_size), mode='bilinear', align_corners=False)
            gt_normal = F.interpolate(gt_normal, (self.opt.output_size, self.opt.output_size), mode='bilinear', align_corners=False)
            gt_alpha = F.interpolate(gt_alpha, (self.opt.output_size, self.opt.output_size), mode='bilinear', align_corners=False)
            gt_depth = F.interpolate(gt_depth, (self.opt.output_size, self.opt.output_size), mode='bilinear', align_corners=False)
            
            
            ### 3.Targetpose GT rendering
            posed_gt_mesh_v_nx3 = batch['posed_scan_v'][idx, :batch['posed_v_len'][idx]]
            posed_gt_mesh_f_fx3 = batch['posed_scan_f'][idx]
            gt_camera_mv_bx4x4 = torch.linalg.inv(c2ws[idx])

            posed_gt_vtx_uv = batch['posed_scan_vt'][idx, :batch['posed_v_len'][idx]]
            posed_gt_uv_idx = batch['posed_scan_ft'][idx]
            posed_gt_tex = batch['posed_gt_tex'][idx]

            try:
                posed_gt_out = gt_renderer.render_mesh(
                            posed_gt_mesh_v_nx3.unsqueeze(dim=0),
                            posed_gt_mesh_f_fx3,
                            gt_camera_mv_bx4x4,
                            posed_gt_mesh_v_nx3.unsqueeze(dim=0),
                            resolution=2048,
                            device=batch['posed_v_len'].device,
                            hierarchical_mask=False,
                            uv=posed_gt_vtx_uv,
                            uv_idx=posed_gt_uv_idx,
                            tex=posed_gt_tex,
                        )
            except:
                posed_gt_out = gt_renderer.render_mesh(
                            posed_gt_mesh_v_nx3.unsqueeze(dim=0),
                            posed_gt_mesh_f_fx3,
                            gt_camera_mv_bx4x4,
                            posed_gt_mesh_v_nx3.unsqueeze(dim=0),
                            resolution=2048,
                            device=batch['posed_v_len'].device,
                            hierarchical_mask=False,
                            uv=posed_gt_vtx_uv,
                            uv_idx=posed_gt_uv_idx,
                            tex=posed_gt_tex,
                        )
            posed_gt_mesh_feature, posed_gt_antialias_mask, posed_gt_hard_mask, posed_gt_rast, posed_gt_v_pos_clip, posed_gt_mask_pyramid, posed_gt_depth, posed_gt_normal, posed_gt_image = posed_gt_out
        
            posed_gt_alpha = posed_gt_hard_mask

            posed_gt_normal = posed_gt_normal * posed_gt_alpha + gt_bg_white * (1-posed_gt_alpha)
            posed_gt_image  = posed_gt_image * posed_gt_alpha + gt_bg_white * (1-posed_gt_alpha)

            posed_gt_image  = posed_gt_image.permute(0, 3, 1, 2).contiguous().float()
            posed_gt_normal = posed_gt_normal.permute(0, 3, 1, 2).contiguous().float()
            posed_gt_alpha  = posed_gt_alpha.permute(0, 3, 1, 2).contiguous().float()
            posed_gt_depth  = posed_gt_depth.permute(0, 3, 1, 2).contiguous().float()


            posed_gt_image  = F.interpolate(posed_gt_image, (self.opt.output_size, self.opt.output_size), mode='bilinear', align_corners=False)
            posed_gt_normal = F.interpolate(posed_gt_normal, (self.opt.output_size, self.opt.output_size), mode='bilinear', align_corners=False)
            posed_gt_alpha  = F.interpolate(posed_gt_alpha, (self.opt.output_size, self.opt.output_size), mode='bilinear', align_corners=False)
            posed_gt_depth  = F.interpolate(posed_gt_depth, (self.opt.output_size, self.opt.output_size), mode='bilinear', align_corners=False)


            ### 4.Targetpose SMPL rendring (used in stage2)
            smpl_v_nx3 = batch['target_smpl_v'][idx]
            smpl_f_fx3 = batch['target_smpl_f'][idx]
 
            try:
                out_smpl = gt_renderer.render_mesh(
                            smpl_v_nx3.unsqueeze(dim=0),
                            smpl_f_fx3,
                            gt_camera_mv_bx4x4,
                            smpl_v_nx3.unsqueeze(dim=0),
                            resolution=2048,
                            device=batch['gt_v_len'].device,
                            hierarchical_mask=False,
                            uv=None,
                            uv_idx=None,
                            tex=None,
                        )
            except:
                out_smpl = gt_renderer.render_mesh(
                            smpl_v_nx3.unsqueeze(dim=0),
                            smpl_f_fx3,
                            gt_camera_mv_bx4x4,
                            smpl_v_nx3.unsqueeze(dim=0),
                            resolution=2048,
                            device=batch['gt_v_len'].device,
                            hierarchical_mask=False,
                            uv=None,
                            uv_idx=None,
                            tex=None,
                        )
            ori_mesh_feature, antialias_mask, hard_mask_smpl, rast, v_pos_clip, mask_pyramid, depth, normal_smpl = out_smpl

            alpha_smpl = hard_mask_smpl

            normal_smpl = normal_smpl * alpha_smpl + gt_bg_white * (1-alpha_smpl)

            normal_smpl = normal_smpl.permute(0, 3, 1, 2).contiguous().float()
            alpha_smpl = alpha_smpl.permute(0, 3, 1, 2).contiguous().float()
            normal_smpl_target = F.interpolate(normal_smpl, (self.opt.output_size, self.opt.output_size), mode='bilinear', align_corners=False)
            alpha_smpl_target = F.interpolate(alpha_smpl, (self.opt.output_size, self.opt.output_size), mode='bilinear', align_corners=False)


            ### 5. Smpl (from inpur) rendering
            smpl_v_nx3 = batch['origin_smpl_v'][idx]
            smpl_f_fx3 = batch['origin_smpl_f'][idx]
 
            try:
                out_smpl = gt_renderer.render_mesh(
                            smpl_v_nx3.unsqueeze(dim=0),
                            smpl_f_fx3,
                            gt_camera_mv_bx4x4,
                            smpl_v_nx3.unsqueeze(dim=0),
                            resolution=2048,
                            device=batch['gt_v_len'].device,
                            hierarchical_mask=False,
                            uv=None,
                            uv_idx=None,
                            tex=None,
                        )
            except:
                out_smpl = gt_renderer.render_mesh(
                            smpl_v_nx3.unsqueeze(dim=0),
                            smpl_f_fx3,
                            gt_camera_mv_bx4x4,
                            smpl_v_nx3.unsqueeze(dim=0),
                            resolution=2048,
                            device=batch['gt_v_len'].device,
                            hierarchical_mask=False,
                            uv=None,
                            uv_idx=None,
                            tex=None,
                        )
            ori_mesh_feature, antialias_mask, hard_mask_smpl, rast, v_pos_clip, mask_pyramid, depth, normal_smpl = out_smpl

            alpha_smpl = hard_mask_smpl

            normal_smpl = normal_smpl * alpha_smpl + gt_bg_white * (1-alpha_smpl)

            normal_smpl = normal_smpl.permute(0, 3, 1, 2).contiguous().float()
            alpha_smpl = alpha_smpl.permute(0, 3, 1, 2).contiguous().float()
            normal_smpl = F.interpolate(normal_smpl, (self.opt.output_size, self.opt.output_size), mode='bilinear', align_corners=False)
            alpha_smpl = F.interpolate(alpha_smpl, (self.opt.output_size, self.opt.output_size), mode='bilinear', align_corners=False)


            # store rendering results of 1.
            origin_image_list.append(origin_image)
            origin_alpha_list.append(origin_alpha)
            origin_depth_list.append(origin_depth)
            origin_normal_list.append(origin_normal)

            # store rendering results of 2.
            gt_image_list.append(gt_image)
            gt_alpha_list.append(gt_alpha)
            gt_depth_list.append(gt_depth)
            gt_normal_list.append(gt_normal)

            # store rendering results of 3.
            posed_gt_image_list.append(posed_gt_image)
            posed_gt_alpha_list.append(posed_gt_alpha)
            posed_gt_depth_list.append(posed_gt_depth)
            posed_gt_normal_list.append(posed_gt_normal)

            # store rendering results of 4.
            normal_list_smpl_target.append(normal_smpl_target)
            alpha_list_smpl_target.append(alpha_smpl_target)

            # store rendering results of 5.
            normal_list_smpl.append(normal_smpl)
            alpha_list_smpl.append(alpha_smpl)

        origin_images = torch.stack(origin_image_list, dim=0).float()                 # (6+V, 3, H, W)
        origin_alphas = torch.stack(origin_alpha_list, dim=0).float()                 # (6+V, 1, H, W)
        origin_depths = torch.stack(origin_depth_list, dim=0).float()                 # (6+V, 1, H, W)
        origin_normals = torch.stack(origin_normal_list, dim=0).float()

        gt_images = torch.stack(gt_image_list, dim=0).float()                 # (6+V, 3, H, W)
        gt_alphas = torch.stack(gt_alpha_list, dim=0).float()                 # (6+V, 1, H, W)
        gt_depths = torch.stack(gt_depth_list, dim=0).float()                 # (6+V, 1, H, W)
        gt_normals = torch.stack(gt_normal_list, dim=0).float()
        
        posed_gt_images = torch.stack(posed_gt_image_list, dim=0).float()                 # (6+V, 3, H, W)
        posed_gt_alphas = torch.stack(posed_gt_alpha_list, dim=0).float()                 # (6+V, 1, H, W)
        posed_gt_depths = torch.stack(posed_gt_depth_list, dim=0).float()                 # (6+V, 1, H, W)
        posed_gt_normals = torch.stack(posed_gt_normal_list, dim=0).float()

        normals_smpl = torch.stack(normal_list_smpl, dim=0).float()
        alphas_smpl = torch.stack(alpha_list_smpl, dim=0).float()

        normals_smpl_target = torch.stack(normal_list_smpl_target, dim=0).float()
        alphas_smpl_target = torch.stack(alpha_list_smpl_target, dim=0).float()

        input_view_num = 4 # batch['input_c2ws'].shape[1] 
        batch['origin_images'] = origin_images[:,:input_view_num].detach() 
        batch['origin_alphas'] = origin_alphas[:,:input_view_num].detach()
        batch['origin_depths'] = origin_depths[:,:input_view_num].detach()
        batch['origin_normals'] = origin_normals[:,:input_view_num].detach()

        batch['gt_images'] = gt_images[:,:].detach()
        batch['gt_alphas'] = gt_alphas[:,:].detach()
        batch['gt_depths'] = gt_depths[:,:].detach()
        batch['gt_normals'] = gt_normals[:,:].detach()
        
        batch['posed_gt_images'] = posed_gt_images[:,:].detach()
        batch['posed_gt_alphas'] = posed_gt_alphas[:,:].detach()
        batch['posed_gt_depths'] = posed_gt_depths[:,:].detach()
        batch['posed_gt_normals'] = posed_gt_normals[:,:].detach()

        batch['input_smpl_normals'] = normals_smpl[:,:4].detach()
        batch['input_smpl_alphas'] = alphas_smpl[:,:4].detach()

        batch['input_smpl_normals_target'] = normals_smpl_target[:,:4].detach()
        batch['input_smpl_alphas_target'] = normals_smpl_target[:,:4].detach()
        return batch

    def for_render_gs(self, data):
        c2ws = torch.cat([data['input_c2ws'],data['target_c2ws']],dim=1).float()
        batch_size = c2ws.size(0)

        results = {}
        results['cam_view'] = []
        results['cam_view_proj'] = []
        results['cam_pos'] = []
        results['c2ws'] = []
        results['images_input'] = []
        results['input'] = []
        results['images_input_normals_smpl'] = []
        results['target_normals_smpl'] = []
        results['input_normals_smpl'] = []
        results['new_target_normals_smpl'] = []

        cam_radius = 1.5
        for idx in range(batch_size):

            cam_poses = c2ws[idx]

            # normalized camera feats as in paper (transform the first pose to a fixed position)
            transform = torch.tensor([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, cam_radius], [0, 0, 0, 1]], dtype=torch.float32).cuda() @ torch.inverse(cam_poses[0])
            cam_poses = transform.unsqueeze(0) @ cam_poses  # [V, 4, 4]

            cam_poses[:, :3, 3] *=  1.5 / cam_radius # 1.5 is the default scale

            cam_poses_input = cam_poses[:4].clone()

            images_input = F.interpolate(data['origin_images'][idx][:4].clone(), size=(self.opt.input_size, self.opt.input_size), mode='bilinear', align_corners=False) # [V, C, H, W]
            results['images_input'].append(images_input)
            images_input = TF.normalize(images_input, IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD)

            # build rays for input views
            rays_embeddings = []

            for i in range(4):
                rays_o, rays_d = get_rays(cam_poses_input[i].half(), self.opt.input_size, self.opt.input_size, self.opt.fovy) # [h, w, 3]
                rays_plucker = torch.cat([torch.cross(rays_o.float(), rays_d, dim=-1), rays_d], dim=-1) # [h, w, 6]
                rays_embeddings.append(rays_plucker)

            rays_embeddings = torch.stack(rays_embeddings, dim=0).permute(0, 3, 1, 2).contiguous() # [V, 6, h, w]
            final_input = torch.cat([images_input, rays_embeddings], dim=1) # [V=4, 9, H, W]
            results['input'].append(final_input)
            data['rays_embeddings_4view'] = rays_embeddings
            cam_poses_input_smpl = cam_poses[:4].clone()

            images_input_normals_smpl = F.interpolate(data['input_smpl_normals'][idx][0:4].clone(), size=(self.opt.input_size, self.opt.input_size), mode='bilinear', align_corners=False) # [V, C, H, W]
            results['images_input_normals_smpl'].append(images_input_normals_smpl)
            images_input_normals_smpl = TF.normalize(images_input_normals_smpl, IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD)
            if random.random() < self.opt.prob_grid_distortion:
                images_input_normals_smpl = grid_distortion(images_input_normals_smpl)

            target_normals_smpl = F.interpolate(data['input_smpl_normals_target'][idx][0:4].clone(), size=(self.opt.input_size, self.opt.input_size), mode='bilinear', align_corners=False) # [V, C, H, W]
            results['target_normals_smpl'].append(target_normals_smpl)
            target_normals_smpl = TF.normalize(target_normals_smpl, IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD)
            if random.random() < self.opt.prob_grid_distortion:
                target_normals_smpl = grid_distortion(target_normals_smpl)

            # apply camera jittering (only to input!)
            if random.random() < self.opt.prob_cam_jitter:
                cam_poses_input_smpl = orbit_camera_jitter(cam_poses_input_smpl)

            # build rays for input views
            rays_embeddings_half_size = []
            for i in range(0, 4):
                rays_o, rays_d = get_rays(cam_poses_input_smpl[i], self.opt.input_size, self.opt.input_size, self.opt.fovy) # [h, w, 3]
                rays_plucker = torch.cat([torch.cross(rays_o, rays_d, dim=-1), rays_d], dim=-1) # [h, w, 6]
                rays_embeddings_half_size.append(rays_plucker)
    
            rays_embeddings_half_size = torch.stack(rays_embeddings_half_size, dim=0).permute(0, 3, 1, 2).contiguous() # [V, 6, h, w]
            final_input_normals_smpl = torch.cat([images_input_normals_smpl, rays_embeddings_half_size], dim=1) # [V=4, 9, H, W]
            results['input_normals_smpl'].append(final_input_normals_smpl)
            final_target_normals_smpl = torch.cat([target_normals_smpl, rays_embeddings_half_size], dim=1)
            results['new_target_normals_smpl'].append(final_target_normals_smpl)

            # cam_poses = torch.from_numpy(pose).unsqueeze(0).to(self.device)
            cam_poses[:, :3, 1:3] *= -1 # invert up & forward direction
            
            # cameras needed by gaussian rasterizer
            cam_view = torch.inverse(cam_poses.float()).transpose(1, 2) # [V, 4, 4]
            cam_view_proj = cam_view @ self.proj_matrix.to(device=data['input_c2ws'].device)#.double() # [V, 4, 4]
            cam_pos = - cam_poses[:, :3, 3] # [V, 3]

            results['cam_view'].append(cam_view)
            results['cam_view_proj'].append(cam_view_proj)
            results['cam_pos'].append(cam_pos)
            results['c2ws'].append(cam_poses)

        data['cam_view'] = torch.stack(results['cam_view'])
        data['cam_view_proj'] = torch.stack(results['cam_view_proj'])
        data['cam_pos'] = torch.stack(results['cam_pos'])
        data['c2ws'] = torch.stack(results['c2ws'])
        data['images_input'] = torch.stack(results['images_input'])
        data['input'] = torch.stack(results['input'])

        data['images_input_normals_smpl'] = torch.stack(results['images_input_normals_smpl'])
        data['input_normals_smpl'] = torch.stack(results['input_normals_smpl'])
        data['target_input_smpl'] = torch.stack(results['target_normals_smpl'])
        data['target_normals_smpl'] = torch.stack(results['new_target_normals_smpl'])
        data['images_output'] = data['gt_images']
        data['masks_output'] = data['gt_alphas']
        data['images_output_normals_human'] = data['gt_normals']

        # data['images_output_normals_smpl'] = torch.cat([data['input_smpl_normals'],data['target_smpl_normals']], dim=1)
        # data['masks_output_normals_smpl'] = torch.cat([data['input_smpl_alphas'],data['target_smpl_alphas']], dim=1)

        data['final_input'] = torch.cat([data['input'], data['input_normals_smpl']], dim = 1)  # 1, 8, 9, 512, 512
        
        return data

    # NOTE: Gaussian animation functions
    def gaussian_animation(self, Apose_gaussians, Apose_param, target_param, data=None):
        """
        Input:
            tpose_gaussians: - Refined Gaussians from LGM [b, 14, n]
            Apose_vertices:  - T-pose SMPL vertices [b, 10475, 3]
            Apose_parameters
            target_parameters

        Return:
            transposed_gaussians: Reposed Gaussian Avatar [b, 14, n]
        """        
        bsz = Apose_gaussians.shape[0]

        Tpose_gaussian_bsz = []
        Posed_gaussian_bsz = []
        for item in range(bsz):
            
            Apose_gaussians = self.gs.prune(Apose_gaussians)
            
            gs_xyz = Apose_gaussians[item, :, :3] #[524288,3]

            # Transform to T-pose:
            model_forward_params_A = dict(
                betas=Apose_param['betas'][item],
                global_orient=Apose_param['global_orient'][item],
                body_pose=Apose_param['body_pose'][item],
                left_hand_pose=Apose_param['left_hand_pose'][item],
                right_hand_pose=Apose_param['right_hand_pose'][item],
                jaw_pose=Apose_param['jaw_pose'][item],
                leye_pose=Apose_param['leye_pose'][item],
                reye_pose=Apose_param['reye_pose'][item],
                expression=Apose_param['expression'][item],
                return_verts=True,
                return_joint_transformation=False,
                return_vertex_transformation=False)

            smpl_out_tpose = self.smpl_model(**model_forward_params_A)

            Apose_mesh = trimesh.Trimesh(smpl_out_tpose.vertices.squeeze(0).float().detach().cpu().numpy(), self.smpl_model.faces)
            
            # To obtain the translation and scales
            Apose_mesh, translation, scales_inv = new_mesh(Apose_mesh, data['ratio'].item())
        
            model_forward_params_A = dict(
                betas=Apose_param['betas'][item],
                global_orient=Apose_param['global_orient'][item],
                body_pose=Apose_param['body_pose'][item],
                left_hand_pose=Apose_param['left_hand_pose'][item],
                right_hand_pose=Apose_param['right_hand_pose'][item],
                jaw_pose=Apose_param['jaw_pose'][item],
                leye_pose=Apose_param['leye_pose'][item],
                reye_pose=Apose_param['reye_pose'][item],
                expression=Apose_param['expression'][item],
                return_verts=True,
                return_joint_transformation=True,
                return_vertex_transformation=True,
                manual_scale = scales_inv,
                manual_transl = translation) 
            smpl_out_tpose = self.smpl_model(**model_forward_params_A)
            
            # Find nearest smpl vertex
            Apose_vertices = torch.from_numpy(Apose_mesh.vertices).float().to(Apose_gaussians.device)
            smpl_tree = cKDTree(Apose_vertices.cpu().numpy()) #[10xxx,3]
            _, idx = smpl_tree.query(gs_xyz.detach().cpu().numpy(), k=3) #[116308,3]
            
            # t-pose for gaussians
            rot_mat_t = (smpl_out_tpose.vertex_transformation.detach()[0][idx[:, 0]]).float()
            homo_coord = torch.ones_like(gs_xyz)[..., :1]
            tpose_gs_xyz = torch.inverse(rot_mat_t) @ torch.cat([gs_xyz, homo_coord],dim=1).unsqueeze(-1)
            tpose_gs_xyz = tpose_gs_xyz[:, :3, 0]
            
            tpose_gaussians = torch.cat([tpose_gs_xyz, Apose_gaussians[item, :, 3:]], dim = -1)
            # self.gs.save_ply(self.tpose_gaussians.detach().cpu().unsqueeze(0), '/hpc2hdd/home/gzhang292/nanjie/project3/LBS_LGM_nanjie/vis/tpose.ply')
            
            # Repose
            model_forward_params_re = dict(
                betas=Apose_param['betas'][item],
                global_orient=target_param['global_orient'][item],
                body_pose=target_param['body_pose'][item],
                left_hand_pose=target_param['left_hand_pose'][item],
                right_hand_pose=target_param['right_hand_pose'][item],
                jaw_pose=target_param['jaw_pose'][item],
                leye_pose=target_param['leye_pose'][item],
                reye_pose=target_param['reye_pose'][item],
                expression=target_param['expression'][item],
                return_verts=False,
                return_joint_transformation=True,
                return_vertex_transformation=True,
                manual_scale = scales_inv,
                manual_transl = translation)
            
            smpl_out_repose = self.smpl_model(**model_forward_params_re)
            
            rot_mat_da = (smpl_out_repose.vertex_transformation.detach()[0][idx[:, 0]]).float()
            rot_gs_xyz = rot_mat_da @ torch.cat([tpose_gs_xyz, homo_coord], dim=-1).unsqueeze(-1)
            rot_gs_xyz = rot_gs_xyz[:, :3, 0]

            # concat the gaussians 
            rot_gs_xyz,_,_ = new_verts(rot_gs_xyz, data['ratio'].item())
            transposed_gaussians = torch.cat([rot_gs_xyz, tpose_gaussians[:, 3:]], dim = -1)

            Tpose_gaussian_bsz.append(tpose_gaussians.unsqueeze(0))
            Posed_gaussian_bsz.append(transposed_gaussians.unsqueeze(0))

        tpose_gaussians = torch.cat(Tpose_gaussian_bsz, dim = 0)
        transposed_gaussians = torch.cat(Posed_gaussian_bsz, dim = 0)
        
        return tpose_gaussians, transposed_gaussians #[1, n, 14]

    
    # Video output
    def visualization(self, gaussians, save_path_dir, sample_id, save_images = False):
        from kiui.cam import orbit_camera
        import imageio
        import cv2
        import os
        images = []
        mask_list = []
        save_path = os.path.join(save_path_dir, f'pred_{sample_id}_normal.mp4',)

        tan_half_fov = np.tan(0.5 * np.deg2rad(self.opt.fovy))
        proj_matrix = torch.zeros(4, 4, dtype=torch.float32, device='cuda')
        proj_matrix[0, 0] = 1 / tan_half_fov
        proj_matrix[1, 1] = 1 / tan_half_fov
        proj_matrix[2, 2] = (self.opt.zfar + self.opt.znear) / (self.opt.zfar - self.opt.znear)
        proj_matrix[3, 2] = - (self.opt.zfar * self.opt.znear) / (self.opt.zfar - self.opt.znear)
        proj_matrix[2, 3] = 1

        azimuth = np.arange(0, 360, 2, dtype=np.int32)
        for azi in azimuth:
            cam_poses = torch.from_numpy(orbit_camera(0, azi, radius=self.opt.cam_radius, opengl=True)).unsqueeze(0).to('cuda')
            cam_poses[:, :3, 3] *=  1.5 / self.opt.cam_radius # 1.5 is the default scale

            cam_poses[:, :3, 1:3] *= -1 # invert up & forward direction
            
            # cameras needed by gaussian rasterizer
            cam_view = torch.inverse(cam_poses).transpose(1, 2) # [V, 4, 4]
            cam_view_proj = cam_view @ proj_matrix # [V, 4, 4]
            cam_pos = - cam_poses[:, :3, 3] # [V, 3]

            out = self.gs.render(gaussians, cam_view.unsqueeze(0), cam_view_proj.unsqueeze(0), cam_pos.unsqueeze(0), scale_modifier=1)
            image = out['image']
            mask = out['mask']
            mask_list.append((mask.detach().squeeze(1).permute(0,2,3,1).contiguous().float().cpu().numpy() * 255).astype(np.uint8))
            images.append((image.detach().squeeze(1).permute(0,2,3,1).contiguous().float().cpu().numpy() * 255).astype(np.uint8))

        images_video = np.concatenate(images, axis=0)
        imageio.mimwrite(save_path, images_video, fps = 30)

        if save_images:
            os.makedirs(f'{save_path_dir}/images/{sample_id}', exist_ok=True)
            for frame in range(len(images)):
                if frame % 45 == 0:
                    frame_mask = mask_list[frame]
                    cv2.imwrite(f'{save_path_dir}/images/{sample_id}/{frame}.png', np.concatenate([images[frame][0][:,:, ::-1], frame_mask[0]], axis=-1))


    def get_transform_params_torch(smpl, params, rot_mats=None, correct_Rs=None):
        """
        obtain the transformation parameters for linear blend skinning
        """
        v_template = smpl['v_template']

        # add shape blend shapes
        shapedirs = smpl['shapedirs']
        betas = params['shapes']
        # v_shaped = v_template[None] + torch.sum(shapedirs[None] * betas[:,None], axis=-1).float()
        v_shaped = v_template[None] + torch.sum(shapedirs[None][...,:betas.shape[-1]] * betas[:,None], axis=-1).float()

        if rot_mats is None:
            # add pose blend shapes
            poses = params['poses'].reshape(-1, 3)
            # bs x 24 x 3 x 3
            rot_mats = batch_rodrigues_torch(poses).view(params['poses'].shape[0], -1, 3, 3)

            if correct_Rs is not None:
                rot_mats_no_root = rot_mats[:, 1:]
                rot_mats_no_root = torch.matmul(rot_mats_no_root.reshape(-1, 3, 3), correct_Rs.reshape(-1, 3, 3)).reshape(-1, rot_mats.shape[1]-1, 3, 3)
                rot_mats = torch.cat([rot_mats[:, 0:1], rot_mats_no_root], dim=1)

        # obtain the joints
        joints = torch.matmul(smpl['J_regressor'][None], v_shaped) # [bs, 24 ,3]

        # obtain the rigid transformation
        parents = smpl['kintree_table'][0]
        A = get_rigid_transformation_torch(rot_mats, joints, parents)

        # apply global transformation
        R = params['R'] 
        Th = params['Th'] 
        return A, R, Th, joints


    def forward(self, data, stage_1_model, step_ratio=1, lpips_weight_additon=0, training=False):
        # data: output of the dataloader
        # return: loss

        results = {}
        loss = 0

        data = self.prepare_render_images(data) # render the targetpose

        data = self.for_render_gs(data)

        to_stage1_data = data

        before_act = None
        smpl_mid_block = None

 
        for key, value in to_stage1_data.items():
            if isinstance(value, torch.Tensor):
                device = value.device
                break 

        # put the stage_1_model to the correct device
        stage_1_model = stage_1_model.to(device)

        with torch.no_grad():
            out_1, data_1 = stage_1_model(to_stage1_data, training=False)

        gaussians_stage1 = out_1['A_gaussians']


        # Firstly Animate the Gaussian Avatar with LBS
        Apose_param = data['param_apose']
        target_param = data['param_targetpose']

        tpose_gaussians, gaussians = self.gaussian_animation(gaussians_stage1, Apose_param=Apose_param, target_param=target_param, data = data)

        # always use white bg
        bg_color = torch.ones(3, dtype=torch.float32, device=gaussians_stage1.device)
        stage_1_output = self.gs.render(gaussians, data['cam_view'], data['cam_view_proj'], data['cam_pos'], data['c2ws'], bg_color=bg_color, outputsize=self.opt.output_size)

        stage_2_input_images = stage_1_output['image']  # 1, 12, 3, 512, 512
        data['stage2_input'] = stage_2_input_images
        image_batch = TF.normalize(stage_2_input_images, IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD)   # 1, 12, 3, 512, 512
        input_image = torch.cat([image_batch[:,:4], data['target_normals_smpl'][:,:,3:]], dim=2) # [V=1, 4, 9, H, W] # 1,4,3,512,512
        images = torch.cat([input_image, data['target_normals_smpl']], dim=1)   # 1, 8, 3, 512, 512

        final_gaussians, final_gaussians_smpl = self.forward_gaussians(images, before_act, smpl_mid_block, data, front_normals=None, training=training) # [B, N, 14]
        
        results['gaussians'] = gaussians
        results['stage_2_input_images'] = stage_2_input_images
        # always use white bg
        bg_color = torch.ones(3, dtype=torch.float32, device=gaussians.device)
        
        # use the other views for rendering and supervision
        results = self.gs.render(final_gaussians, data['cam_view'], data['cam_view_proj'], data['cam_pos'], data['c2ws'], bg_color=bg_color, outputsize=self.opt.output_size)

        pred_images = results['image'] # [B, V, C, output_size, output_size]
        pred_alphas = results['alpha'] # [B, V, 1, output_size, output_size]
        pred_depths = results['depth'] # [B, V, 1, output_size, output_size]

        gt_images = data['posed_gt_images'] # [B, V, 3, output_size, output_size], ground-truth novel views
        gt_masks = data['posed_gt_alphas'] # [B, V, 1, output_size, output_size], ground-truth masks

        results['images_pred'] = pred_images
        results['alphas_pred'] = pred_alphas

        gt_images = gt_images * gt_masks + bg_color.view(1, 1, 3, 1, 1) * (1 - gt_masks)
        loss_mask = F.mse_loss(pred_alphas, gt_masks)
        loss_mse = F.mse_loss(pred_images, gt_images) + loss_mask
        loss = loss + loss_mse # + depth_loss
        
        try:
            gt_indices = torchvision.ops.masks_to_boxes(gt_masks.view(-1, gt_masks.shape[-2], gt_masks.shape[-1]))
            pred_indices = torchvision.ops.masks_to_boxes(pred_alphas.view(-1, pred_alphas.shape[-2], pred_alphas.shape[-1]))
            bx1y1x2y2 = torch.cat([gt_indices, pred_indices], dim=0)
            x1_min = int(bx1y1x2y2[:,0].min())
            y1_min = int(bx1y1x2y2[:,1].min())
            x2_max = int(bx1y1x2y2[:,2].max())
            y2_max = int(bx1y1x2y2[:,3].max())

            temp_gt_images = gt_images[...,y1_min:y2_max,x1_min:x2_max]
            temp_pred_images =  pred_images[...,y1_min:y2_max,x1_min:x2_max]
        except:
            temp_gt_images = gt_images
            temp_pred_images = pred_images
        
        loss_lpips_detail = 0

        if self.opt.lambda_lpips > 0:
            loss_lpips = self.lpips_loss(
                # gt_images.view(-1, 3, self.opt.output_size, self.opt.output_size) * 2 - 1,
                # pred_images.view(-1, 3, self.opt.output_size, self.opt.output_size) * 2 - 1,
                # downsampled to at most 256 to reduce memory cost
                F.interpolate(temp_gt_images[:,:].reshape(-1, 3, temp_gt_images.shape[-2], temp_gt_images.shape[-1]) * 2 - 1, (256, 256), mode='bilinear', align_corners=False), 
                F.interpolate(temp_pred_images[:,:].reshape(-1, 3, temp_gt_images.shape[-2], temp_gt_images.shape[-1]) * 2 - 1, (256, 256), mode='bilinear', align_corners=False),
            ).mean()

            B,V,C,H,W = temp_gt_images.shape

            # ssim = self.rgb_metrics(
            #     temp_gt_images.reshape(B*V, C, H, W),
            #     temp_pred_images.reshape(B*V, C, H, W),
            # )
            results['loss_lpips'] = loss_lpips#+ (1-ssim)/2 #+ loss_lpips_front
            # loss = loss + self.opt.lambda_lpips * loss_lpips
            loss = loss + (self.opt.lambda_lpips + lpips_weight_additon) * loss_lpips

        results['loss_mask'] = loss_mask
        results['loss'] = loss

        results['stage_2_input'] = image_batch[:,:4]

        # metric
        with torch.no_grad():
            psnr = -10 * torch.log10(torch.mean((pred_images.detach() - gt_images) ** 2))
            results['psnr'] = psnr
            results['ssim'] = psnr
            results['lpips'] = psnr

            if training==True:               
                B,V,C,H,W = gt_images.shape
                psnr, ssim, lpips = self.rgb_metrics(
                    gt_images.reshape(B*V, C, H, W),
                    pred_images.reshape(B*V, C, H, W).detach(),
                )
                results['psnr'] = psnr
                results['ssim'] = ssim
                results['lpips'] = lpips

        results['gaussians'] = gaussians
        results['smpl_gau'] = gaussians #self.sparsify(gaussians_smpl)

        results['images_pred_normals'] = pred_images

        return results, data