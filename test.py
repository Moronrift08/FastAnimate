import tyro
import time
import random

import torch
from core.options import AllConfigs
from core.models_test import LGM_1, LGM_2
from accelerate import Accelerator, DistributedDataParallelKwargs
from safetensors.torch import load_file

import kiui
import cv2
import numpy as np
import tqdm
import imageio
from kiui.cam import orbit_camera
import os
import matplotlib.pyplot as plt
import sys
import trimesh
# os.environ['NCCL_P2P_DISABLE']='1'   if cuda error occurs, open this line
import copy



def main():    

    if len(sys.argv) <= 1:
        sys.argv = ["main.py", "big", "--workspace", "workspace_final_debug"]

    opt = tyro.cli(AllConfigs)
    import trimesh

    accelerator = Accelerator(
        mixed_precision=opt.mixed_precision,
        gradient_accumulation_steps=opt.gradient_accumulation_steps,
    )

    # model
    trained_model_path = './path_to_your_stage_2_safetensors'
    model = LGM_2(opt)
    model.config_conv_layer_smpl()

    # resume
    if opt.resume is not None:
        if opt.resume.endswith('safetensors'):
            ckpt = load_file(trained_model_path, device='cpu')
        else:
            ckpt = torch.load(trained_model_path, map_location='cpu')
        
        attn_names = []
        attn_shapes = []
        for key in ckpt:
            if 'attn.qkv.weight' in key:
                attn_names.append(key)
                attn_shapes.append(ckpt[key].shape)

        state_dict = model.state_dict()
        for k, v in ckpt.items():
            if k in state_dict: 
                if state_dict[k].shape == v.shape:
                    state_dict[k].copy_(v)
                else:
                    accelerator.print(f'[WARN] mismatching shape for param {k}: ckpt {v.shape} != model {state_dict[k].shape}, ignored.')
            else:
                accelerator.print(f'[WARN] unexpected param {k}: {v.shape}')
    opt_one = tyro.cli(AllConfigs)


    model_path = './path_to_your_stage_1_safetensors'

    stage_one_model = LGM_1(opt_one)
    stage_one_model.config_conv_layer_smpl()
    ckpt = load_file(model_path)
    
    one_attn_names = []
    one_attn_shapes = []
    for key in ckpt:
        if 'attn.qkv.weight' in key:
            one_attn_names.append(key)
            one_attn_shapes.append(ckpt[key].shape)

    state_dict_one = stage_one_model.state_dict()
    for k, v in ckpt.items():
        if k in state_dict_one: 
            if state_dict_one[k].shape == v.shape:
                state_dict_one[k].copy_(v)
            else:
                accelerator.print(f'[WARN] mismatching shape for param {k}: ckpt {v.shape} != model {state_dict_one[k].shape}, ignored.')
        else:
            accelerator.print(f'[WARN] unexpected param {k}: {v.shape}')
    stage_one_model.eval()
    for param in stage_one_model.parameters():
        param.requires_grad = False

    params = []
    params.append({'params': [p for p in model.parameters()], 'lr': opt.lr})

    # data
    if opt.data_mode == 's3':
        from core.provider_objvarse_LBS_xhuman_new_cur import ObjaverseDataset as Dataset
    else:
        raise NotImplementedError

    test_dataset = Dataset(opt, training=False)
    test_dataloader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
        drop_last=False,
    )

     # optimizer
    optimizer = torch.optim.AdamW(params, weight_decay=0.05, betas=(0.9, 0.95))

    # scheduler (per-iteration)
    # scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=3000, eta_min=1e-6)
    total_steps = opt.num_epochs * len(test_dataloader)
    # pct_start = 300 / total_steps
    pct_start = 0.15
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=opt.lr, total_steps=total_steps, pct_start=pct_start)
    # print("1")
    # accelerate
    model, optimizer, test_dataloader, scheduler = accelerator.prepare(
        model, optimizer, test_dataloader, scheduler
    )
    # print("2")

    # eval
    with torch.no_grad():
        model.eval()
        total_psnr = 0
        total_ssim = 0
        total_lpips = 0
        total_psnr_face = 0

        for i, data in enumerate(test_dataloader):

            out, data = model(data, stage_one_model)

            # psnr = out['psnr']
            # ssim = out['ssim']
            # lpips = out['lpips']

            path = f'./{opt.workspace}/{i:03d}'
            os.makedirs(path, exist_ok=True)


            # save some images
            if accelerator.is_main_process:

                gt_images = data['posed_gt_images'].detach().cpu().numpy() # [B, V, 3, output_size, output_size]
                # gt_images = gt_images.transpose(0, 3, 1, 4, 2).reshape(-1, gt_images.shape[1] * gt_images.shape[3], 3) # [B*output_size, V*output_size, 3]
                gt_images = gt_images[0]  # [12, 3, 1024, 1024]
                for j in range(gt_images.shape[0]):  # 12
                    img = gt_images[j].transpose(1, 2, 0)  # [3, 1024, 1024] -> [1024, 1024, 3]
                    kiui.write_image(f'{opt.workspace}/{i:03d}/gt_images_{i}_{j}.jpg', img)

                pred_images = out['images_pred'].detach().cpu().numpy() # [B, V, 3, output_size, output_size]
                # pred_images = pred_images.transpose(0, 3, 1, 4, 2).reshape(-1, pred_images.shape[1] * pred_images.shape[3], 3)
                print("pred_images shape:", pred_images.shape)
                pred_images = pred_images[0]  # [12, 3, 1024, 1024]
                for j in range(pred_images.shape[0]):  # 12
                    img = pred_images[j].transpose(1, 2, 0)  # [3, 1024, 1024] -> [1024, 1024, 3]
                    kiui.write_image(f'{opt.workspace}/{i:03d}/pred_images_{i}_{j}.jpg', img)
            images = []
            elevation = 0

            tan_half_fov = np.tan(0.5 * np.deg2rad(opt.fovy))
            proj_matrix = torch.zeros(4, 4, dtype=torch.float32, device='cuda')
            proj_matrix[0, 0] = 1 / tan_half_fov
            proj_matrix[1, 1] = 1 / tan_half_fov
            proj_matrix[2, 2] = (opt.zfar + opt.znear) / (opt.zfar - opt.znear)
            proj_matrix[3, 2] = - (opt.zfar * opt.znear) / (opt.zfar - opt.znear)
            proj_matrix[2, 3] = 1

            azimuth = np.arange(0, 360, 2, dtype=np.int32)
            elevation_offset = 0
            azi_offset = 0
            radius_offset = 0
            for azi in azimuth:
                
                cam_poses = torch.from_numpy(orbit_camera(elevation + elevation_offset, azi+azi_offset, radius=opt.cam_radius+radius_offset, opengl=True)).unsqueeze(0).to('cuda')

                cam_poses[:, :3, 3] *=  1.5 / opt.cam_radius # 1.5 is the default scale

                cam_poses[:, :3, 1:3] *= -1
                
                # cameras needed by gaussian rasterizer
                cam_view = torch.inverse(cam_poses).transpose(1, 2) # [V, 4, 4]
                cam_view_proj = cam_view @ proj_matrix # [V, 4, 4]
                cam_pos = - cam_poses[:, :3, 3] # [V, 3]

                image = model.gs.render(out['gaussians'], cam_view.unsqueeze(0), cam_view_proj.unsqueeze(0), cam_pos.unsqueeze(0), cam_poses.unsqueeze(0), scale_modifier=1)['image']
                images.append((image.squeeze(1).permute(0,2,3,1).contiguous().float().cpu().numpy() * 255).astype(np.uint8))

            images = np.concatenate(images, axis=0)
            os.makedirs(os.path.join(opt.workspace, 'demo'), exist_ok=True)
            imageio.mimwrite(os.path.join(opt.workspace, f'{i:03d}/render_pred_{i}' + '.mp4'), images, fps=30)
            model.gs.save_ply(out['gaussians'], os.path.join(opt.workspace, f'{i:03d}/gaussian_pred_{i}' + '.ply'))
            model.gs.save_ply(out['final_gaussians'], os.path.join(opt.workspace, f'{i:03d}/final_gaussian_pred_{i}' + '.ply'))
            model.gs.save_ply(out['smpl_gau'], os.path.join(opt.workspace, f'{i:03d}/animated_gaussian_{i}' + '.ply'))


if __name__ == "__main__":
    main()