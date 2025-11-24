import tyro
import time
import random

import torch
from core.options import AllConfigs
from core.models import LGM_1, LGM_2
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
# os.environ['NCCL_P2P_DISABLE']='1' if cuda error occurs, open this line
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
    model = LGM_2(opt)
    model.config_conv_layer_smpl()

    # resume
    if opt.resume is not None:
        if opt.resume.endswith('safetensors'):
            ckpt = load_file(opt.resume, device='cpu')
        else:
            ckpt = torch.load(opt.resume, map_location='cpu')
        
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

    # model.config_conv_layer_smpl()
    opt_one = tyro.cli(AllConfigs)


    # path to your pre-trained stage 1 model
    stage_1_model_path = '/hpc2hdd/home/jshu704/Animation/FastAnimate/model.safetensors'

    stage_one_model = LGM_1(opt_one) 
    stage_one_model.config_conv_layer_smpl()
    ckpt = load_file(stage_1_model_path)    
    
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

    stage_one_model.eval()  # stage_1 used for inference
    for param in stage_one_model.parameters():
        param.requires_grad = False

    params = []
    params.append({'params': [p for p in model.parameters()], 'lr': opt.lr})



    # data
    if opt.data_mode == 's3':
        from core.provider_objvarse_train import ObjaverseDataset as Dataset
    else:
        raise NotImplementedError

    train_dataset = Dataset(opt, training=True)

    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=opt.batch_size,
        shuffle=True,
        num_workers=opt.num_workers,
        pin_memory=True,
        drop_last=True,
    )

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

   
    total_steps = opt.num_epochs * len(train_dataloader)
    pct_start = 300 / total_steps
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=opt.lr, total_steps=total_steps, pct_start=pct_start)

    # accelerate
    model, optimizer, train_dataloader, test_dataloader, scheduler = accelerator.prepare(
        model, optimizer, train_dataloader, test_dataloader, scheduler
    )

    training_main_loss_list = []
    training_lpips_loss_list = []
    training_mask_loss_list = []

    trainset_psnr_list = []
    trainset_ssim_list = []
    trainset_lpips_list = []

    # loop
    for epoch in range(opt.num_epochs):
        # train
        model.train()
        total_loss = 0
        total_loss_mask = 0
        total_loss_lpips = 0

        total_psnr = 0
        total_ssim = 0
        total_lpips = 0

        def calculate_addition(i):
            if i < 10:
                addition = 1
            else:
                addition = 1
            return addition
        lpips_weight_additon = calculate_addition(epoch)

        for i, data in enumerate(train_dataloader):
            with accelerator.accumulate(model):
                optimizer.zero_grad()

                step_ratio = (epoch + i / len(train_dataloader)) / opt.num_epochs

                out, data  = model(data, stage_one_model, step_ratio, lpips_weight_additon, training = True)

                loss = out['loss']
                psnr = out['psnr']
                ssim = out['ssim']
                lpips = out['lpips']

                accelerator.backward(loss)
                loss_lpips = out['loss_lpips']
                loss_mask = out['loss_mask']

                # gradient clipping
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(model.parameters(), opt.gradient_clip)

                optimizer.step()
                scheduler.step()

                total_loss += loss.detach()
                total_psnr += psnr.detach()
                total_ssim += ssim.detach()
                total_lpips += lpips.detach()
                total_loss_lpips += loss_lpips.detach()
                total_loss_mask += loss_mask.detach()

            if accelerator.is_main_process:
                # logging
                if i % 100 == 0:
                    mem_free, mem_total = torch.cuda.mem_get_info()    
                    print(f"[INFO] {i}/{len(train_dataloader)} mem: {(mem_total-mem_free)/1024**3:.2f}/{mem_total/1024**3:.2f}G lr: {scheduler.get_last_lr()[0]:.7f} step_ratio: {step_ratio:.4f}")
                    print(f"loss: {loss.item():.6f} loss_lpips: {loss_lpips.item():.6f} loss_mask: {loss_mask.item():.6f} lpips_weight_additon: {lpips_weight_additon:.6f}")
                
                # save log images
                if i % 100 == 0:
                    
                    images_input = data['stage2_input'][:,:4].detach().cpu().numpy() # [B, V, 3, output_size, output_size]
                    images_input = images_input.transpose(0, 3, 1, 4, 2).reshape(-1, images_input.shape[1] * images_input.shape[3], 3) # [B*output_size, V*output_size, 3]
                    kiui.write_image(f'{opt.workspace}/train/train_input_images_{epoch}_{i}_stage2.jpg', images_input)

                    images_input = data['target_input_smpl'].detach().cpu().numpy() # [B, V, 3, output_size, output_size]
                    images_input = images_input.transpose(0, 3, 1, 4, 2).reshape(-1, images_input.shape[1] * images_input.shape[3], 3) # [B*output_size, V*output_size, 3]
                    kiui.write_image(f'{opt.workspace}/train/train_input_images_{epoch}_{i}_stage2_smpl.jpg', images_input)

                    gt_images = data['posed_gt_images'].detach().cpu().numpy() # [B, V, 3, output_size, output_size]
                    gt_images = gt_images.transpose(0, 3, 1, 4, 2).reshape(-1, gt_images.shape[1] * gt_images.shape[3], 3) # [B*output_size, V*output_size, 3]
                    kiui.write_image(f'{opt.workspace}/train/train_gt_images_{epoch}_{i}_stage2.jpg', gt_images)

                    pred_images = out['images_pred'].detach().cpu().numpy() # [B, V, 3, output_size, output_size]
                    pred_images = pred_images.transpose(0, 3, 1, 4, 2).reshape(-1, pred_images.shape[1] * pred_images.shape[3], 3)
                    kiui.write_image(f'{opt.workspace}/train/train_pred_images_{epoch}_{i}.jpg', pred_images)

                    pred_images = out['images_pred'].detach().cpu().numpy() # [B, V, 3, output_size, output_size]
                    pred_images = pred_images.transpose(0, 3, 1, 4, 2).reshape(-1, pred_images.shape[1] * pred_images.shape[3], 3)
                    kiui.write_image(f'{opt.workspace}/train/train_pred_images_{epoch}_{i}.jpg', pred_images)


                    pred_images = out['images_pred_normals'].detach().cpu().numpy() # [B, V, 3, output_size, output_size]
                    pred_images = pred_images.transpose(0, 3, 1, 4, 2).reshape(-1, pred_images.shape[1] * pred_images.shape[3], 3)
                    kiui.write_image(f'{opt.workspace}/train/train_pred_images_{epoch}_{i}_normals.jpg', pred_images)

        total_loss = accelerator.gather_for_metrics(total_loss).mean()
        total_psnr = accelerator.gather_for_metrics(total_psnr).mean()
        total_ssim = accelerator.gather_for_metrics(total_ssim).mean()
        total_lpips = accelerator.gather_for_metrics(total_lpips).mean()
        total_loss_lpips = accelerator.gather_for_metrics(total_loss_lpips).mean()
        total_loss_mask = accelerator.gather_for_metrics(total_loss_mask).mean()


        if accelerator.is_main_process:
            total_loss /= len(train_dataloader)
            total_psnr /= len(train_dataloader)
            total_ssim /= len(train_dataloader)
            total_lpips /= len(train_dataloader)
            total_loss_lpips /= len(train_dataloader)
            total_loss_mask /= len(train_dataloader)

            accelerator.print(f"[train] epoch: {epoch} loss: {total_loss.item():.6f} loss_lpips: {total_loss_lpips.item():.6f} loss_mask: {total_loss_mask.item():.6f} psnr: {total_psnr.item():.4f} ssim: {total_ssim.item():.4f} lpips: {total_lpips.item():.4f}")

            x_epoch_num = [x for x in range(0, int(epoch+1))]
            training_main_loss_list.append(total_loss.item())
            os.makedirs(f'{opt.workspace}/loss_figure', exist_ok=True)
            plt.plot(x_epoch_num, training_main_loss_list, '-')
            plt.title('Training Main Loss Line Chart')
            plt.xlabel('X-axis Epoch_num')
            plt.ylabel('Y-axis Training Main Loss')

            plt.savefig(f'{opt.workspace}/loss_figure/training_main_loss_line_{epoch}.jpg')
            plt.cla()

            training_lpips_loss_list.append(total_loss_lpips.item())

            plt.plot(x_epoch_num, training_lpips_loss_list, '-')
            plt.title('Training Lpips Loss Line Chart')
            plt.xlabel('X-axis Epoch_num')
            plt.ylabel('Y-axis Training Lpips Loss')

            plt.savefig(f'{opt.workspace}/loss_figure/training_lpips_loss_line_{epoch}.jpg')
            plt.cla()

            training_mask_loss_list.append(total_loss_mask.item())

            plt.plot(x_epoch_num, training_mask_loss_list, '-')
            plt.title('Training Mask Loss Line Chart')
            plt.xlabel('X-axis Epoch_num')
            plt.ylabel('Y-axis Training Mask Loss')

            trainset_psnr_list.append(total_psnr.item())
            plt.plot(x_epoch_num, trainset_psnr_list, '-')
            plt.title('Trainset Psnr Line Chart')
            plt.xlabel('X-axis Epoch_num')
            plt.ylabel('Y-axis Trainset Psnr')

            plt.savefig(f'{opt.workspace}/loss_figure/trainset_psnr_line_{epoch}.jpg')
            plt.cla()

            trainset_ssim_list.append(total_ssim.item())
            plt.plot(x_epoch_num, trainset_ssim_list, '-')
            plt.title('Trainset Ssim Line Chart')
            plt.xlabel('X-axis Epoch_num')
            plt.ylabel('Y-axis Trainset Ssim')

            trainset_lpips_list.append(total_lpips.item())
            plt.plot(x_epoch_num, trainset_lpips_list, '-')
            plt.title('Trainset Lpips Line Chart')
            plt.xlabel('X-axis Epoch_num')
            plt.ylabel('Y-axis Trainset Lpips')

        # eval
        with torch.no_grad():
            model.eval()
            total_psnr = 0
            total_ssim = 0
            total_lpips = 0
            total_psnr_face = 0

            for i, data in enumerate(test_dataloader):

                out, data = model(data, stage_one_model)
    
                psnr = out['psnr']
                ssim = out['ssim']
                lpips = out['lpips']

                # save some images
                if accelerator.is_main_process:
                    gt_images = data['images_output'].detach().cpu().numpy() # [B, V, 3, output_size, output_size]
                    gt_images = gt_images.transpose(0, 3, 1, 4, 2).reshape(-1, gt_images.shape[1] * gt_images.shape[3], 3) # [B*output_size, V*output_size, 3]
                    kiui.write_image(f'{opt.workspace}/eval/eval_gt_images_{epoch}_{i}.jpg', gt_images)

                    pred_images = out['images_pred'].detach().cpu().numpy() # [B, V, 3, output_size, output_size]
                    pred_images = pred_images.transpose(0, 3, 1, 4, 2).reshape(-1, pred_images.shape[1] * pred_images.shape[3], 3)
                    kiui.write_image(f'{opt.workspace}/eval/eval_pred_images_{epoch}_{i}.jpg', pred_images)

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

                    cam_poses[:, :3, 1:3] *= -1 # invert up & forward d/home/hdd/zhanggangjian/2_Avatar/LGM3/scriptsirection
                    
                    # cameras needed by gaussian rasterizer
                    cam_view = torch.inverse(cam_poses).transpose(1, 2) # [V, 4, 4]
                    cam_view_proj = cam_view @ proj_matrix # [V, 4, 4]
                    cam_pos = - cam_poses[:, :3, 3] # [V, 3]

                    image = model.module.gs.render(out['gaussians'], cam_view.unsqueeze(0), cam_view_proj.unsqueeze(0), cam_pos.unsqueeze(0), cam_poses.unsqueeze(0), scale_modifier=1)['image']
                    images.append((image.squeeze(1).permute(0,2,3,1).contiguous().float().cpu().numpy() * 255).astype(np.uint8))

                images = np.concatenate(images, axis=0)
                os.makedirs(os.path.join(opt.workspace, 'eval'), exist_ok=True)
                imageio.mimwrite(os.path.join(opt.workspace, f'eval/render_pred_{epoch}_{i}' + '.mp4'), images, fps=30)


        # checkpoint
        if epoch % 10 == 0 or epoch == opt.num_epochs - 1:
            accelerator.wait_for_everyone()
            accelerator.save_model(model, f'{opt.workspace}/checkpoints')
            print(f"Model_{epoch} saved")

if __name__ == "__main__":
    main()