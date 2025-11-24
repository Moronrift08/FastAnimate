## Installation

To deploy and run FastAnimate, run the following scripts:
```
conda create -n fastanimate
pip install -r requirements.txt
conda activate fastanimate
```

Then, download and compile ```diff-gaussian-rasterization```, ```kaolin``` and ```nvidiffrast```.

## Download models and data 

- SMPL/SMPL-X model: register and download [SMPL](https://smpl.is.tue.mpg.de/) and [SMPL-X](https://smpl-x.is.tue.mpg.de/), and put these files in ```/smpl_related/``` to replace the pseudo files. The folder should have the following structure:
```
smpl_related
 └── models
     └──smpl
       ├── ...
       ├── ...
       └── ...
     └──smplx
       ├── ...
       ├── ...
       └── ...
 └── smpl_data
   ├── ...
   ├── ...
   └── ...
```

- Data: register and download [X-Humans](https://github.com/Skype-line/X-Avatar?tab=readme-ov-file)


### Train

```
CUDA_VISIBLE_DEVICES=0,1,2,3 nohup accelerate launch --config_file acc_configs/gpu4.yaml train.py  big --workspace workspace_train --resume ./path_to_your_stage_1_models
```

### Test

```
CUDA_VISIBLE_DEVICES=0 python test.py big --resume ./path_to_your_stage_2_models --workspace workspace_test
```

## Acknowledgements

This project is built on source codes shared by [LGM](https://github.com/3DTopia/LGM), [SiTH](https://github.com/SiTH-Diffusion/SiTH), [MVDream](https://github.com/bytedance/MVDream) and [MultiGo](https://github.com/gzhang292/MultiGO).
