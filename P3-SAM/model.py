import os
import sys
import torch 
import torch.nn as nn 
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'XPart/partgen'))
from models import sonata
from utils.misc import smart_load_model

'''
This is the P3-SAM model.
The model is composed of three parts:
1. Sonata: a 3D-CNN model for point cloud feature extraction.
2. SEG1+SEG2: a two-stage multi-head segmentor
3. IoU prediction: an IoU predictor
'''
def build_P3SAM(self): #build p3sam
    ######################## Sonata ########################
    self.sonata = sonata.load("sonata", repo_id="facebook/sonata", download_root='/root/sonata')
    self.mlp = nn.Sequential(
            nn.Linear(1232, 512),
            nn.GELU(),
            nn.Linear(512, 512),
            nn.GELU(),
            nn.Linear(512, 512),
        )
    self.transform = sonata.transform.default()
    ######################## Sonata ########################

    ######################## SEG1 ########################
    self.seg_mlp_1 = nn.Sequential(#seg1
            nn.Linear(512+3+3, 512),
            nn.GELU(),
            nn.Linear(512, 512),
            nn.GELU(),
            nn.Linear(512, 1),
        )
    self.seg_mlp_2 = nn.Sequential( #seg2
            nn.Linear(512+3+3, 512),
            nn.GELU(),
            nn.Linear(512, 512),
            nn.GELU(),
            nn.Linear(512, 1),
        )
    self.seg_mlp_3 = nn.Sequential(
            nn.Linear(512+3+3, 512),
            nn.GELU(),
            nn.Linear(512, 512),
            nn.GELU(),
            nn.Linear(512, 1),
        )
    ######################## SEG1 ########################

    ######################## SEG2 ########################
    self.seg_s2_mlp_g = nn.Sequential( #seg2
            nn.Linear(512+3+3+3, 256),
            nn.GELU(),
            nn.Linear(256, 256),
            nn.GELU(),
            nn.Linear(256, 256),
        )
    self.seg_s2_mlp_1 = nn.Sequential( #seg2
            nn.Linear(512+3+3+3+256, 256),
            nn.GELU(),
            nn.Linear(256, 256),
            nn.GELU(),
            nn.Linear(256, 1),
        )
    self.seg_s2_mlp_2 = nn.Sequential( #seg2
            nn.Linear(512+3+3+3+256, 256),
            nn.GELU(),
            nn.Linear(256, 256),
            nn.GELU(),
            nn.Linear(256, 1),
        )
    self.seg_s2_mlp_3 = nn.Sequential( #seg2
            nn.Linear(512+3+3+3+256, 256),
            nn.GELU(),
            nn.Linear(256, 256),
            nn.GELU(),
            nn.Linear(256, 1),
        )
    ######################## SEG2 ########################

    
    self.iou_mlp = nn.Sequential( #iou predictor
            nn.Linear(512+3+3+3+256, 256),
            nn.GELU(),
            nn.Linear(256, 256),
            nn.GELU(),
            nn.Linear(256, 256),
        )
    self.iou_mlp_out = nn.Sequential( #iou predictor
            nn.Linear(256, 256),
            nn.GELU(),
            nn.Linear(256, 256),
            nn.GELU(),
            nn.Linear(256, 3),
        )
    self.iou_criterion = torch.nn.MSELoss() #iou predictor

'''
Load the P3-SAM model from a checkpoint.
If ckpt_path is not None, load the checkpoint from the given path.
If state_dict is not None, load the state_dict from the given state_dict.
If both ckpt_path and state_dict are None, download the model from huggingface and load the checkpoint.
'''
def load_state_dict(self, 
                    ckpt_path=None, 
                    state_dict=None, 
                    strict=True, 
                    assign=False, 
                    ignore_seg_mlp=False, 
                    ignore_seg_s2_mlp=False, 
                    ignore_iou_mlp=False):   # load checkpoint
    if ckpt_path is not None:
        state_dict = torch.load(ckpt_path, map_location="cpu")["state_dict"]
    elif state_dict is None:
        # download from huggingface
        print(f'trying to download model from huggingface...')
        from huggingface_hub import hf_hub_download
        ckpt_path = hf_hub_download(repo_id="tencent/Hunyuan3D-Part", filename="p3sam.ckpt", local_dir='weights')
        print(f'download model from huggingface to: {ckpt_path}')
        state_dict = torch.load(ckpt_path, map_location="cpu")["state_dict"]

    local_state_dict = self.state_dict()
    seen_keys = {k: False for k in local_state_dict.keys()}
    for k, v in state_dict.items():
        if k.startswith("dit."):
            k = k[4:]
        if k in local_state_dict:
            seen_keys[k] = True
            if local_state_dict[k].shape == v.shape:
                local_state_dict[k].copy_(v)
            else:
                print(f"mismatching shape for key {k}: loaded {local_state_dict[k].shape} but model has {v.shape}")
        else:
            print(f"unexpected key {k} in loaded state dict")
    seg_mlp_flag = False #ignore seg_mlp
    seg_s2_mlp_flag = False #ignore seg_s2_mlp
    iou_mlp_flag = False #ignore iou_mlp
    for k in seen_keys: #check missing keys
        if not seen_keys[k]:
            if ignore_seg_mlp and 'seg_mlp' in k:
                seg_mlp_flag = True
            elif ignore_seg_s2_mlp and'seg_s2_mlp' in k:
                seg_s2_mlp_flag = True
            elif ignore_iou_mlp and 'iou_mlp' in k:
                iou_mlp_flag = True
            else: #missing key
                print(f"missing key {k} in loaded state dict")
    if ignore_seg_mlp and seg_mlp_flag: #ignore seg_mlp
        print("seg_mlp is missing in loaded state dict, ignore seg_mlp in loaded state dict")
    if ignore_seg_s2_mlp and seg_s2_mlp_flag: #ignore seg_s2_mlp
        print("seg_s2_mlp is missing in loaded state dict, ignore seg_s2_mlp in loaded state dict")
    if ignore_iou_mlp and iou_mlp_flag: #ignore iou_mlp
        print("iou_mlp is missing in loaded state dict, ignore iou_mlp in loaded state dict")   
