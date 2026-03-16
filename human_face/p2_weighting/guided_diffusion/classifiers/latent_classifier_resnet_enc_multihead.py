"""
Implementation of a time-aware multi-prediction-headed latent classifier that uses 
Resnet-X architecture as feature extractor.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from collections import OrderedDict
import argparse
from typing import Dict, Callable, List, Tuple

GENDER_ID2LABEL = {
    0: "male",
    1: "female"
}

RACE_ID2LABEL = {
    0: "wmelh",
    1: "black",
    2: "asian",
    3: "indian"
}

AGE_GROUP_ID2LABEL = {
    0: '0-2',
    1: '3-6', 
    2: '7-9',
    3: '10-14',
    4: '15-19', 
    5: '20-29',
    6: '30-39',
    7: '40-49',
    8: '50-69',
    9: '70-120'
}

AGE_10_TO_3_MAP = {
    ## {"0-19", "20-49", "50+"}
    0: 0,  # 0-2   -> 0-19
    1: 0,  # 3-6   -> 0-19
    2: 0,  # 7-9   -> 0-19
    3: 0,  # 10-14 -> 0-19
    4: 0,  # 15-19 -> 0-19
    5: 1,  # 20-29 -> 20-49
    6: 1,  # 30-39 -> 20-49
    7: 1,  # 40-49 -> 20-49
    8: 2,  # 50-69 -> 50+
    9: 2   # 70-120   -> 50+
}


def mount_resnet_multi_head_latent_classifier_configs(yaml_path, args=None):
    """Utiliy function to mount attributes from YAML config to args object."""
    
    import yaml
    import argparse
    
    with open(yaml_path, 'r') as f:
        config = yaml.safe_load(f)
    
    if args is None:
        args = argparse.Namespace()
    
    # Dir for checkpoints, logs, samples, etc.
    args.output_dir = str(config.get('output_dir', 'outputs/ldm_train/base'))
    
    # Device
    config['device'] = config.get('device', {})
    args.device               =  str(config['device'].get('device', None))
    args.deterministic        = bool(config['device'].get('deterministic', True))
    
    # Optional W&B
    config['wandb'] = config.get('wandb', {})
    args.use_wandb            = bool(config['wandb'].get('use', False))
    args.wandb_project        =  str(config['wandb'].get('project', 'FFHQ_Aging_ldm'))
    args.wandb_name           =  str(config['wandb'].get('name', 'base'))
    
    # Data params
    config['data'] = config.get('data', {})
    args.images_dir             =   str(config['data'].get('images_dir', 'ffhq_aging/images/images512x512'))
    args.label_json_filepath    =   str(config['data'].get('label_json_filepath', 'ffhq_aging/labels/labels.json'))
    args.train_images           =   str(config['data'].get('train_images', '00000 - 68999'))
    args.test_images            =   str(config['data'].get('test_images', '69000 - 69999'))
    args.age_group_classes      = [str(i) for i in config['data'].get('age_group_classes', ["0-2", "3-6"])]
    args.gender_classes         = [str(i) for i in config['data'].get('gender_classes', ["male", "female"])]
    args.race_classes           = [str(i) for i in config['data'].get('race_classes', [])]
    args.age_group_conf_thres   = float(config['data'].get('age_group_conf_thres', -1))
    args.gender_conf_thres      = float(config['data'].get('gender_conf_thres', -1))
    args.race_conf_thres      = float(config['data'].get('race_conf_thres', -1))
    args.img_height             =   int(config['data'].get('img_height', 512))
    args.img_width              =   int(config['data'].get('img_width', 512))
    args.img_format             =   str(config['data'].get('img_format', 'png'))
    args.reweight_class_loss    =   str(config['data'].get('reweight_class_loss', 'none'))
    args.smoothing_factor       = float(config['data'].get('smoothing_factor', 1.0))
    args.dataloader_num_workers =   int(config['data'].get('dataloader_num_workers', 24))
    train_images = args.train_images.split("-")
    test_images  = args.test_images.split("-")
    args.train_images = [str(i).zfill(5) for i in range(int(train_images[0].strip()), int(train_images[1].strip()) + 1)]
    args.test_images  = [str(i).zfill(5) for i in range(int(test_images[0].strip()), int(test_images[1].strip()) + 1)]
    
    # Latent Classifier params
    config['classifier'] = config.get('classifier', {})
    args.cls_ldm_id              =   str(config['classifier'].get('ldm_id', '"stabilityai/stable-diffusion-2-1-base"'))
    args.cls_in_channels         =   int(config['classifier'].get('in_channels', 4))
    args.cls_backbone            =   str(config['classifier'].get('backbone', 'resnet18'))
    args.cls_pretrained_backbone =  bool(config['classifier'].get('pretrained_backbone', False))
    args.cls_max_timestep        =   int(config['classifier'].get('max_timestep', 999))
    args.cls_timestep_cond       =  bool(config['classifier'].get('timestep_condition', True))
    args.cls_head_configs = {}
    for head_name, specs in config['classifier'].get("head_configs", {}).items():
        num_classes =   int(specs.get("num_classes", 10))
        hidden_dim  =   int(specs.get("hidden_dim", 256))
        dropout_p   = float(specs.get("dropout_p", 0.1))
        criterion_w = float(specs.get("loss_weight", 10.))
        if "race" in head_name: head_name = "race"
        args.cls_head_configs[head_name] = (num_classes, hidden_dim, dropout_p, criterion_w)
    
    # Training hyperparams
    config['training'] = config.get('training', {})
    args.train_seed                    =   int(config['training'].get('seed', 42))
    args.train_batch_size              =   int(config['training'].get('batch_size', 128))
    args.train_epochs                  =   int(config['training'].get('epochs', 100))
    args.train_lr_scheduler_type       =   str(config['training'].get('lr_scheduler', 'constant'))
    args.train_lr                      = float(config['training'].get('lr', 1e-4))
    args.train_lr_rop_patience         =   int(config['training'].get('lr_rop_patience', 1))
    args.train_lr_rop_factor           = float(config['training'].get('lr_rop_factor', 0.75))
    args.train_lr_rop_min              = float(config['training'].get('lr_rop_min', 1e-6))
    args.train_lr_cosw_warmup_steps    =   int(config['training'].get('lr_cosw_warmup_steps', 2500))
    args.train_lr_cosw_max_train_steps =   int(config['training'].get('lr_cosw_max_train_steps', 51200))
    args.train_lr_cosw_cosine_cycles   = float(config['training'].get('lr_cosw_cosine_cycles', 0.5))
    args.train_lr_cosw_last_epoch      =   int(config['training'].get('lr_cosw_last_epoch', -1))
    args.train_lr_conw_warmup_steps    =   int(config['training'].get('lr_conw_warmup_steps', 2500))
    args.train_sample_test_count       =   int(config['training'].get('sample_test_count', 10))
    args.train_debug_mode              =  bool(config['training'].get('debug_mode', False))
    args.train_debug_break_num_steps   =   int(config['training'].get('debug_break_num_steps', 10))
    
    # Normalization
    config['value_range'] = config.get('value_range', {})
    args.data_min             = float(config['value_range'].get('data_min', 0.0))
    args.data_max             = float(config['value_range'].get('data_max', 1.0))
    args.model_min            = float(config['value_range'].get('model_min', -1.0))
    args.model_max            = float(config['value_range'].get('model_max',  1.0))
    args.ldm_min              = float(config['value_range'].get('ldm_min', -1.0))
    args.ldm_max              = float(config['value_range'].get('ldm_max',  1.0))
    
    return args



def build_time_aware_multi_pred_head_classifier(args, device):
    """Function to build a custom time-aware multi-prediction-headed classifier, based on Resnet."""
    return TimeAwareDynamicHeadResnetFeatureClassifier(
        in_channels         = args.cls_in_channels,
        head_configs        = args.cls_head_configs,
        backbone            = args.cls_backbone,
        pretrained_backbone = args.cls_pretrained_backbone,
        max_timestep        = args.cls_max_timestep,
        timestep_cond       = args.cls_timestep_cond,
        device              = device,
        args                = args
    ).to(device)


def get_pcd_classifier(config_pth, weights_pth, device):
    img_classifier_args = mount_resnet_multi_head_latent_classifier_configs(config_pth, argparse.Namespace())
    img_classifier = build_time_aware_multi_pred_head_classifier(img_classifier_args, device)
    img_classifier.load_state_dict(torch.load(weights_pth, map_location=device))
    img_classifier.eval()

    return img_classifier


def map_tensor_range(
    tensor: torch.Tensor,
    in_range: Tuple[float, float],
    out_range: Tuple[float, float]
) -> torch.Tensor:
    """
    Linearly map a tensor from in_range [a, b] to out_range [c, d].
    """
    a, b = in_range
    c, d = out_range
    a, b, c, d = float(a), float(b), float(c), float(d)
    
    ret = ((tensor.float() - a) / (b - a)) * (d - c) + c
    return ret.to(tensor)

def resize_imgs(x, m, mode='bicubic'):
    """
    x:  [B, 3, H, W]
    m:  target size (int)
    mode: 'bicubic' (good general choice), 'area' (best for big downscales)
    """
    if x.shape[-1] == m:
        return x
    # antialias only matters when downscaling; safe to always set True on >= PyTorch 2.0
    return F.interpolate(
        x, size=(m, m), mode=mode,
        align_corners=False if mode in ('bilinear', 'bicubic', 'trilinear') else None,
        antialias=True
    )


def infer_pcd_classifier(img_classifier, batch_latents, heads=[], timesteps=None):
    # already mapped to [0., 1.] during diffusion
    batch_latents = map_tensor_range(
        tensor    = batch_latents,
        in_range  = (0., 1.),
        out_range = (img_classifier.model_min, img_classifier.model_max),
    )
    if (batch_latents.shape[2] == img_classifier.img_height) and \
        (batch_latents.shape[3] == img_classifier.img_width):
        batch_resized = batch_latents # no need to resize
    else:
        batch_resized = resize_imgs(
            x = batch_latents,  
            m = img_classifier.img_height,
            mode = 'bicubic'
        )
    
    ## The classifier itself handles time conditioning internally
    ## Also handles whether it's image or latent based on in_channels
    logits_dict = img_classifier(batch_resized, timesteps=timesteps)
    if len(heads) == 0:
        ret = logits_dict
    else:
        ret = {h: logits_dict[h] for h in heads}

    return ret 

def infer_pcd_latent_classifier(latent_classifier, batch_latents, heads=[], timesteps=None):
    batch_latents = map_tensor_range(
        tensor    = batch_latents,
        in_range  = (-1, 1),          # diffusion latents are in [-1, 1]
        out_range = (latent_classifier.model_min, latent_classifier.model_max),
    )
    ## heads not specified → return all heads 
    logits_dict = latent_classifier(batch_latents, timesteps=timesteps)
    if len(heads) == 0:
        ret = logits_dict
    else:
        ret = {h: logits_dict[h] for h in heads}

    return ret 


class TimeAwareDynamicHeadResnetFeatureClassifier(nn.Module):
    def __init__(
        self,
        in_channels: int,
        head_configs: dict,
        backbone: str = "resnet18",
        pretrained_backbone: bool = False,
        max_timestep: int = 999,      # for normalizing t
        device: torch.device = None,
        timestep_cond: bool = True,
        args = None
    ):
        """
        Args:
            in_channels: number of channels in your latent input (e.g. 4)
            head_configs: dict of
                head_name → (num_classes:int, hidden_dim:int, dropout_prob:float)
              e.g. {
                "age": (10, 256, 0.5),
                "gender": (2, 128, 0.3),
                # you can add more...
              }
            backbone: name of a torchvision ResNet, e.g. "resnet18", "resnet34", ...
            pretrained_backbone: whether to load imagenet-pretrained weights
        """
        super().__init__()
        
        for k, v in vars(args).items():
            setattr(self, k, v)
        
        # 1) Build backbone
        assert hasattr(models, backbone), f"No torchvision model {backbone}"
        self.backbone = getattr(models, backbone)(pretrained=pretrained_backbone)
        # only replace if time conditioning is taken as an additional channel
        if timestep_cond:
            old_conv = self.backbone.conv1
            self.backbone.conv1 = nn.Conv2d(
                in_channels=in_channels + 1,  # +1 for time dimension
                out_channels=old_conv.out_channels,
                kernel_size=old_conv.kernel_size,
                stride=old_conv.stride,
                padding=old_conv.padding,
                bias=False
            )
        else:
            old_conv = self.backbone.conv1
            self.backbone.conv1 = nn.Conv2d(
                in_channels=in_channels,  # modify input channels
                out_channels=old_conv.out_channels,
                kernel_size=old_conv.kernel_size,
                stride=old_conv.stride,
                padding=old_conv.padding,
                bias=False
            )
            print("[CLS] >> Time not taken as input!")
        
        num_feat = self.backbone.fc.in_features
        self.backbone.fc = nn.Identity()
        
        # 2) Dynamic heads
        self.heads = nn.ModuleDict()
        self.loss_weights = {}
        for name, (num_classes, hidden_dim, dropout_p, criterion_w) in head_configs.items():
            self.heads[name] = nn.Sequential(OrderedDict([
                ("lin1", nn.Linear(num_feat, hidden_dim)),
                ("act",  nn.ReLU(inplace=True)),
                ("drop", nn.Dropout(dropout_p)),
                ("lin2", nn.Linear(hidden_dim, num_classes)),
            ]))
            self.loss_weights[name] = criterion_w
        
        self.max_timestep = max_timestep
        self.timestep_cond = timestep_cond
        self.device = device
        self.to(device)
    
    def forward(self, latents: torch.Tensor, timesteps: torch.Tensor = None, return_penulminate: bool = False):
        """
        latents: [B, 4, 64, 64]
        timesteps: [B] or [B,]
        returns: {head_name: logits [B, num_classes], ...}
        """
        B, C, H, W = latents.shape
        # normalize t to [0,1]
        if self.timestep_cond:
            t = timesteps.float().view(B, 1, 1, 1) / float(self.max_timestep)
            t_map = t.expand(-1, 1, H, W)               # [B,1,H,W]
            x = torch.cat([latents, t_map], dim=1)      # now [B,5,H,W]
        else:
            x = latents  # [B,3,H,W]
        
        feats = self.backbone(x)                    # [B, num_feat]
        if return_penulminate:
            return feats
        return {name: head(feats) for name, head in self.heads.items()}
