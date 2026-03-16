import torch
import torchvision
from torchvision import transforms
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple


RACE_7_ID2LABEL = {
    0: 'white',
    1: 'black',
    2: 'latino_hispanic',
    3: 'east_asian',
    4: 'southeast_asian',
    5: 'indian',
    6: 'middle_eastern'
}

RACE_4_ID2LABEL = {
    0: 'wmelh',
    1: 'black',
    2: 'asian',
    3: 'indian'
}

RACE_7_TO_4_MAP = {
    0: 0,  # white -> WMELH
    1: 1,  # black -> black
    2: 0,  # latino_hispanic -> WMELH
    3: 2,  # east_asian -> asian
    4: 2,  # southeast_asian -> asian
    5: 3,  # indian -> indian
    6: 0   # middle_eastern -> WMELH
}

GENDER_ID2LABEL = {
    0: 'male',
    1: 'female'
}

AGE_GROUP_ID2LABEL = {
    0: '0-2',
    1: '3-9',
    2: '10-19',
    3: '20-29',
    4: '30-39',
    5: '40-49',
    6: '50-59',
    7: '60-69',
    8: '70+'
}

AGE8_TO_3_MAP = {
    ## {"0-19", "20-49", "50+"}
    0: 0,  # 0-2 -> 0-19
    1: 0,  # 3-9 -> 0-19
    2: 0,  # 10-19 -> 0-19
    3: 1,  # 20-29 -> 20-49
    4: 1,  # 30-39 -> 20-49
    5: 1,  # 40-49 -> 20-49
    6: 2,  # 50-59 -> 50+
    7: 2,  # 60-69 -> 50+
    8: 2   # 70+   -> 50+ 
}


_DATA_MEAN = [0.485, 0.456, 0.406]
_DATA_STD = [0.229, 0.224, 0.225]
DATA_MEAN_TENSOR = torch.tensor(_DATA_MEAN).view(1, 3, 1, 1)
DATA_STD_TENSOR = torch.tensor(_DATA_STD).view(1, 3, 1, 1)

def id_label_mapping_inverse(mapping: dict) -> dict:
    return {v: k for k, v in mapping.items()}


def get_fairface_classifier_7(weights_path: str, device) -> nn.Module:
    model_fair_7 = torchvision.models.resnet34(pretrained=True)
    model_fair_7.fc = nn.Linear(model_fair_7.fc.in_features, 18)
    model_fair_7.load_state_dict(torch.load(weights_path, weights_only=True))
    model_fair_7 = model_fair_7.to(device)
    model_fair_7.eval()
    return model_fair_7

def get_fairface_classifier_4(weights_path: str, device) -> nn.Module:
    model_fair_4 = torchvision.models.resnet34(pretrained=True)
    model_fair_4.fc = nn.Linear(model_fair_4.fc.in_features, 18)
    model_fair_4.load_state_dict(torch.load(weights_path, weights_only=True))
    model_fair_4 = model_fair_4.to(device)
    model_fair_4.eval()
    return model_fair_4



def preprocess_resnet34_bchw_01(x: torch.Tensor, size=(224, 224), antialias: bool = True) -> torch.Tensor:
    """
    Applies differentiable resizing and ImageNet normalization to input tensor.

    Args:
        x (torch.Tensor): Input float tensor in [0, 1] with shape [B, 3, H, W] (BCHW).
        size (tuple, optional): Target spatial size (height, width). Defaults to (224, 224).
        antialias (bool, optional): Whether to apply antialiasing during resizing. Defaults to True.

    Returns:
        torch.Tensor: Normalized float tensor with shape [B, 3, 224, 224].
    """
    if x.ndim != 4 or x.shape[1] != 3:
        raise ValueError(f"Expected x with shape [B,3,H,W], got {tuple(x.shape)}")

    ## Operate in float32 for better precision
    x_dtype = x.dtype
    # if not torch.is_floating_point(x):
    #     x = x.float()
    x = x.to(dtype=torch.float32)

    # Differentiable resize
    x = F.interpolate(x, size=size, mode="bilinear", align_corners=False, antialias=antialias)

    # Normalize (make mean/std live on same device/dtype)
    mean = DATA_MEAN_TENSOR.to(x)
    std = DATA_STD_TENSOR.to(x)
    x = (x - mean) / std

    return x.to(dtype=x_dtype)


def get_fairface_transforms(differentiable: bool = True) -> transforms.Compose:
    if differentiable:
        return preprocess_resnet34_bchw_01
    else:
        trans = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=_DATA_MEAN, std=_DATA_STD)
        ])
    
    return trans

def infer_fairface(model: nn.Module, images: torch.Tensor, 
                    num_race_cls:int=4,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    assert num_race_cls in [4, 7], "Only support 4 or 7 race classes"
    outputs = model(images)
    assert outputs.shape[-1] == 18

    dim_race = num_race_cls
    race_outputs = outputs[..., :dim_race]              # 4 or 7 races 
    gender_outputs = outputs[..., 7:9]  # 2 genders
    age_outputs = outputs[..., 9:18]  # 9 age groups

    return {
        "race": race_outputs, 
        "gender": gender_outputs, 
        "age_group": age_outputs
    }

