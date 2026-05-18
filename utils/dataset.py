import os
import numpy as np
import torch
import torch.nn as nn
import timm
from tqdm import tqdm
from torchvision import models, transforms
from torchvision.models import ResNet50_Weights
from PIL import Image
from models.glnet import G_LiteNet

def load_backbone(name, device, custom_pth_path=None):
    if name == "resnet":
        print("Loading ResNet50 with IMAGENET1K_V2 weights...")
        net = models.resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
        net = nn.Sequential(*list(net.children())[:-1])
        net = net.to(device).eval()
        tfm = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        dim = 2048

    elif name == "custom_resnet":
        print(f"Loading G_LiteNet from {custom_pth_path}...")
        net = G_LiteNet(num_classes=1000)
        net.load_state_dict(torch.load(custom_pth_path, map_location=device, weights_only=True))
        # net.load_state_dict(torch.load(custom_pth_path, map_location=device))
        net.fc = nn.Identity()
        net    = net.to(device).eval()
        tfm    = transforms.Compose([transforms.Resize(256),
                                     transforms.CenterCrop(224),
                                     transforms.ToTensor(), 
                                     transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
        dim    = 2048

    elif name == "convnext":
        print("Loading timm model: convnextv2_huge...")
        net = timm.create_model("convnextv2_huge", pretrained=True, num_classes=0, in_chans=3)
        net = net.to(device).eval()
        data_config = timm.data.resolve_model_data_config(net)
        tfm = timm.data.create_transform(**data_config, is_training=False)
        dim = net.num_features

    elif name == "swin":
        print("Loading timm model: swinv2_large...")
        net = timm.create_model("swinv2_large_window12to16_192to256", pretrained=True, num_classes=0, in_chans=3)
        net = net.to(device).eval()
        data_config = timm.data.resolve_model_data_config(net)
        tfm = timm.data.create_transform(**data_config, is_training=False)
        dim = net.num_features if hasattr(net, 'num_features') else 1536
    else:
        raise ValueError(f"Unknown backbone {name}")
    
    return net, tfm, dim

def extract_feat(path, model, tfm, device):
    img = tfm(Image.open(path).convert("RGB")).unsqueeze(0).to(device)
    with torch.no_grad():
        out = model(img)
    return out.view(-1).cpu().numpy()

def build_dataset(df, backbone, root_glauc, root_normal, device, custom_pth_path=None):
    model, tfm, dim     = load_backbone(backbone, device, custom_pth_path)
    feats, labels, idxs = [], [], []
    print(f"\nProcessing {len(df)} images through {backbone}...")
    
    # Wrap df.iterrows() with tqdm for a visual progress bar
    for rid, row in tqdm(df.iterrows(), total=len(df), desc=f"Extracting {backbone}"):
        lbl  = int(row["Label"])
        path = os.path.join(root_glauc if lbl == 1 else root_normal, row["Original Image Name"])
        
        if not os.path.exists(path):
            continue
            
        feats.append(extract_feat(path, model, tfm, device))
        labels.append(lbl)
        idxs.append(rid)
    
    if len(feats) == 0:
        raise ValueError(f"CRITICAL ERROR: No features extracted for {backbone}.")

    # print(f"  -> Extracted {len(feats)} features for {backbone}.")
    print(f"  -> Extracted {dim}-dimensional features for {len(feats)} images using {backbone}.")
    return np.vstack(feats), np.array(labels), np.array(idxs)