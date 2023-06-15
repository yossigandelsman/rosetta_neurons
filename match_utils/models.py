import tensorflow
import torch
from pytorch_pretrained_biggan import (BigGAN, one_hot_from_names, truncated_noise_sample,
                                       save_as_images, display_in_terminal)
from torchvision.models import resnet50
from transformers import CLIPProcessor, CLIPModel
import clip
import dnnlib
import styleganxl.legacy
import pickle
from stylegan2.model import Generator
from mae import load_mae
import timm.models.vision_transformer
from typing import Text
import os.path


def load_gan(mode, device = 'cpu', path: Text = '.'):
    if mode == "biggan":
        gan = BigGAN.from_pretrained('biggan-deep-256').to(device)
        gan_layers = []
        for name, layer in gan.named_modules():
            if "conv" in name:
                gan_layers.append(name)
                
    elif mode == "stylegan3-afhq":
        with open('stylegan3-r-afhqv2-512x512.pkl', 'rb') as f:
            gan = pickle.load(f)['G_ema'].to(device) 
            gan_layers = []
            for name, layer in gan.named_modules():
                if "synthesis." in name and "affine" not in name:
                    gan_layers.append(name)
    elif mode == "stylegan3-ffhq":
        with open('stylegan3-r-ffhqu-1024x1024.pkl', 'rb') as f:
            gan = pickle.load(f)['G_ema'].to(device) 
            gan_layers = []
            for name, layer in gan.named_modules():
                if "synthesis." in name and "affine" not in name:
                    gan_layers.append(name)
    elif mode == "projgan":
        with open(os.path.join(path, 'bedroom.pkl'), 'rb') as f:
            gan = pickle.load(f)['G'].to(device)
            gan_layers = []
            for name, layer in gan.named_modules():
                if "synthesis." in name:
                    gan_layers.append(name)   
    elif mode == "styleganxl":
        with dnnlib.util.open_url("https://s3.eu-central-1.amazonaws.com/avg-projects/stylegan_xl/models/imagenet256.pkl") as f:
            gan = styleganxl.legacy.load_network_pkl(f)['G_ema'].to(device)
            gan_layers = []
            for name, layer in gan.named_modules():
                if "synthesis" in name and "affine" not in name:
                    gan_layers.append(name)
    elif mode == "vqgan":
        model = _load_vqgan_model(
            os.path.join(path, "vqgan_imagenet_f16_1024.yaml"),
            os.path.join(path, "vqgan_imagenet_f16_1024.ckpt"),)
    elif mode == "stylegan2-lsun_cat":
        gan = Generator(256, 512, 8, channel_multiplier=2)
        ckpt = torch.load(os.path.join(path, "stylegan2-cat-config-f.pt"), map_location='cpu')
        gan.load_state_dict(ckpt['g_ema'], strict = False)
        gan.to(device)
        gan_layers = []
        for name, layer in gan.named_modules():
            if ("convs." in name and "activate" in name) or ("to_rgbs" in name and "conv" in name and "modulation" not in name):
                gan_layers.append(name) 
    elif mode == "stylegan2-lsun_horse":
        gan = Generator(256, 512, 8, channel_multiplier=2)
        ckpt = torch.load(os.path.join(path, "stylegan2-horse-config-f.pt"), map_location='cpu')
        gan.load_state_dict(ckpt['g_ema'], strict = False)
        gan.to(device)
        gan_layers = []
        for name, layer in gan.named_modules():
            if ("convs." in name and "activate" in name) or ("to_rgbs" in name and "conv" in name and "modulation" not in name):
                gan_layers.append(name) 
    elif mode == "stylegan2-lsun_car":
        gan = Generator(512, 512, 8, channel_multiplier=2)
        ckpt = torch.load(os.path.join(path, "stylegan2-car-config-f.pt"), map_location='cpu')
        gan.load_state_dict(ckpt['g_ema'], strict = False)
        gan.to(device)
        gan_layers = []
        for name, layer in gan.named_modules():
            if ("convs." in name and "activate" in name) or ("to_rgbs" in name and "conv" in name and "modulation" not in name):
                gan_layers.append(name) 
    elif mode == "stylegan2-ffhq":
        gan = Generator(512, 512, 8, channel_multiplier=2)
        ckpt = torch.load(os.path.join(path, "stylegan2-ffhq-config-f.pt"), map_location='cpu')
        gan.load_state_dict(ckpt['g_ema'], strict = False)
        gan.to(device)
        gan_layers = []
        for name, layer in gan.named_modules():
            if ("convs." in name and "activate" in name) or ("to_rgbs" in name and "conv" in name and "modulation" not in name):
                gan_layers.append(name) 
    else: 
        print("Not a valid GAN model")
        
    return gan, gan_layers


def load_discr(mode, device='cpu', path: Text = '.'):
    if mode == 'mae':
        discr = load_mae(os.path.join(path, 'mae_pretrain_vit_base.pth')).to(device)
        #discr_layers = [f"blocks.{i}" for i in range(12)]
        discr_layers = []
        for name, layer in discr.named_modules():
            if  "mlp.act" in name:
                discr_layers.append(name)
                
    elif mode in ['dino_vitb16', 'dino_vitb8']:
        discr = torch.hub.load('facebookresearch/dino:main', mode).to(device)
        for p in discr.parameters(): 
            p.data = p.data.float() 
        
        discr_layers = []
        for name, layer in discr.named_modules():
            if  "mlp.act" in name:
                discr_layers.append(name)
        #discr_layers = [f"blocks.{i}" for i in range(12)]
    elif mode == "dino":
        discr = torch.hub.load('facebookresearch/dino:main', 'dino_resnet50').to(device)
        for p in discr.parameters(): 
            p.data = p.data.float() 
        discr_layers = [ "layer1", "layer2", "layer3", "layer4"]
    elif mode == "clip":
        discr, _ = clip.load("RN50", device=device)
        discr_layers = [ "visual.layer1", "visual.layer2", "visual.layer3", "visual.layer4"]
        for p in discr.parameters(): 
            p.data = p.data.float()
    elif mode == "resnet50":
        discr = resnet50(num_classes=1000, pretrained='imagenet').to(device)
        discr_layers = [ "layer1", "layer2", "layer3", "layer4"]
        for p in discr.parameters(): 
            p.data = p.data.float()
    else:
        print("Not a valid discriminative model")

    return discr, discr_layers
