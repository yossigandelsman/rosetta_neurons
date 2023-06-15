from transformers import CLIPProcessor, CLIPModel
import torch
import torchvision
from torchvision.models import resnet50
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
import clip
from PIL import Image
import requests
import torch.hub
import time
import pickle
import math

from match_utils import matching, stats, proggan, nethook, dataset, loading, plotting

def get_layers(gan, gan_layers, discr,discr_layers, gan_mode, discr_mode, device):
    '''Get a dictionary of the layer dimensions for the GAN and discriminative model.'''
    
    gan.eval()
    discr.eval()
    
    #### hook layers for GAN
    gan = nethook.InstrumentedModel(gan)
    gan.retain_layers(gan_layers)
    
    
    #### hook layers for discriminator
    discr = nethook.InstrumentedModel(discr)
    discr.retain_layers(discr_layers)
    
           
    #### Forward through GAN
    with torch.no_grad():
        if gan_mode == "biggan":
            z = torch.randn((1,128)).to(device)
            c = torch.zeros((1,1000)).to(device)
            img = gan(z,c,1)
            del z
        elif "stylegan3" in gan_mode:
            z = torch.randn((1,512)).to(device)
            c = [None]
            img = gan(z,c)
            del z
        elif gan_mode == "projgan":
            z = torch.randn((1,256)).to(device)
            c = [None]
            img = gan(z,c)
            del z
        elif gan_mode == "styleganxl":
            z = torch.randn((1,64)).to(device)
            c = torch.zeros((1,1000)).to(device)
            img = gan(z,c)
            del z
        elif "stylegan2" in gan_mode:
            z = torch.randn((1,512)).to(device)
            img, _  = gan([z])
            img = (img+1)/2
            del z
        
        
    #### append GAN layer activations for batch
    gan_activs = []
    for layer in gan_layers:
        gan_activation = gan.retained_layer(layer, clear = True).detach()
        gan_activs.append(gan_activation)
         
            
    #### Prepare images for discr (normalization doesn't matter since we are just getting activation shapes)
    img = torch.nn.functional.interpolate(img, size = (224,224), mode = "bicubic")
    
    #### Forward through discriminator
    with torch.no_grad():
    
        if discr_mode == "clip":
            _ = discr.model.encode_image(img)
        else:
            _ = discr(img)
    del img
    
    
    #### append discr layer activations for batch
    discr_activs = []
    for layer in discr_layers:
        discr_activation = discr.retained_layer(layer, clear = True).detach()
        discr_activs.append(discr_activation)
        

    #create dict of layers
    all_gan_layers = {}
    for iii, gan_activ in enumerate(gan_activs):
        all_gan_layers[gan_layers[iii]] = gan_activ.shape[1]
        
    all_discr_layers = {}
    for jjj, discr_activ in enumerate(discr_activs):
        all_discr_layers[discr_layers[jjj]] = discr_activ.shape[1]
            
    return all_gan_layers, all_discr_layers


def find_act(act_num, net_dict):
    '''Turn raw unit number into (layer, unit) tuple).'''
    
    layers_list = list(net_dict)
    
    layer = 0
    counter =0
    
    while act_num >= counter:
        layer +=1
        counter += net_dict[layers_list[layer-1]]
        
        
    act = act_num-counter+net_dict[layers_list[layer-1]]
    
    del layers_list
    torch.cuda.empty_cache()
    return (layer-1), act
        
