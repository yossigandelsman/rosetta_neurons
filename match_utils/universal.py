from transformers import CLIPProcessor, CLIPModel
import torch
import torch.hub
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
from functools import reduce
from match_utils import matching, stats, proggan, nethook, dataset, loading, plotting, layers, models

def get_universal_activations(resnet_path, mae_path, dino_path, dino_vitb16_path, clip_path,  device, n = 5):
    ''' Obtain the Rosetta Neurons of the set of five discriminative models (ResNet50, MAE, DINO ResNet50, DINO ViTb16, CLIP ResNet50) 
    and a GAN using the n nearest neighbors. Returns dictionary of "best buddies" for the GAN and the five other models along with 
    activation statistics.'''
    
    table1, gan1_stats, discr1_stats = loading.load_stats(resnet_path, device)
    table2, gan2_stats, discr2_stats = loading.load_stats(mae_path, device)
    table3, gan3_stats, discr3_stats = loading.load_stats(dino_path, device)
    table4, gan4_stats, discr4_stats = loading.load_stats(dino_vitb16_path, device)
    table5, gan5_stats, discr5_stats = loading.load_stats(clip_path, device)
   
    #1
    _,gan_matches1 = torch.topk(table1,k=1,dim=1)
    _,discr_matches1 = torch.topk(table1,k=n, dim=0)

    perfect_matches1 = {}
    perfect_match_scores1= []
    discr_perfect_matches1 = []

    num_kmatches = 0 
    for i in range(table1.shape[0]):
        gan_match = gan_matches1[i].item()
        discr_matches = discr_matches1[:, gan_match]
    
        for unit in discr_matches:
            if unit == i:
                num_kmatches += 1
                perfect_matches1[i] = gan_match
                discr_perfect_matches1.append(gan_match)
                perfect_match_scores1.append(table1[i, gan_match])
                break
                
                
                
    #2
    _,gan_matches2 = torch.topk(table2,k=1,dim=1)
    _,discr_matches2 = torch.topk(table2,k=n, dim=0)

    perfect_matches2 = {}
    perfect_match_scores2= []
    discr_perfect_matches2 = []

    num_kmatches = 0 
    for i in range(table2.shape[0]):
        gan_match = gan_matches2[i].item()
        
        discr_matches = discr_matches2[:, gan_match]
    
        for unit in discr_matches:
            if unit == i:
                num_kmatches += 1
                perfect_matches2[i] = gan_match
                discr_perfect_matches2.append(gan_match)
                perfect_match_scores2.append(table2[i, gan_match])
                break
    
    #3
    _,gan_matches3 = torch.topk(table3,k=1,dim=1)
    _,discr_matches3 = torch.topk(table3,k=n, dim=0)

    perfect_matches3 = {}
    perfect_match_scores3= []
    discr_perfect_matches3 = []

    num_kmatches = 0 
    for i in range(table3.shape[0]):
        gan_match = gan_matches3[i].item()
        discr_matches = discr_matches3[:, gan_match]
    
        for unit in discr_matches:
            if unit == i:
                num_kmatches += 1
                perfect_matches3[i] = gan_match
                discr_perfect_matches3.append(gan_match)
                perfect_match_scores3.append(table3[i, gan_match])
                break
    

    _,gan_matches4 = torch.topk(table4,k=1,dim=1)
    _,discr_matches4 = torch.topk(table4,k=n, dim=0)

    perfect_matches4 = {}
    perfect_match_scores4= []
    discr_perfect_matches4 = []

    num_kmatches = 0 
    for i in range(table4.shape[0]):
        gan_match = gan_matches4[i].item()
        discr_matches = discr_matches4[:, gan_match]
    
        for unit in discr_matches:
            if unit == i:
                num_kmatches += 1
                perfect_matches4[i] = gan_match
                discr_perfect_matches4.append(gan_match)
                perfect_match_scores4.append(table4[i, gan_match])
                break
    
    _,gan_matches5 = torch.topk(table5,k=1,dim=1)
    _,discr_matches5 = torch.topk(table5,k=n, dim=0)

    perfect_matches5 = {}
    perfect_match_scores5= []
    discr_perfect_matches5 = []

    num_kmatches = 0 
    for i in range(table5.shape[0]):
        gan_match = gan_matches5[i].item()
        discr_matches = discr_matches5[:, gan_match]
    
        for unit in discr_matches:
            if unit == i:
                num_kmatches += 1
                perfect_matches5[i] = gan_match
                discr_perfect_matches5.append(gan_match)
                perfect_match_scores5.append(table5[i, gan_match])
                break
    

    
    
    all_gan_matches = [list(perfect_matches1.keys()),
                   list(perfect_matches2.keys()), list(perfect_matches3.keys()), 
                   list(perfect_matches4.keys()), list(perfect_matches5.keys())]
    all_discr_matches = [perfect_matches1,
                   perfect_matches2, perfect_matches3, 
                   perfect_matches4, perfect_matches5]
    num_universal_matches = 0

    gan_universal_matches = list(reduce(lambda i, j: i & j, (set(x) for x in all_gan_matches)))

    #if you want to filter to higher level units
    #gan_universal_matches = list(filter(lambda x: (x < 3000), gan_universal_matches)) 



    universal_matches = {}
    for gan_unit in gan_universal_matches:
        discr_matches = []
        for i in range(len(all_discr_matches)):
            discr_matches.append(all_discr_matches[i][gan_unit])
    
        universal_matches[gan_unit] = discr_matches
        
        
    return universal_matches, [(gan1_stats, discr1_stats),(gan2_stats, discr2_stats), (gan3_stats, discr3_stats), (gan4_stats, discr4_stats), (gan5_stats, discr5_stats) ]
    
    
    
