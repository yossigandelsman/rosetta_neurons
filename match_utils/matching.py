from match_utils import nethook, dataset, stats, helpers

import torch
import torchvision
import numpy as np
import pickle
import os
import tqdm

def normalize(activation, stats_table):
    '''Normalize activations based on statistic from dataset.'''
    eps = 0.00001
    norm_input = (activation- stats_table[0])/(stats_table[1]+eps)
    
    return norm_input


def store_activs(model, layernames):
    '''Store the activations in a list.'''
    activs = []
    for layer in layernames:
        activation = model.retained_layer(layer, clear = True)
        activs.append(activation)
        
    return activs


def dict_layers(activs):
    '''Return dictionary of layer sizes.'''
    all_layers = {}
    for iii, activ in enumerate(activs):
        all_layers[activs[iii]] = activ.shape[1]
    return all_layers


def activ_match_gan(gan, gan_layers, discr,discr_layers, gan_mode, discr_mode,
                    dataset, epochs, batch_size, save_path, device):
    
    '''Main function for matching units between two models. Returns a two dimensional table of pairwise unit scores averaged over the entire dataset.'''
    gan.eval()
    discr.eval()
    
    #### hook layers for GAN
    gan = nethook.InstrumentedModel(gan)
    gan.retain_layers(gan_layers)
    
    #### hook layers for discriminator
    discr = nethook.InstrumentedModel(discr)
    discr.retain_layers(discr_layers)
    
    #get dataset stats
    gan_stats_table, discr_stats_table = stats.get_mean_std(gan, gan_layers, discr, discr_layers, gan_mode, discr_mode, dataset, epochs, batch_size, device)
    helpers.save_array(gan_stats_table, os.path.join(save_path, "gan_stats.pkl"))
    helpers.save_array(discr_stats_table, os.path.join(save_path, "discr_stats.pkl"))
    
    print("Done")
    print("Starting Activation Matching")
    
    

    for iteration in tqdm.trange(0, epochs):
        with torch.no_grad():
            #### dataset
            z = dataset[0][iteration*batch_size: (iteration+1)*batch_size ]
            c = dataset[1][iteration*batch_size: (iteration+1)*batch_size ]
            
            #### Forward through GAN
            if gan_mode == "biggan":
                img = gan(z,c,1)
                img = (img+1)/2
            elif "stylegan3" in gan_mode:
                img = gan(z,c)
                img = (img+1)/2
            elif gan_mode == "projgan":
                img = gan(z,c, truncation_psi = 0.7)
                img = (img+1)/2
            elif gan_mode == "styleganxl":
                img = gan(z,c, truncation_psi = 0.7)
                img = (img+1)/2
            elif "stylegan2" in gan_mode:
                img, _  = gan([z], 0.7, c)
                img = (img+1)/2
                
            del z
                
            
            #### append GAN layer activations for batch
            gan_activs = store_activs(gan, gan_layers)

            #### Prepare images for discriminator
            
            if discr_mode == "clip":
                img = torch.nn.functional.interpolate(img, size = (224,224), mode = "bicubic")
                img = torchvision.transforms.Normalize((0.48145466, 0.4578275, 0.40821073), 
                                                       (0.26862954, 0.26130258, 0.27577711))(img)
                _ = discr.model.encode_image(img)
            else: 
                img = torch.nn.functional.interpolate(img, size = (224,224), mode = "bicubic")
                img = torchvision.transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))(img)
                _ = discr(img)
            del img


            #### append discriminator layer activations for batch
            discr_activs =  store_activs(discr, discr_layers)

            #create dict of layers with number of activations
            all_gan_layers = dict_layers(gan_activs)
            all_discr_layers = dict_layers(discr_activs)
            
            
            if iteration == 0:
                num_gan_activs = sum(all_gan_layers.values())
                num_discr_activs = sum(all_discr_layers.values())
                final_match_table = torch.zeros((num_gan_activs, num_discr_activs)).to(device)


            ##### Matching
            all_match_table = []

            for ii, gan_activ in enumerate(gan_activs):
                match_table = []
                gan_activ = normalize(gan_activ, gan_stats_table[ii])
                gan_activ_shape = gan_activ.shape

                for jj, discr_activ in enumerate(discr_activs):
                    discr_activ_new = normalize(discr_activ, discr_stats_table[jj]) 
                    #### Get maps to same size
                    map_size = max((gan_activ_shape[2], discr_activ.shape[2]))
                    gan_activ_new = torch.nn.Upsample(size=(map_size,map_size), mode='bilinear')(gan_activ)
                    discr_activ_new = torch.nn.Upsample(size=(map_size,map_size), mode='bilinear')(discr_activ_new)            
                    #Pearson Correlation 
                    prod = torch.einsum('aixy,ajxy->ij', gan_activ_new,discr_activ_new)
                    div1 = torch.einsum('aixy->i', gan_activ_new**2)
                    div2 = torch.einsum('ajxy->j', discr_activ_new**2)
                    div = torch.einsum('i,j->ij', div1,div2)
                    scores = prod/torch.sqrt(div)
                    nans = torch.isnan(scores)
                    scores[nans] = 0
                    scores = scores.cpu()
                    
                    match_table.append(scores)
                    del gan_activ_new
                    del discr_activ_new
                    del scores

                all_match_table.append(match_table)
                del match_table


            ##create table
            batch_match_table = create_final_table(all_match_table, all_gan_layers, all_discr_layers, batch_size, device)
            final_match_table += batch_match_table
        
            del all_match_table
            del batch_match_table
            del gan_activs
            del discr_activs
            torch.cuda.empty_cache()
            
    #average and save
    final_match_table /= epochs
    helpers.save_array(final_match_table, os.path.join(save_path, "table.pkl"))
    
    
    
def create_final_table(all_match_table, model1_dict, model2_dict, batch_size, device ):
    num_activs1 = sum(model1_dict.values())
    num_activs2 = sum(model2_dict.values())
    final_match_table = torch.zeros((num_activs1, num_activs2)).to(device)
    
    model1_activ_count = 0 
    for ii in range(len(all_match_table)):
        model2_activ_count = 0
        for jj in range(len(all_match_table[ii])):
            num_model1activs = all_match_table[ii][0].shape[0]
            num_model2activs = all_match_table[0][jj].shape[1]
            final_match_table[model1_activ_count: model1_activ_count+num_model1activs, \
                            model2_activ_count:model2_activ_count+num_model2activs] = all_match_table[ii][jj]
            model2_activ_count += num_model2activs
        model1_activ_count += num_model1activs
    return final_match_table
