import torch
import numpy as np
import torchvision.transforms as transforms
import torchvision
import matplotlib.pyplot as plt
from PIL import Image
import pickle
import math
import cv2
from skimage import img_as_ubyte
import logging
import warnings
warnings.filterwarnings("ignore")

from match_utils import matching, stats, proggan, nethook, dataset, loading, plotting, layers




def viz_matches(gan, discr, dataset, ganlayers, discrlayers, ganstats, discrstats, gan_mode, discr_mode, global_matches):
    
    '''Visualize Rosetta Neurons across all 6 models.'''
    
    #one example image
    num_ex = 1
    
    
    gan.eval()
    
    for i,_ in enumerate(discr):
        discr[i].eval()
    
    z = dataset[0]
    c = dataset[1]
    
    
    #### hook layers for GAN
    gan = nethook.InstrumentedModel(gan)
    gan.retain_layers(ganlayers)
    
    #### hook layers for discriminator
    for i, _ in enumerate(discr):
        model = nethook.InstrumentedModel(discr[i])
        model.retain_layers(discrlayers[i])
        discr[i] = model
    
    
    with torch.no_grad():
        
        for rank,_ in enumerate(global_matches):
            viz_list = []
            info_dict = {}
            for iii in range(len(discr)):
                ganidx = layers.find_act(list(global_matches.keys())[rank], ganlayers)
                
                info_dict["ganidx"] = ganidx

                if gan_mode == "biggan":
                    img = gan(z,c,1)
                    img = (img+1)/2
                elif gan_mode == "stylegan3":
                    img = gan(z,c)
                    img = (img+1)/2
                elif gan_mode == "projgan":
                    img = gan(z,c, truncation_psi = 0.7)
                    img = (img+1)/2
                elif gan_mode == "styleganxl":
                    img = gan(z,c, truncation_psi = 0.5)
                    img = (img+1)/2
                elif "stylegan2" in gan_mode:
                    img, _  = gan([z], 0.7, c)
                    img = (img+1)/2

                gan_act = gan.retained_layer(list(ganlayers)[ganidx[0]], clear = True)
                gan_act = matching.normalize(gan_act, ganstats[iii][ganidx[0]])

                ### through discriminator
            
                
                discridx = layers.find_act(global_matches[list(global_matches.keys())[rank]][iii], discrlayers[iii] )
                info_dict["discridx"+str(iii)] = discridx

                if discr_mode[iii] == "clip":
                    img = torch.nn.functional.interpolate(img, size = (224,224), mode = "bicubic")
                    img = torchvision.transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))(img)
                    _ = discr[iii].model.encode_image(img)
                else:
                    img = torch.nn.functional.interpolate(img, size = (224,224), mode = "bicubic")
                    img = torchvision.transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))(img)
                    _ = discr[iii](img)



                discr_act = discr[iii].retained_layer(list(discrlayers[iii])[discridx[0]], clear = True)
                discr_act = matching.normalize(discr_act, discrstats[iii][discridx[0]])


                ##### resize
                map_size = map_size = max((gan_act.shape[2], discr_act.shape[2]))
                gan_act = torch.nn.Upsample(size=(map_size,map_size), mode='bilinear')(gan_act)
                discr_act = torch.nn.Upsample(size=(map_size,map_size), mode='bilinear')(discr_act)        
                scores = torch.mul(gan_act[:, ganidx[1], :, :], discr_act[:, discridx[1], :, :])
                scores = scores.view(scores.shape[0], -1)
                scores = torch.mean(scores, dim=-1, dtype = torch.float32)
                scores_sorted, indices = torch.sort(scores, descending = True)

                ex_list = []
                for ex in range(num_ex):
                    discr_act_viz1 = discr_act[indices[ex], discridx[1]]
                    discr_act_viz = discr_act_viz1.unsqueeze(0).unsqueeze(0)
                    discr_act_viz = torch.nn.Upsample(size=(img.shape[2], img.shape[3]), mode='nearest')(discr_act_viz).cpu()
                    
                    gan_act_viz1 = gan_act[indices[ex], ganidx[1]]
                    gan_act_viz = gan_act_viz1.unsqueeze(0).unsqueeze(0)
                    gan_act_viz = torch.nn.Upsample(size=(img.shape[2], img.shape[3]), mode='nearest')(gan_act_viz).cpu()
                    

                    img_viz = torch.permute(img[indices[ex]].cpu(), (1,2,0))
                    img_viz = ((img_viz-torch.min(img_viz))/(torch.max(img_viz)-torch.min(img_viz))).numpy()
                    
                    ### list of visualizations for this example and discr
                    ex_list.append([img_viz, gan_act_viz, discr_act_viz, gan_act[indices[ex], ganidx[1]].cpu(),discr_act[indices[ex], discridx[1]].cpu()  ])
                
                viz_list.append(ex_list)

            for jjjj in range(len(ex_list)):
                fig=plt.figure(figsize=(15, 5))
                plt.axis("off")
                plt.title("Universal Match #" + str(rank+1)+" , Ex. "+str(jjjj+1), y=0.85)

                logger = logging.getLogger()
                old_level = logger.level
                logger.setLevel(100)
                alpha = 0.003
                num_plots = 2+len(discr)
                
                for kkkk in range(num_plots):
                    minifig= fig.add_subplot(1,num_plots, kkkk+1)
                    minifig.axis('off')
                    if kkkk == 0:
                        ax = plt.gca()       
                        ax.set_title('Original Image',size=10) 
                        plt.imshow(viz_list[kkkk][jjjj][0])
                        
                    elif (kkkk==1):
                        ax = plt.gca()       
                        ax.set_title("GAN Layer "+str(info_dict["ganidx"][0])+", Unit "+str(info_dict["ganidx"][1]),size=8) 
                        plt.imshow(viz_list[kkkk][jjjj][3])
                    else:
                        discrunit = info_dict["discridx"+str(kkkk-2)]
                        ax = plt.gca()
                        ax.set_title("Discr. Layer "+str(discrunit[0])+", Unit "+str(discrunit[1]), fontsize = 8)
                        plt.imshow(viz_list[kkkk-2][jjjj][2][0,0])
                    
                    logger.setLevel(old_level)

           
            
