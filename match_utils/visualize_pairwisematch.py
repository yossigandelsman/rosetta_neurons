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

from PIL import Image

from match_utils import matching, stats, proggan, nethook, dataset, loading, plotting, layers




def viz_matches(table, gan, discr, dataset, ganlayers, discrlayers, ganstats, discrstats, gan_mode, discr_mode, global_matches, global_scores):
    '''Visualize matches between a GAN and one discriminative model.'''
    
    
    gan.eval()
    discr.eval()
    
    z = dataset[0]
    c = dataset[1]
    
    
    #### hook layers for GAN
    gan = nethook.InstrumentedModel(gan)
    gan.retain_layers(ganlayers)
    
    #### hook layers for discriminator
    discr = nethook.InstrumentedModel(discr)
    discr.retain_layers(discrlayers)
    
    
    with torch.no_grad():
        
        for rank, unit in enumerate(global_matches):
            match1 = table[unit]
            scores, flat_indices = torch.sort(match1, descending = True)
            ganidx = layers.find_act(unit, ganlayers)
            
            
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
            gan_act = matching.normalize(gan_act, ganstats[ganidx[0]])

            ### through discriminator
            
            discridx = layers.find_act(flat_indices[0], discrlayers )
            
            
            if discr_mode == "clip":
                img = torch.nn.functional.interpolate(img, size = (224,224), mode = "bicubic")
                img = torchvision.transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))(img)
                _ = discr.model.encode_image(img)
            else:
                img = torch.nn.functional.interpolate(img, size = (224,224), mode = "bicubic")
                img = torchvision.transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))(img)
                _ = discr(img)
            
            
            
            discr_act = discr.retained_layer(list(discrlayers)[discridx[0]], clear = True)
            discr_act = matching.normalize(discr_act, discrstats[discridx[0]])
             
                
            ##### resize
            map_size = map_size = max((gan_act.shape[2], discr_act.shape[2]))
            gan_act = torch.nn.Upsample(size=(map_size,map_size), mode='bilinear')(gan_act)
            discr_act = torch.nn.Upsample(size=(map_size,map_size), mode='bilinear')(discr_act)        
            scores = torch.mul(gan_act[:, ganidx[1], :, :], discr_act[:, discridx[1], :, :])
            scores = scores.view(scores.shape[0], -1)
            scores = torch.mean(scores, dim=-1, dtype = torch.float32)
            scores_sorted, indices = torch.sort(scores, descending = True)


            for ex in range(1):
                discr_act_viz1 = discr_act[indices[ex], discridx[1]]
                discr_act_viz = discr_act_viz1.unsqueeze(0).unsqueeze(0)
                discr_act_viz = torch.nn.Upsample(size=(img.shape[2], img.shape[3]), mode='bilinear')(discr_act_viz).cpu()
                discr_act_viz = (discr_act_viz-torch.min(discr_act_viz))/(torch.max(discr_act_viz)-torch.min(discr_act_viz))
                discr_act_viz = img_as_ubyte(discr_act_viz)
                discr_act_viz = cv2.applyColorMap(discr_act_viz[0][0], cv2.COLORMAP_JET)

                gan_act_viz1 = gan_act[indices[ex], ganidx[1]]
                gan_act_viz = gan_act_viz1.unsqueeze(0).unsqueeze(0)
                gan_act_viz = torch.nn.Upsample(size=(img.shape[2], img.shape[3]), mode='bilinear')(gan_act_viz).cpu()
                gan_act_viz = (gan_act_viz-torch.min(gan_act_viz))/(torch.max(gan_act_viz)-torch.min(gan_act_viz))
                gan_act_viz = img_as_ubyte(gan_act_viz)
                gan_act_viz = cv2.applyColorMap(gan_act_viz[0][0], cv2.COLORMAP_JET)


                img_viz = torch.permute(img[indices[ex]].cpu(), (1,2,0))
                img_viz = ((img_viz-torch.min(img_viz))/(torch.max(img_viz)-torch.min(img_viz))).numpy()


                ###Plot
                fig=plt.figure(figsize=(13, 5))
                plt.axis("off")
                plt.title("Best Buddy Match Rank #" + str(rank+1)+" , Ex. "+str(ex+1)+" Global Score: "+str(round(global_scores[rank], 3)), y=0.85)

                logger = logging.getLogger()
                old_level = logger.level
                logger.setLevel(100)

                alpha = 0.003

                minifig= fig.add_subplot(1, 5, 1)
                minifig.axis('off')
                minifig.title.set_text("Original Image")
                        
                plt.imshow(img_viz)


                minifig2 = fig.add_subplot(1, 5, 2)
                minifig2.axis('off')
                minifig2.title.set_text("GAN Layer "+str(ganidx[0])+", Unit "+str(ganidx[1]))
                plt.imshow(alpha*gan_act_viz+img_viz)

                minifig3 = fig.add_subplot(1, 5, 3)
                minifig3.axis('off')
                minifig3.title.set_text("Discr. Layer "+str(discridx[0])+", Unit "+str(discridx[1].item()))
                plt.imshow(alpha*discr_act_viz+img_viz)



                minifig4 = fig.add_subplot(1, 5, 4)
                minifig4.axis('off')
                minifig4.title.set_text("GAN Map")
                
                ganmap = gan_act[indices[ex], ganidx[1]].cpu().unsqueeze(0).unsqueeze(0)
                ganmap = torch.nn.Upsample(size=(256, 256), mode='nearest')(ganmap).cpu()
                plt.imshow(gan_act[indices[ex], ganidx[1]].cpu())


                minifig5 = fig.add_subplot(1, 5, 5)
                minifig5.axis('off')
                minifig5.title.set_text("Discr. Map")
                
                plt.imshow(discr_act[indices[ex], discridx[1]].cpu())








                logger.setLevel(old_level)

            del img
            del discr_act_viz
            del discr_act_viz1
            del gan_act_viz
            del gan_act_viz1
            torch.cuda.empty_cache()
