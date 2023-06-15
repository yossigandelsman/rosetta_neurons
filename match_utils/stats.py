from match_utils import nethook, dataset
import torch
import torchvision
import tqdm


def get_mean_std(gan, gan_layers, discr, discr_layers, gan_mode, discr_mode, dataset, epochs, batch_size, device):
    '''Get activation statistics over dataset for GAN and discriminative model.'''
    
    print("Collecting Dataset Statistics")
    gan_stats_list = []
    discr_stats_list = []
    with torch.no_grad():
        for iteration in tqdm.trange(0, epochs):
            z = dataset[0][iteration*batch_size: (iteration+1)*batch_size ]
            c = dataset[1][iteration*batch_size: (iteration+1)*batch_size ]
            
            if gan_mode == "biggan":
                img = gan(z,c,1)
                img = (img+1)/2
            elif gan_mode == "stylegan3-afhq":
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

            #### append GAN layer activations for batch
            gan_activs = {}
            for layer in gan_layers:
                gan_activs[layer] = []    
                gan_activation = gan.retained_layer(layer, clear = True)
                gan_activs[layer].append(gan_activation)
            #### Prepare images for discriminator
            
            if discr_mode == "clip":
                img = torch.nn.functional.interpolate(img, size = (224,224), mode = "bicubic")
                img = torchvision.transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))(img)
                _ = discr.model.encode_image(img)
            else:
                img = torch.nn.functional.interpolate(img, size = (224,224), mode = "bicubic")
                img = torchvision.transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))(img)
                _ = discr(img)

            del img
            
            discr_activs = {}
            for layer in discr_layers:
                discr_activs[layer] = []
                discr_activation = discr.retained_layer(layer, clear = True)
                discr_activs[layer].append(discr_activation)
            batch_gan_stats_list = []
            for layer in gan_layers:
                gan_activs[layer] = torch.cat(gan_activs[layer], 0) #images x channels x m x m
                gan_activs[layer] = torch.permute(gan_activs[layer], (1,0,2,3)).contiguous() #channels x images x m x m
                gan_activs[layer] = gan_activs[layer].view(gan_activs[layer].shape[0], -1) 
                batch_gan_stats_list.append([torch.mean(gan_activs[layer],dim=-1, dtype=torch.float64).unsqueeze(0).unsqueeze(2).unsqueeze(3).to(device),\
                                      torch.std(gan_activs[layer], dim=-1).unsqueeze(0).unsqueeze(2).unsqueeze(3).to(device)])
            del gan_activs
            gan_stats_list.append(batch_gan_stats_list)

            batch_discr_stats_list = []
            discr_stats_list.append(batch_discr_stats_list)
            for layer in discr_layers:
                discr_activs[layer] = torch.cat(discr_activs[layer], 0)
                discr_activs[layer] = torch.permute(discr_activs[layer], (1,0,2,3)).contiguous()
                discr_activs[layer] = discr_activs[layer].view(discr_activs[layer].shape[0], -1)
                batch_discr_stats_list.append([torch.mean(discr_activs[layer], dim=-1, dtype=torch.float64).unsqueeze(0).unsqueeze(2).unsqueeze(3).to(device),\
                                      torch.std(discr_activs[layer], dim=-1).unsqueeze(0).unsqueeze(2).unsqueeze(3).to(device)])

            del discr_activs
            torch.cuda.empty_cache()

    
    ####################### After iterating
        print("Finished Iterating for Stats")
        final_discr_stats_list = []

        for iii in range(len(batch_discr_stats_list)):
            means = torch.zeros_like(batch_discr_stats_list[iii][0])
            stds = torch.zeros_like(batch_discr_stats_list[iii][1])
            for jjj in range(epochs):
                means+=discr_stats_list[jjj][iii][0]
                stds+=discr_stats_list[jjj][iii][1]**2

            final_discr_stats_list.append([means/epochs, torch.sqrt(stds/epochs)])



        final_gan_stats_list = []

        for iii in range(len(batch_gan_stats_list)):
            means = torch.zeros_like(batch_gan_stats_list[iii][0])
            stds = torch.zeros_like(batch_gan_stats_list[iii][1])
            for jjj in range(epochs):
                means+=gan_stats_list[jjj][iii][0]
                stds+=gan_stats_list[jjj][iii][1]**2

            final_gan_stats_list.append([means/epochs, torch.sqrt(stds/epochs)])




    return final_gan_stats_list, final_discr_stats_list
