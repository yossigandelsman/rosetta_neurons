import torch
import torchvision
from scipy.stats import truncnorm




def truncate_noise(size, truncation):
    '''
    Function for creating truncated noise vectors: Given the dimensions (n_samples, z_dim)
    and truncation value, creates a tensor of that shape filled with random
    numbers from the truncated normal distribution.
    Parameters:
        n_samples: the number of samples to generate, a scalar
        z_dim: the dimension of the noise vector, a scalar
        truncation: the truncation value, a non-negative scalar
    '''
    
    truncated_noise = truncnorm.rvs(-1*truncation, truncation, size=size)
    
    return torch.Tensor(truncated_noise)



def create_dataset(gan, ganmode, batch_size, epochs, classidx, device):
    '''Create dataset for GAN.'''
    
    if ganmode == "biggan":
        z_dataset = truncate_noise((batch_size*epochs,128), 1).to(device)
        c_dataset = torch.zeros((batch_size*epochs,1000)).to(device)
        c_dataset[:, classidx] = 1
    elif "stylegan3" in ganmode:
        z_dataset = torch.randn([batch_size*epochs, 512]).to(device) 
        c_dataset = []
        for iii in range(batch_size*epochs):
            c_dataset.append(None)
    elif ganmode == "projgan":
        z_dataset = truncate_noise((batch_size*epochs,256), 1).to(device)
        c_dataset = []
        for iii in range(batch_size*epochs):
            c_dataset.append(None)
    elif ganmode == "styleganxl":
        z_dataset = truncate_noise((batch_size*epochs,64), 1).to(device)
        c_dataset = torch.zeros((batch_size*epochs,1000)).to(device)
        c_dataset[:, classidx] = 1
    elif "stylegan2" in ganmode:
        z_dataset = truncate_noise((batch_size*epochs,512), 1).to(device)
        with torch.no_grad():
            mean_latent = gan.mean_latent(4096)
        c_dataset = []
        for iii in range(batch_size*epochs):
            c_dataset.append(mean_latent)
            
    return z_dataset, c_dataset
