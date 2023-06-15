import numpy as np
import torch
import matplotlib.pyplot as plt

def plot_stats(gan_stats, discr_stats, table):
    '''Plot means and standard deviations for GAN and discriminative statistics.'''
    gan_means = []
    gan_stds = []
    for iii, layer in enumerate(gan_stats):
        gan_means.append(gan_stats[iii][0].flatten().unsqueeze(0))
        gan_stds.append(gan_stats[iii][1].flatten().unsqueeze(0))

    gan_means = torch.cat(gan_means,1)
    gan_stds = torch.cat(gan_stds,1)
    
    
    
    
    
    discr_means = []
    discr_stds = []
    for iii, layer in enumerate(discr_stats):
        discr_means.append(discr_stats[iii][0].flatten().unsqueeze(0))
        discr_stds.append(discr_stats[iii][1].flatten().unsqueeze(0))

    discr_means = torch.cat(discr_means,1)
    discr_stds = torch.cat(discr_stds,1)
    
    
    
    
    ### scores
    
    table_flattened = table.flatten()
    scores, flat_indices = torch.sort(table_flattened, descending = True)
    flat_indices_matches = flat_indices.cpu()
    indices_matches = np.unravel_index(flat_indices_matches, (table.shape[0], table.shape[1]))
    
 
    fig, ax = plt.subplots(figsize =(3, 3))
    ax.hist(discr_means.cpu(), bins = [-20, -10, -5, -1, 0, 1, 5, 10, 20])
    plt.title("Discriminator Means")
    plt.show()
    
    
    fig, ax = plt.subplots(figsize =(3, 3))
    ax.hist(discr_stds.cpu(), bins = [0, 0.1, 0.2, 0.3, 0.4, 0.5,1, 5])
    plt.title("Discriminator STDs")
    plt.show()
    
    
    fig, ax = plt.subplots(figsize =(3, 3))
    ax.hist(gan_means.cpu(), bins = [-20, -10, -5, -1, 0, 1, 5, 10, 20])
    plt.title("GAN Means")
    plt.show()
    
    
    fig, ax = plt.subplots(figsize =(3, 3))
    ax.hist(gan_stds.cpu(), bins = [-20, -10, -5, -1, 0, 1, 5, 10, 20])
    plt.title("GAN STDs")
    plt.show()
    
    
    
    
    gan_score_idxs,_ = torch.max(table, dim = 1)
    fig, ax = plt.subplots(figsize =(10,6))
    ax.plot(gan_score_idxs.cpu())
    plt.title("Scores vs. GAN Depth")
    plt.show()
    
    
    
    discr_score_idxs,_ = torch.max(table, dim = 0)
    fig, ax = plt.subplots(figsize =(10,6))
    ax.plot(discr_score_idxs.cpu())
    plt.title("Scores vs. Discriminator Depth")
    plt.show()
