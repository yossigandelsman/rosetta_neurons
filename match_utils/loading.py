import torch
import pickle
import os


def load_stats(root, device):
    '''Load table and stats.'''
    print("Loading...")
    file_name = os.path.join(root, "table.pkl")
    with open(file_name, 'rb') as f:
        table = pickle.load(f)
        table = table.to(device)#cpu()
    
    with open(os.path.join(root,"discr_stats.pkl"), 'rb') as f:
        discr_stats = pickle.load(f)
        
        for iii, item1 in enumerate(discr_stats):
            for jjj, item2 in enumerate(discr_stats[iii]):
                discr_stats[iii][jjj] = discr_stats[iii][jjj].to(device)
                
        
    with open(os.path.join(root,"gan_stats.pkl"), 'rb') as f:
        gan_stats = pickle.load(f)
        for iii, item1 in enumerate(gan_stats):
            for jjj, item2 in enumerate(gan_stats[iii]):
                gan_stats[iii][jjj] = gan_stats[iii][jjj].to(device)
                
        
        
    print("Done")
    return table, gan_stats, discr_stats
