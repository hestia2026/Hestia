import torch
from transformers import set_seed

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    set_seed(seed)