import torch


def compute_model_l2norm(model):
    total_norm=0.
    params=model.state_dict()
    for key,var in params.items():
        value=torch.tensor(var.data.clone().detach(),dtype = torch.double)
        total_norm+=value.norm(2)**2
    total_norm=total_norm**0.5 
    return  