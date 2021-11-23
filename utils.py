import torch

def save_param(model, pth_path):
    '''
    save the parameters of the model
    
    Args:
        model: the model to which the params belong
        pth_path: the path where .pth file is saved
    '''
    torch.save(model.state_dict(), pth_path)
    
def load_param(model, pth_path):
    '''
    load the parameters of the model
    
    Args:
        model: the model where the params go into
        pth_path: the path where .pth (to be loaded) is saved
    '''
    model.load_state_dict(torch.load(pth_path))