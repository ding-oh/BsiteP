import torch
from collections import OrderedDict
from SEResnet import SEResNet

def load_model(model_path):
    model = SEResNet().cuda()
    state_dict = torch.load(model_path, map_location=torch.device('cuda'), weights_only=True)
    new_dict = OrderedDict((key[7:], value) for key, value in state_dict.items())
    model.load_state_dict(new_dict)
    model.eval().to('cuda')
    return model
