from args import *
from torch.ao.quantization import QuantStub, DeQuantStub, prepare_qat, prepare,convert,QConfig,get_default_qat_qconfig,default_observer
from torch.ao.quantization.observer import MinMaxObserver
import torch

class MLP(torch.nn.Module):

    def __init__(self, num_i, num_h, num_o):
        super(MLP, self).__init__()
        self.quant = QuantStub()
        self.linear1 = torch.nn.Linear(num_i, num_h if args.layer_num>1 else num_o)
        self.relu = torch.nn.ReLU()
        self.linear2 = torch.nn.Linear(num_h, num_h if args.layer_num>2 else num_o) 
        self.relu2 = torch.nn.ReLU()
        self.linear3 = torch.nn.Linear(num_h, num_h if args.layer_num>3 else num_o)  
        self.relu3 = torch.nn.ReLU()
        self.linear4 = torch.nn.Linear(num_h, num_o)
        self.dequant = DeQuantStub()
        
    
    def forward(self, x):
        x = self.quant(x)
        x = self.linear1(x)
        if args.layer_num>1:
            x = self.relu(x)
            x = self.linear2(x) 
        if args.layer_num>2:    
            x = self.relu2(x)
            x = self.linear3(x) 
        if args.layer_num>3:
            x = self.relu(x)
            x = self.linear4(x)
        x = self.dequant(x)
        return x
    
def get_default_qat_qconfig_per_tensor(backend='fbgemm'):
    activation = default_observer.with_args(reduce_range=False,quant_min=-16,quant_max=15)
    weight = default_observer.with_args(dtype=torch.qint8, reduce_range=False,quant_min=-16,quant_max=15)
    return QConfig(activation=activation, weight=weight)