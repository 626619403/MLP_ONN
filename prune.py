import torch.nn.utils.prune as prune
from train import *
import wandb
from dataloader import *

prune_method = prune.L1Unstructured 
def prune_model(model,prune_amount):
    parameters_to_prune = []
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            parameters_to_prune.append((module, 'weight'))
    prune.global_unstructured(
        parameters_to_prune,
        pruning_method=prune_method,
        amount=prune_amount
    )
    for module, _ in parameters_to_prune:
        prune.remove(module, 'weight')
    train_student(model,5,trainloader)
    wandb.log({"stu_test_score_undistilled_pruned":test_student(model, testloader)})
    return model