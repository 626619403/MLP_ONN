import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms.functional as TF
import time
import random
from args import *
from resnet import ResNet18
from dataloader import *
from train import *
from MLP import *
import prune
from distill import train_knowledge_distillation
import wandb

#wandb.init(project="MLP",config=args)
wandb.init(project="MLP",config=args,mode="disabled")
image_size=args.image_size

num_epochs = args.train_epoch
prune_amount=args.prune_amount
num_i = image_size * image_size  
num_h =args.hidden_layer_size
num_o = 10  
batch_size = 64
parameter_num=image_size*image_size*args.hidden_layer_size
for i in range(1,args.layer_num):
    parameter_num+=num_h*num_h
parameter_num+=num_h*num_o

teacher_model = ResNet18()

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
teacher_model.to(device)

train_teacher(teacher_model,num_epochs,trainloader) 
test_teacher(teacher_model,testloader)
student_model = MLP(num_i, num_h, num_o).to(device)
student_model.qconfig = get_default_qat_qconfig_per_tensor()
model_qat = prepare_qat(student_model)
criterion = nn.CrossEntropyLoss()

train_student(model_qat,num_epochs,trainloader)
wandb.log({"stu_test_score_undistilled_unpruned":test_student(model_qat, testloader)})

model_qat=prune.prune_model(model_qat,prune_amount)

model_qat.train()
train_knowledge_distillation(teacher=teacher_model, student=model_qat, train_loader=trainloader, epochs=args.distill_epoch, learning_rate=0.005, T=2, soft_target_loss_weight=0.25, ce_loss_weight=0.75, device=device)

parameter_num=parameter_num*(1-prune_amount)
model_qat.eval()

wandb.log({"stu_test_score_distilled":test_student(model_qat, testloader)})
wandb.log({"paremeter_num":parameter_num})

model_quantized = convert(model_qat, inplace=False)
torch.save(model_quantized.state_dict(),'./student_model.pth')