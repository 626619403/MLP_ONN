import torch
import torch.nn as nn
import torch.optim as optim
import wandb
criterion = nn.CrossEntropyLoss()

def train_teacher(model,num_epochs,trainloader):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
    for epoch in range(num_epochs):
        running_loss = 0.0
        for i, (inputs, labels) in enumerate(trainloader, 0):
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs.to(device))
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            
def train_student(model,epochs,trainloader):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    cost = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters())
    for epoch in range(epochs):
        sum_loss = 0
        train_correct = 0
        for data in trainloader:
            inputs, labels = data
            inputs = torch.flatten(inputs, start_dim=1).to(device)
            outputs = model(inputs).to(device)
            optimizer.zero_grad()
            labels=labels.to(device)
            loss = cost(outputs, labels )
            loss.backward()
            optimizer.step()
            _, id = torch.max(outputs.data, 1)
            sum_loss += loss.data
            train_correct += torch.sum(id == labels.data)
        wandb.log({"stu_loss":sum_loss})
        wandb.log({"stu_acc":train_correct/60000})
        if epoch > 5:
            model.apply(torch.ao.quantization.disable_observer)
        if epoch > 3:
            model.apply(torch.nn.intrinsic.qat.freeze_bn_stats)
    model.eval()

def test_teacher(model,testloader):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    correct = 0
    total = 0
    with torch.no_grad():
        for i, (inputs, labels) in enumerate(testloader, 0):
            inputs, labels = inputs.to(device), labels.to(device)   
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    wandb.log({"teacher_testscore":100 * correct / total})
    print("teacher_testscore:",100 * correct / total)
    
def test_student(model,testloader):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    correct = 0
    total = 0
    with torch.no_grad():
        for i, (inputs, labels) in enumerate(testloader, 0):
            inputs, labels = inputs.to(device), labels.to(device)
            inputs = torch.flatten(inputs, start_dim=1).to(device) 
            outputs = model(inputs.to(device))
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    print("student_testscore:",100 * correct / total)
    return 100 * correct / total