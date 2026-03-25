"""Script version of backward-gradient-5bit-noise-simulation.ipynb.

The execution flow is kept aligned with the original notebook:
1. Build MNIST loaders and visualize ground-truth t-SNE.
2. Train/evaluate the teacher model.
3. Train/evaluate the quantization-aware MLP student.
4. Run knowledge distillation and visualize student predictions.
5. Replace Linear layers with noisy 5-bit backward-path QLinear layers.
6. Retrain, evaluate, quantize, and save the final student model.
"""

import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from sklearn.manifold import TSNE
from torch.ao.quantization import (
    DeQuantStub,
    QConfig,
    QuantStub,
    convert,
    default_observer,
    prepare_qat,
)
from quantized_pytorch_master.models.modules.quantize import QLinear, set_gradient_noise_std


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
image_size = 14
INPUT_NOISE_STD = 0.1
GRADIENT_NOISE_STD = 0.1


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=3,
            stride=stride,
            padding=1,
            bias=False,
        )
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(
            out_channels,
            out_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False,
        )
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != self.expansion * out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(
                    in_channels,
                    self.expansion * out_channels,
                    kernel_size=1,
                    stride=stride,
                    bias=False,
                ),
                nn.BatchNorm2d(self.expansion * out_channels),
            )

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        out += self.shortcut(residual)
        out = self.relu(out)

        return out


class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10):
        super().__init__()
        self.in_channels = 64

        self.conv1 = nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)

    def _make_layer(self, block, out_channels, num_blocks, stride):
        layers = [block(self.in_channels, out_channels, stride)]
        self.in_channels = out_channels * block.expansion
        for _ in range(1, num_blocks):
            layers.append(block(self.in_channels, out_channels))
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.maxpool(out)

        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)

        out = self.avgpool(out)
        out = torch.flatten(out, 1)
        out = self.fc(out)
        return out


def ResNet18(num_classes=10):
    return ResNet(BasicBlock, [2, 2, 2, 2], num_classes)


class AddGaussianNoise:
    def __init__(self, std=0.0):
        self.std = std

    def __call__(self, tensor):
        if self.std <= 0:
            return tensor
        noise = torch.randn_like(tensor) * self.std
        return torch.clamp(tensor + noise, 0.0, 1.0)


class StudentMLP(nn.Module):
    def __init__(self, num_i, num_h, num_o, linear_cls=nn.Linear, linear_kwargs=None):
        super().__init__()
        linear_kwargs = linear_kwargs or {}
        self.quant = QuantStub()
        self.linear1 = linear_cls(num_i, num_h, **linear_kwargs)
        self.relu = nn.ReLU()
        self.linear2 = linear_cls(num_h, num_h, **linear_kwargs)
        self.relu2 = nn.ReLU()
        self.linear3 = linear_cls(num_h, num_h, **linear_kwargs)
        self.relu3 = nn.ReLU()
        self.linear4 = linear_cls(num_h, num_o, **linear_kwargs)
        self.dequant = DeQuantStub()

    def forward(self, x):
        x = self.quant(x)
        x = self.linear1(x)
        x = self.relu(x)
        x = self.linear2(x)
        x = self.relu2(x)
        x = self.linear3(x)
        x = self.relu(x)
        x = self.linear4(x)
        x = self.dequant(x)
        return x


def get_default_qat_qconfig_per_tensor():
    activation = default_observer.with_args(reduce_range=False, quant_min=-16, quant_max=15)
    weight = default_observer.with_args(
        dtype=torch.qint8,
        reduce_range=False,
        quant_min=-16,
        quant_max=15,
    )
    return QConfig(activation=activation, weight=weight)


def build_dataloaders():
    trainset = torchvision.datasets.MNIST(
        root="..//data//",
        transform=torchvision.transforms.Compose(
            [
                torchvision.transforms.Resize((image_size, image_size)),
                torchvision.transforms.ToTensor(),
                AddGaussianNoise(std=INPUT_NOISE_STD),
            ]
        ),
        train=True,
        download=True,
    )
    testset = torchvision.datasets.MNIST(
        root="..//data//",
        transform=torchvision.transforms.Compose(
            [
                torchvision.transforms.Resize((image_size, image_size)),
                torchvision.transforms.ToTensor(),
            ]
        ),
        train=False,
        download=True,
    )
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True)
    testloader = torch.utils.data.DataLoader(testset, batch_size=64, shuffle=False)
    return trainloader, testloader


def collect_flattened_data(trainloader, testloader):
    train_data = []
    train_labels = []
    for images, labels in trainloader:
        train_data.append(images.view(images.size(0), -1).numpy())
        train_labels.append(labels.numpy())

    test_data = []
    test_labels = []
    for images, labels in testloader:
        test_data.append(images.view(images.size(0), -1).numpy())
        test_labels.append(labels.numpy())

    return (
        np.concatenate(train_data, axis=0),
        np.concatenate(train_labels, axis=0),
        np.concatenate(test_data, axis=0),
        np.concatenate(test_labels, axis=0),
    )


def plot_tsne(points, labels, title):
    plt.figure(figsize=(10, 10))
    for i in range(10):
        plt.scatter(points[labels == i, 0], points[labels == i, 1], label=str(i))
    plt.legend()
    plt.title(title)
    plt.show()


def train_teacher(model, num_epochs, trainloader, criterion):
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
    model.to(device)
    for epoch in range(num_epochs):
        running_loss = 0.0
        for i, (inputs, labels) in enumerate(trainloader, 0):
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            if i % 600 == 599:
                print("[%d, %5d] loss: %.3f" % (epoch + 1, i + 1, running_loss / 600))
                running_loss = 0.0
    print("Training finished.")


def train_student(model, epochs, trainloader):
    cost = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters())
    model.to(device)
    for epoch in range(epochs):
        sum_loss = 0
        train_correct = 0
        for inputs, labels in trainloader:
            inputs = torch.flatten(inputs, start_dim=1).to(device)
            inputs = (inputs > 0.25).float()
            outputs = model(inputs).to(device)
            optimizer.zero_grad()
            labels = labels.to(device)
            loss = cost(outputs, labels)
            loss.backward()
            optimizer.step()
            _, predicted = torch.max(outputs.data, 1)
            sum_loss += loss.data
            train_correct += torch.sum(predicted == labels.data)
        if epoch > 5:
            model.apply(torch.ao.quantization.disable_observer)
        if epoch > 3:
            model.apply(torch.nn.intrinsic.qat.freeze_bn_stats)
        print(
            "[%d/%d] loss:%.3f, correct:%.3f%%"
            % (
                epoch + 1,
                epochs,
                sum_loss / len(trainloader),
                100 * train_correct / len(trainloader) / 64,
            )
        )
    model.eval()


def test_teacher(model, testloader):
    correct = 0
    total = 0
    model.to(device)
    model.eval()
    with torch.no_grad():
        for inputs, labels in testloader:
            inputs, labels = inputs.to(device), labels.to(device)
            inputs = (inputs > 0.25).float()
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    print("Accuracy on the test set: %.2f %%" % (100 * correct / total))


def test_student(model, testloader):
    correct = 0
    total = 0
    model.to(device)
    model.eval()
    with torch.no_grad():
        for inputs, labels in testloader:
            inputs, labels = inputs.to(device), labels.to(device)
            inputs = torch.flatten(inputs, start_dim=1).to(device)
            inputs = (inputs > 0.25).float()
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    accuracy = 100 * correct / total
    print("Accuracy on the test set: %.2f %%" % accuracy)
    return accuracy


def train_knowledge_distillation(
    teacher,
    student,
    train_loader,
    epochs,
    learning_rate,
    T,
    soft_target_loss_weight,
    ce_loss_weight,
):
    ce_loss = nn.CrossEntropyLoss()
    optimizer = optim.Adam(student.parameters(), lr=learning_rate)

    teacher.eval()
    student.train()
    teacher.to(device)
    student.to(device)

    for epoch in range(epochs):
        running_loss = 0.0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()

            with torch.no_grad():
                teacher_logits = teacher(inputs)

            student_logits = student(torch.flatten(inputs, start_dim=1).to(device))

            soft_targets = nn.functional.softmax(teacher_logits / T, dim=-1)
            soft_prob = nn.functional.log_softmax(student_logits / T, dim=-1)
            soft_targets_loss = -torch.sum(soft_targets * soft_prob) / soft_prob.size()[0] * (T**2)
            label_loss = ce_loss(student_logits, labels)
            loss = soft_target_loss_weight * soft_targets_loss + ce_loss_weight * label_loss

            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        print(f"Epoch {epoch + 1}/{epochs}, Loss: {running_loss / len(train_loader)}")


def build_qat_student(num_i, num_h, num_o):
    student_model = StudentMLP(num_i, num_h, num_o).to(device)
    student_model.qconfig = get_default_qat_qconfig_per_tensor()
    return prepare_qat(student_model)


def build_noisy_backward_student(num_i, num_h, num_o, num_bits=5):
    set_gradient_noise_std(GRADIENT_NOISE_STD)
    linear_kwargs = {
        "num_bits": num_bits,
        "num_bits_weight": num_bits,
        "num_bits_grad": num_bits,
    }
    student_model = StudentMLP(
        num_i,
        num_h,
        num_o,
        linear_cls=QLinear,
        linear_kwargs=linear_kwargs,
    ).to(device)
    student_model.qconfig = get_default_qat_qconfig_per_tensor()
    return prepare_qat(student_model)


def visualize_student_predictions(model_qat, testloader, test_data):
    model_qat.eval()
    y_pred = []
    tsne = TSNE(n_components=2, random_state=42)
    test_data_tsne = tsne.fit_transform(test_data)
    for images, _labels in testloader:
        images = torch.flatten(images, start_dim=1).to(device)
        outputs = model_qat(images)
        _, predicted = torch.max(outputs.data, 1)
        y_pred.extend(predicted.to("cpu").numpy().tolist())
    y_pred = np.array(y_pred)
    plot_tsne(test_data_tsne, y_pred, "t-SNE Visualization of MNIST Classification(our model)")


def main():
    trainloader, testloader = build_dataloaders()
    teacher_model = ResNet18()
    criterion = nn.CrossEntropyLoss()

    _train_data, _train_labels, test_data, test_labels = collect_flattened_data(trainloader, testloader)
    test_data_tsne = TSNE(n_components=2, random_state=42).fit_transform(test_data)
    plot_tsne(test_data_tsne, test_labels, "t-SNE Visualization of MNIST Classification(ground truth)")

    train_teacher(teacher_model, 10, trainloader, criterion)
    test_teacher(teacher_model, testloader)

    num_i = image_size * image_size
    num_h = 14
    num_o = 10

    model_qat = build_qat_student(num_i, num_h, num_o)
    train_student(model_qat, 20, trainloader)
    test_accuracy_light_ce_and_kd = test_student(model_qat, testloader)
    print(test_accuracy_light_ce_and_kd)
    _model_quantized = convert(model_qat.eval(), inplace=False)

    model_qat.train()
    train_knowledge_distillation(
        teacher=teacher_model,
        student=model_qat,
        train_loader=trainloader,
        epochs=10,
        learning_rate=0.005,
        T=2,
        soft_target_loss_weight=0.25,
        ce_loss_weight=0.75,
    )
    visualize_student_predictions(model_qat, testloader, test_data)
    test_accuracy_light_ce_and_kd = test_student(model_qat, testloader)
    print(test_accuracy_light_ce_and_kd)

    model_qat = build_noisy_backward_student(num_i, num_h, num_o, num_bits=5)
    train_student(model_qat, 10, trainloader)
    test_student(model_qat, testloader)

    quantized_model = convert(model_qat, inplace=False)
    torch.save(quantized_model.state_dict(), "./student_model.pth")


if __name__ == "__main__":
    main()
