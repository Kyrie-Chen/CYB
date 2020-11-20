import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import torchvision
import torchvision.transforms as transforms


## 1 加载并归一化 CIFAR10 中的数据
def loadDataset():
    '''
        torchvision数据集的输出是在[0, 1]范围内的PILImage图片。
        此处使用归一化的方法将其转化为Tensor，数据范围为[-1, 1]
    '''
    # 格式转换形式
    trans = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    # 加载训练集数据
    trainset = torchvision.datasets.CIFIAR10(root='/data', train=True, download=True, transform=trans)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=4, shuffle=True, num_workers=2)

    # 加载测试集数据
    testset = torchvision.datasets.CIFIAR10(root='/data', train=False, download=True, transform=trans)
    testloader = torch.utils.data.DataLoader(trainset, batch_size=4, shuffle=False, num_workers=2)

    classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
    return trainloader, testloader


## 2 定义一个卷积神经网络
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        # 卷积层
        self.conv1 = nn.Conv2d(1, 6, 5)  # 1 input image channel, 6 output channels, 5x5 square convolution
        self.conv2 = nn.Conv2d(6, 16, 5)
        # 池化层
        self.pool = nn.MaxPool2d(2, 2)  # Max pooling over a (2, 2) window
        # 全连接层
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    # 前向传播
    def forward(self, x):
        # Conv2d -> ReLU -> MaxPool
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))

        # 展开为向量
        x = x.view(-1, 16 * 5 * 5)

        # 全连接层
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


## 3 模型训练
def modelTrain(model, trainloader, testloader, criterion, optimizer, num_epochs=25):
    ## 模型训练
    for epoch in range(num_epochs):
        running_loss = 0.0

        # 批量梯度下降
        for i, data in enumerate(trainloader, 0):
            # 加载数据集
            X, y = data
            X, y = X.to(device), y.to(device)  # GPU训练

            optimizer.zero_grad = 0

            # forward + backward + optimize（传入变量）
            y_pred = model(X)
            loss = citerion(y_pred, y)
            loss.backward()
            optimizer.step()

            # 每2000次打印损失函数
            running_loss += loss.item()
            if i % 2000 == 1999:
                print('[%d, %5d] loss: %3f' % (epoch + 1, i + 1, running_loss / 2000))
                running_loss = 0.0

    return model


if __main__:
    # GPU训练
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    # 加载数据集（迭代器）
    trainloader, testloader = loadDataset()

    # 定义模型
    model = Net()
    #     model = nn.DataParallel(model)    # 多GPU训练
    model.to(device)  # GPU训练，或者直接 model.cude()

    # 定义损失函数和优化器
    citerion = nn.CrossEntropyLoss()  # 分类交叉熵函数Cross-Entropy作为损失函数
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentun=0.9)

    ## 模型训练
    model = modelTrain(model, trainloader, testloader, criterion, optimizer)

    #     # 直接训练（不通过函数）
    #     for epoch in range(2):
    #         running_loss = 0.0

    #         # 批量梯度下降
    #         for i, data in enumerate(trainloader, 0):
    #             # 加载数据集
    #             X, y = data
    #             X, y = X.to(device), y.to(device)    # GPU训练

    #             optimizer.zero_grad = 0

    #             # forward + backward + optimize（传入变量）
    #             y_pred = model(X)
    #             loss = citerion(y_pred, y)
    #             loss.backward()
    #             optimizer.step()

    #             # 每2000次打印损失函数
    #             running_loss += loss.item()
    #             if i % 2000 == 1999:
    #                 print('[%d, %5d] loss: %3f' % (epoch + 1, i + 1, running_loss / 2000))
    #                 running_loss = 0.0

    # 预测
    total = 0
    correct = 0
    with torch.no_grad():
        for data in testloader:
            X_test, y_test = data
            X_test, y_test = X_test.to(device), y_test.to(device)  # GPU训练

            y_pred = model(X_test)
            _, predicted = torch.max(y_pred.data, 1)  # 预测类型
            total += y.size()  # 样本总数
            correct += (predicted == y_test).sum().item()  # 预测正确的数量
    print('Accuracy of the network on the 10000 test images: %d %%' % (100 * correct / total))



