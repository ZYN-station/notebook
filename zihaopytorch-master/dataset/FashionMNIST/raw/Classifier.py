import torch  # 导入pytorch
import torch.nn.functional as F
from torch import nn, optim  # 导入神经网络与优化器对应的类

class Classifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(784, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 64)
        self.fc4 = nn.Linear(64, 10)

        # 构造Dropout方法，在每次训练过程中都随机“掐死”百分之二十的神经元，防止过拟合。
        self.dropout = nn.Dropout(p=0.2)

    def forward(self, x):
        # 确保输入的tensor是展开的单列数据，把每张图片的通道、长度、宽度三个维度都压缩为一列
        x = x.view(x.shape[0], -1)

        # 在训练过程中对隐含层神经元的正向推断使用Dropout方法
        x = self.dropout(F.relu(self.fc1(x)))
        x = self.dropout(F.relu(self.fc2(x)))
        x = self.dropout(F.relu(self.fc3(x)))

        # 在输出单元不需要使用Dropout方法
        x = F.log_softmax(self.fc4(x), dim=1)

        return x


# %%
# 对上面定义的Classifier类进行实例化

# class Classifier(nn.Module):
#     def __init__(self):
#         # nn.Module子类的函数必须在构造函数中执行父类的构造函数
#         super(Classifier, self).__init__()
#         # 卷积层conv1
#         self.conv1 = nn.Sequential(
#             nn.Conv2d(1, 16, kernel_size=5, padding=2),
#             nn.BatchNorm2d(16),
#             nn.ReLU())
#         # 池化层pool1
#         self.pool1 = nn.MaxPool2d(2)
#         # 卷积层conv2
#         self.conv2 = nn.Sequential(
#             nn.Conv2d(16, 32, kernel_size=3),
#             nn.BatchNorm2d(32),
#             nn.ReLU())
#         # 卷积层conv3
#         self.conv3 = nn.Sequential(
#             nn.Conv2d(32, 64, kernel_size=3),Classifier.py
#             nn.BatchNorm2d(64),
#             nn.ReLU())
#         # 池化层pool2
#         self.pool2 = nn.MaxPool2d(2)
#         # 全连接层fc(输出层)
#         self.fc = nn.Linear(5 * 5 * 64, 10)
#
#     # 前向传播
#     def forward(self, x):
#         out = self.conv1(x)
#         out = self.pool1(out)
#         out = self.conv2(out)
#         out = self.conv3(out)
#         out = self.pool2(out)
#         # 压缩成向量以供全连接
#         out = out.view(out.size(0), -1)
#         out = self.fc(out)
#         return out
