import torch  # 导入pytorch
from torch import nn, optim  # 导入神经网络与优化器对应的类
import torch.nn.functional as F
from torchvision import datasets, transforms ## 导入数据集与数据预处理的方法
from torch.utils.tensorboard import SummaryWriter
#%%
# 数据预处理：标准化图像数据，使得灰度数据在-1到+1之间
transform = transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.5,), (0.5,))])

# 下载Fashion-MNIST训练集数据，并构建训练集数据载入器trainloader,每次从训练集中载入64张图片，每次载入都打乱顺序
trainset = datasets.FashionMNIST('dataset/', download=False, train=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True)

# 下载Fashion-MNIST测试集数据，并构建测试集数据载入器trainloader,每次从测试集中载入64张图片，每次载入都打乱顺序
testset = datasets.FashionMNIST('dataset/', download=False, train=False, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=64, shuffle=True)
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
model = Classifier()

# 定义损失函数为负对数损失函数
criterion = nn.NLLLoss()

# 优化方法为Adam梯度下降方法，学习率为0.003
optimizer = optim.Adam(model.parameters(), lr=0.003)

# 对训练集的全部数据学习15遍，这个数字越大，训练时间越长
epochs = 15

# 将每次训练的训练误差和测试误差存储在这两个列表里，后面绘制误差变化折线图用
train_losses, test_losses = [], []

writer = SummaryWriter("log_dir")

print('开始训练')
for e in range(epochs):
    running_loss = 0

    # 对训练集中的所有图片都过一遍
    for images, labels in trainloader:
        # 将优化器中的求导结果都设为0，否则会在每次反向传播之后叠加之前的
        optimizer.zero_grad()

        # 对64张图片进行推断，计算损失函数，反向传播优化权重，将损失求和
        log_ps = model(images)
        loss = criterion(log_ps, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()

    # 每次学完一遍数据集，都进行以下测试操作
    else:
        test_loss = 0
        accuracy = 0
        # 测试的时候不需要开自动求导和反向传播
        with torch.no_grad():
            # 关闭Dropout
            model.eval()

            # 对测试集中的所有图片都过一遍
            for images, labels in testloader:
                # 对传入的测试集图片进行正向推断、计算损失，accuracy为测试集一万张图片中模型预测正确率
                log_ps = model(images)
                test_loss += criterion(log_ps, labels)
                ps = torch.exp(log_ps)
                top_p, top_class = ps.topk(1, dim=1)
                equals = top_class == labels.view(*top_class.shape)

                # 等号右边为每一批64张测试图片中预测正确的占比
                accuracy += torch.mean(equals.type(torch.FloatTensor))
        # 恢复Dropout
        model.train()
        # 将训练误差和测试误差存在两个列表里，后面绘制误差变化折线图用
        train_losses.append(running_loss / len(trainloader))
        test_losses.append(test_loss / len(testloader))

        writer.add_scalar('training loss', running_loss / len(trainloader), e)

        # 记录测试损失
        writer.add_scalar('testing loss', test_loss / len(testloader), e)

        # 记录模型分类准确率
        writer.add_scalar('accuracy', accuracy / len(testloader), e)

        print("训练集学习次数: {}/{}.. ".format(e + 1, epochs),
              "训练误差: {:.3f}.. ".format(running_loss / len(trainloader)),
              "测试误差: {:.3f}.. ".format(test_loss / len(testloader)),
              "模型分类准确率: {:.3f}".format(accuracy / len(testloader)))

writer.close()

model.eval()

# 定义图片预处理步骤
transform = transforms.Compose([
    transforms.Resize((28, 28)),  # 调整图片大小为 28x28 像素
    # transforms.Grayscale(),  # 转换为灰度图
    transforms.ToTensor(),  # 将 PIL 图片转换为 Tensor
    transforms.Normalize((0.5,), (0.5,))  # 归一化
])

# 加载网络图片
# 假设你已经有了图片的路径
from PIL import Image
image_path = 'image5.jpg'
image = Image.open(image_path)
image = image.convert('L')

# 应用预处理步骤
image_tensor = transform(image)  # 增加批次维度
image_tensor = image_tensor.reshape(28,28).numpy()
img = torch.from_numpy(image_tensor)
img = img.view(1, 784)

# 使用模型进行预测
with torch.no_grad():
    output = model.forward(img)
    _, predicted = torch.max(output, 1)

# 将预测的类别转换为类别名称
# 假设你有一个从索引到类别名称的字典
class_names = ['T-shirt', 'Pants', 'Pullover', 'Skirt', 'Coat', 'Sandals', 'Tank top', 'Sneakers', 'Bag', 'Boots']
predicted_class = class_names[predicted.item()]

print(f'预测的物品种类是: {predicted_class}')