# from Classifier2 import Classifier2
from torchvision import datasets, transforms
from Classifier_old import Classifier_old
from Classifier import Classifier
from torch.utils.tensorboard import SummaryWriter
import torch
from torch import nn, optim
from PIL import Image

# 数据预处理：标准化图像数据，使得灰度数据在-1到+1之间
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

# 下载Fashion-MNIST训练集数据，并构建训练集数据载入器trainloader
trainset = datasets.FashionMNIST('dataset/', download=False, train=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=200, shuffle=True)

# 下载Fashion-MNIST测试集数据，并构建测试集数据载入器testloader
testset = datasets.FashionMNIST('dataset/', download=False, train=False, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=200, shuffle=True)

# 检查是否有可用的GPU，如果有则使用GPU，否则使用CPU
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 初始化模型并移动到GPU
# model = Classifier2().to(DEVICE)
model = Classifier().to(DEVICE)

# 定义损失函数为负对数损失函数，并移动到GPU
criterion = nn.NLLLoss().to(DEVICE)

# 优化方法为Adam梯度下降方法，学习率为0.001
optimizer = optim.Adam(model.parameters(), lr=0.002)

# 训练轮次
epochs = 30

# 用于存储训练和测试误差
train_losses, test_losses = [], []

writer = SummaryWriter()

print('开始训练')
for e in range(epochs):
    running_loss = 0

    # 遍历训练集
    for images, labels in trainloader:
        # 将数据移动到GPU
        images, labels = images.to(DEVICE), labels.to(DEVICE)

        # 梯度归零
        optimizer.zero_grad()

        # 前向传播
        log_ps = model(images)

        # 计算损失
        loss = criterion(log_ps, labels)

        # 反向传播和优化
        loss.backward()
        optimizer.step()

        # 累加损失
        running_loss += loss.item()

    # 每个epoch结束后评估模型
    test_loss = 0
    accuracy = 0
    model.eval()  # 设置为评估模式
    with torch.no_grad():
        for images, labels in testloader:
            # 将数据移动到GPU
            images, labels = images.to(DEVICE), labels.to(DEVICE)

            # 前向传播
            log_ps = model(images)

            # 计算损失
            test_loss += criterion(log_ps, labels)

            # 计算准确率
            ps = torch.exp(log_ps)
            top_p, top_class = ps.topk(1, dim=1)
            equals = top_class == labels.view(*top_class.shape)
            accuracy += torch.mean(equals.type(torch.FloatTensor))

    model.train()  # 设置回训练模式
    train_losses.append(running_loss / len(trainloader))
    test_losses.append(test_loss / len(testloader))


    writer.add_scalars('Loss&Accurcay_new',
                       {'trainLoss':running_loss/len(trainloader),'testLoss':test_loss/len(testloader),'Accuracy':accuracy/len(testloader)},e)
    print(f"训练集学习次数: {e + 1}/{epochs}.. "
          f"训练误差: {running_loss / len(trainloader):.3f}.. "
          f"测试误差: {test_loss / len(testloader):.3f}.. "
          f"模型分类准确率: {accuracy / len(testloader):.3f}")

writer.close()
torch.save(model.state_dict(), 'model_new.pt')

# image = Image.open("images/image5.jpg")
#
# imagel = image.convert('L')
# # %%
# # 将图片转换为1X28X28的tensor
# transform = transforms.Compose(
#     [transforms.Resize((28, 28)), transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
# image_tensor = transform(imagel)
# print("调整后图像的形状", image_tensor.shape)
#
# with torch.no_grad():
#     output = model.forward(image_tensor)
# ps = torch.exp(output)
#
# top_p, top_class = ps.topk(1, dim=1)
# labellist = ['T恤', '裤子', '套衫', '裙子', '外套', '凉鞋', '汗衫', '运动鞋', '包包', '靴子']
# print(top_class)
# prediction = labellist[top_class]
# probability = float(top_p)
# print(f'神经网络猜测图片里是 {prediction}，概率为{probability * 100}%')