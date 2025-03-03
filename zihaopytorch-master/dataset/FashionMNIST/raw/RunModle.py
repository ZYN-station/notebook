#%%
import matplotlib.pyplot as plt
from PIL import Image
import torchvision
from torchvision import transforms
from Classifier2 import Classifier2
import torch
from PIL import Image, ImageDraw, ImageFont
from torchvision import datasets, transforms ## 导入数据集与数据预处理的方法
import numpy as np


def getImageTensor(image,filename):

    transform = transforms.Compose([transforms.Resize((28, 28)),  # 调整图片大小为 28x28 像素
                                    transforms.Grayscale(),  # 转换为灰度图
                                    transforms.ToTensor(),  # 将 PIL 图片转换为 Tensor
                                    transforms.Normalize((0.5,), (0.5,))]) # 归一化

    image_tensor = transform(image).unsqueeze(0)

    to_pil = transforms.ToPILImage()

    # 假设tensor是你的图像Tensor
    image = to_pil(image_tensor.cpu().squeeze(0))
    # image.save('images/output_'+filename)  # 保存图像
    return image_tensor


#%%
#转换为灰度图像

# gray_image.show()
#%%
#加载模型
def predict(images_tensor,predictions=[],probabilitys = []):
    model_load = torch.load("model_3.pt", map_location=torch.device('cuda'), weights_only=True)
    model = Classifier2()
    model.load_state_dict(model_load)
    model.eval()
    for image_tensor in images_tensor:
        with torch.no_grad():
            output = model.forward(image_tensor)
        ps = torch.exp(output)
        print(ps)

        top_p, top_class = ps.topk(1, dim=1)
        labellist = ['T恤', '裤子', '套衫', '裙子', '外套', '凉鞋', '汗衫', '运动鞋', '包包', '靴子']
        prediction = labellist[top_class]
        probability = float(top_p)
        predictions.append(prediction)
        probabilitys.append(probability)
        print(f'神经网络猜测图片里是 {prediction}，概率为{probability * 100}%')
    return predictions,probabilitys
#%%
def DrawImage(image,prediction,probability):

    draw = ImageDraw.Draw(image)
    position = (50, 50)  # 文字位置
    font = ImageFont.truetype(font='arial.ttf', size=10)
    text_color = (255, 0, 0)  # 设置文本颜色，这里设置为红色
    text = f'The neural network guesses that the image is a {prediction} with a probability of {probability * 100}%.'  # 文字内容
    draw.text(position, text, font=font, fill=text_color)
    image.save('example_with_text.jpg')


import os
from PIL import Image

# 设置文件夹路径
folder_path = 'images'
images_tensor = []

# 遍历文件夹中的所有文件
for filename in os.listdir(folder_path):
    # 检查文件是否是图片（这里以.jpg和.png为例）
    if filename.endswith(".jpg") or filename.endswith(".png"):
        # 构建完整的文件路径
        file_path = os.path.join(folder_path, filename)
        # 打开图片
        with Image.open(file_path) as img:
            image_tensor = getImageTensor(img,filename)
            images_tensor.append(image_tensor)
            # DrawImage(img,prediction,probability)

predict(images_tensor)



