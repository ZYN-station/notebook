# 该代码是进行的测试集t10k-images-idx3-ubyte和t10k-labels-idx1-ubyte的图像提取
# 对于训练集train-labels-idx1-ubyte和train-labels-idx1-ubyte，进行相应的替换就行
import numpy as np
import struct

from PIL import Image
import os

data_file = 'dataset/FashionMNIST/raw/t10k-images-idx3-ubyte'
fsize = os.path.getsize(data_file)
data_file_size = fsize
data_file_size = str(data_file_size - 16) + 'B'

data_buf = open(data_file, 'rb').read()

magic, numImages, numRows, numColumns = struct.unpack_from(
    '>IIII', data_buf, 0)
datas = struct.unpack_from(
    '>' + data_file_size, data_buf, struct.calcsize('>IIII'))
datas = np.array(datas).astype(np.uint8).reshape(
    numImages, 1, numRows, numColumns)

label_file = 'dataset/FashionMNIST/raw/t10k-labels-idx1-ubyte'

# It's 60008B, but we should set to 60000B
fsize = os.path.getsize(label_file)
label_file_size = fsize
label_file_size = str(label_file_size - 8) + 'B'

label_buf = open(label_file, 'rb').read()

magic, numLabels = struct.unpack_from('>II', label_buf, 0)
labels = struct.unpack_from(
    '>' + label_file_size, label_buf, struct.calcsize('>II'))
labels = np.array(labels).astype(np.int64)

datas_root = 'mnist_test'
if not os.path.exists(datas_root):
    os.mkdir(datas_root)

for i in range(10):
    file_name = datas_root + os.sep + str(i)
    if not os.path.exists(file_name):
        os.mkdir(file_name)

for ii in range(numLabels):
    img = Image.fromarray(datas[ii, 0, 0:28, 0:28])
    label = labels[ii]
    file_name = datas_root + os.sep + str(label) + os.sep + \
                'mnist_test_' + str(ii) + '.png'
    img.save(file_name)