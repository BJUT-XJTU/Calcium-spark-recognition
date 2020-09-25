import struct
import csv

from openpyxl.compat import file
from skimage import data, exposure, img_as_float
from PIL import Image
import cv2
from numpy import *
import cv2 as cv
import os
import os.path
import math
import matplotlib as mpl
from scipy.optimize import curve_fit
from scipy import asarray as ar,exp
import numpy as np
import matplotlib.pyplot as plt
import re

from scipy.stats import norm

# 读取图片，计算pixel_size的具体函数
def read_resolution(file):
    #pixel_size_x = 0.8
    #pixel_size_t = 2.5
    with open(file, 'rb') as file_object:
        # print(file)
        pixel_size_x = 0.8
        pixel_size_t = 2.5

        file_object.seek(4, 0)
        #print(file_object)
        ifh_offset = struct.unpack("L", file_object.read(4))[0]
        #print(ifh_offset)
        file_object.seek(ifh_offset, 0)
        de_num = struct.unpack("H", file_object.read(2))[0]
        #print("de_num:",de_num)
        for index in range(de_num):
            de_tag = struct.unpack("H", file_object.read(2))[0]
            #print("de_tag",de_tag)
            if de_tag == 270:
                de_type = struct.unpack("H", file_object.read(2))[0]
                de_length = struct.unpack("L", file_object.read(4))[0]
                de_value_offset = struct.unpack("L", file_object.read(4))[0]
                length = str(de_length)
                file_object.seek(de_value_offset, 0)
                img_description = struct.unpack(length + "s", file_object.read(de_length))[0]
                img_desc = bytes.decode(img_description, errors="ignore")
                pos = img_desc.find("Width")
                size_x = int(img_desc[pos + 7:pos + 12])
                pos = img_desc.find("Length")
                size_t = int(img_desc[pos + 8:pos + 13])
                pos = img_desc.find("Voxel-Width")
                pixel_size_x = float(img_desc[pos + 17:pos + 26])
                pos = img_desc.find("Size-Height")
                pixel_size_t = float(img_desc[pos + 17:pos + 27])
                pixel_size_t /= (size_t * 0.001)
                print("1",pixel_size_x, pixel_size_t)

            else:
                file_object.seek(10, 1)
                #pixel_size_x = 0.8
                #pixel_size_t = 2.5
                #print(pixel_size_x, pixel_size_t)
                #return pixel_size_x, pixel_size_t
    print("2",pixel_size_x, pixel_size_t)
    return pixel_size_x, pixel_size_t

# 读取图片名称
def name_read():
    pic_load = "D:/研究生/导师/西安交大项目/实验用数据/原始数据/心房肌细胞_atrial/3.16实验数据/train_pic/pic_train_原图/"
    L = []
    for root, dirs, files in os.walk(pic_load):
        for file in files:
            name = os.path.join(root, file)
            # print(name)
            name1 = name[73:]
            #print(name1)
            L.append(name1)
    return L

#计算pixel_size的综合，最后获得一个矩阵，【图片名称，pixel_size_x,pixel_size_t】
def pixel_size_compute():
    name_list = name_read()
    pixel_size = np.zeros(shape=(100, 3))
    for i in range(len(name_list)):
        pic_name_list = "D:/研究生/导师/西安交大项目/实验用数据/原始数据/心房肌细胞_atrial/3.16实验数据/train_pic/pic_train_原图/" + name_list[i]

        pixel_size_x, pixel_size_t = read_resolution(pic_name_list)
        pixel_size[i] = [int(i + 1), round(pixel_size_x, 2), round(pixel_size_t, 2)]
    return pixel_size


def pic_data_analyse(left, top, right, bottom, pic_name):
    img = cv2.imread(pic_name)
    spark_box = img[top:bottom, left:right, 1]
    # 乘pixel_size 图片名称，pixel_size_x, pixel_size_t】

    # 1、分辨名字:正则提取出名字
    take_name = re.findall(r'z(.*)_', pic_name)
    # 2、乘对应的pixel_size
    pixel_size = pixel_size_compute()
    pixel_size_x = pixel_size[int(take_name) - 1][1]
    pixel_size_t = pixel_size[int(take_name) - 1][2]
    # 3、继续运算
    max = spark_box.max()
    for i in range(bottom - top):
        for j in range(right - left):
            if (spark_box[i, j] == max):
                max_index_x = i
                max_index_y = j
    # 列 35 19  18,19,20 列
    spark_space = spark_box[:, max_index_y - 1:max_index_y + 2]
    # print("列的大小", spark_space.shape)
    # 行 35 19  34,35,36 行
    spark_time = spark_box[max_index_x - 1:max_index_x + 2, :]
    # print("行的大小", spark_time.shape)




"""
#1、分辨名字:正则提取出名字
        take_name = re.findall(r'z(.*)_', pic_name)
        #2、乘对应的pixel_size
        pixel_size_x = self.pixel_size[int(take_name)-1][1]
        pixel_size_t = self.pixel_size[int(take_name)-1][2]
        #3、继续运算

"""