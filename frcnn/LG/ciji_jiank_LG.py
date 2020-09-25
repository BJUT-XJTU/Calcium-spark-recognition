import csv

from openpyxl.compat import file
from skimage import data, exposure, img_as_float
from PIL import Image
from sklearn.linear_model import LogisticRegression
from numpy import *
import cv2
from sklearn.tree  import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import MultinomialNB
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
import os
import os.path
import math
import matplotlib as mpl
from scipy.optimize import curve_fit
from scipy import asarray as ar,exp
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from scipy.stats import norm
import pandas as pd

from sklearn.metrics import confusion_matrix
from sklearn.metrics import recall_score
import matplotlib.pyplot as plt


def read_txt_to_csv():
    """file = open("healthy.txt", "r")
    list = file.readlines()  # 每一行数据写入到list中
    lists = []
    # 将txt文件转换成数组形式保存
    pic_num = 0
    for fields in list:
        fields = fields.strip()  # fields.strip()用来删除字符串两端的空白字符。
        fields = fields.strip("\n")  # fields.strip("[]")用来删除字符串两端方括号。
        fields = fields.split(",")  # fields.split(",")的作用是以逗号为分隔符，将字符串进行分隔。
        for i in range(10):
            fields[i] = fields[i].strip()  # fields.strip()用来删除字符串两端的空白字符。
        fields[0] = pic_num;
        # pic_num=pic_num+1;#图片编号
        lists.append(fields)
    lists.pop(0)  # 删除第一行
    tracks = np.array(lists, dtype=float)  # 将其转换成numpy的数组，并定义数据类型为float
    text_picture1_spark = tracks
    # print(text_picture1_spark)
    df = pd.DataFrame(text_picture1_spark)

    df.columns = ['111', '#', 'dF/F0', 'FWHM', 'rise time', 't50', 'FDHM', 'tau', 'x_pos', 't_pos']
    print(df.head(3))
    df = df.drop(['111', '#', 'tau', 'x_pos', 't_pos'], axis=1)
    print(df.head(3))
    df.to_csv("healthy.csv")

    file = open("stimulate.txt", "r")
    list = file.readlines()  # 每一行数据写入到list中
    lists = []
    # 将txt文件转换成数组形式保存
    pic_num = 0
    for fields in list:
        fields = fields.strip()  # fields.strip()用来删除字符串两端的空白字符。
        fields = fields.strip("\n")  # fields.strip("[]")用来删除字符串两端方括号。
        fields = fields.split(",")  # fields.split(",")的作用是以逗号为分隔符，将字符串进行分隔。
        for i in range(10):
            fields[i] = fields[i].strip()  # fields.strip()用来删除字符串两端的空白字符。
        fields[0] = pic_num;
        # pic_num=pic_num+1;#图片编号
        lists.append(fields)
    lists.pop(0)  # 删除第一行
    tracks = np.array(lists, dtype=float)  # 将其转换成numpy的数组，并定义数据类型为float
    text_picture1_spark = tracks
    # print(text_picture1_spark)
    df = pd.DataFrame(text_picture1_spark)

    df.columns = ['111', '#', 'dF/F0', 'FWHM', 'rise time', 't50', 'FDHM', 'tau', 'x_pos', 't_pos']
    print(df.head(3))
    df = df.drop(['111', '#', 'tau', 'x_pos', 't_pos'], axis=1)
    print(df.head(3))
    df.to_csv("stimulate.csv")"""

    file = open("nonhealthy.txt", "r")
    list = file.readlines()  # 每一行数据写入到list中
    lists = []
    # 将txt文件转换成数组形式保存
    pic_num = 0
    for fields in list:
        fields = fields.strip()  # fields.strip()用来删除字符串两端的空白字符。
        fields = fields.strip("\n")  # fields.strip("[]")用来删除字符串两端方括号。
        fields = fields.split(",")  # fields.split(",")的作用是以逗号为分隔符，将字符串进行分隔。
        for i in range(10):
            fields[i] = fields[i].strip()  # fields.strip()用来删除字符串两端的空白字符。
        fields[0] = pic_num;
        # pic_num=pic_num+1;#图片编号
        lists.append(fields)
    lists.pop(0)  # 删除第一行
    tracks = np.array(lists, dtype=float)  # 将其转换成numpy的数组，并定义数据类型为float
    text_picture1_spark = tracks
    # print(text_picture1_spark)
    df = pd.DataFrame(text_picture1_spark)

    df.columns = ['111', '#', 'dF/F0', 'FWHM', 'rise time', 't50', 'FDHM', 'tau', 'x_pos', 't_pos']
    print(df.head(3))
    df = df.drop(['111', '#', 'tau', 'x_pos', 't_pos'], axis=1)
    print(df.head(3))
    df.to_csv("nonhealthy.csv")

def read_txt():

    readcsv_0 = pd.read_csv('healthy.csv')
    readcsv_1 = pd.read_csv('nonhealthy.csv')
    readcsv_2 = pd.read_csv('stimulate.csv')
    print("健康的数据", readcsv_0.shape)
    print("病态的数据", readcsv_1.shape)
    print("刺激的数据", readcsv_2.shape)
    print(readcsv_0.head())
    print(readcsv_1.head())
    # data_all = pd.concat([readcsv_0, readcsv_1])
    # data_all.to_csv('healthy+stimulate.csv')
    return readcsv_0, readcsv_1, readcsv_2

def draw_fenbu():
    data_0, data_1, data_2 = read_txt()

    # df/f0
    plt.scatter(data_0['num'], data_0['dF/F0'], s=10)
    plt.scatter(data_1['num'], data_1['dF/F0'], s=10)
    plt.scatter(data_2['num'], data_2['dF/F0'], s=10)
    plt.legend(labels=["health", "nonhealth", "stimulate"])
    plt.title("dF/F0")
    plt.savefig("dFF0_分布1.png")
    plt.show()

    # rise time
    plt.scatter(data_0['num'], data_0['rise time'], s=10)
    plt.scatter(data_1['num'], data_1['rise time'], s=10)
    plt.scatter(data_2['num'], data_2['rise time'], s=10)
    plt.legend(labels=["health", "nonhealth", "stimulate"])
    plt.title("rise time")
    plt.savefig("rise time_分布1.png")
    plt.show()


    # t50
    plt.scatter(data_0['num'], data_0['t50'], s=10)
    plt.scatter(data_1['num'], data_1['t50'], s=10)
    plt.scatter(data_2['num'], data_2['t50'], s=10)
    plt.legend(labels=[ "health", "nonhealth", "stimulate"])
    plt.title("t50")
    plt.savefig("t50_分布1.png")
    plt.show()
 
    # FDHM
    plt.scatter(data_0['num'], data_0['FDHM'], s=10)
    plt.scatter(data_1['num'], data_1['FDHM'], s=10)
    plt.scatter(data_2['num'], data_2['FDHM'], s=10)
    plt.legend(labels=["health", "nonhealth", "stimulate"])
    plt.title("FDHM")
    plt.savefig("FDHM_分布1.png")
    plt.show()

    # FWHM
    plt.scatter(data_0['num'], data_0['FWHM'], s=10)
    plt.scatter(data_1['num'], data_1['FWHM'], s=10)
    plt.scatter(data_2['num'], data_2['FWHM'], s=10)
    plt.legend(labels=["health", "nonhealth", "stimulate"])
    plt.title("FWHM")
    plt.savefig("FWHM_分布1.png")
    plt.show()

def LG():
    print("LG:")
    readcsv = pd.read_csv('healthy+stimulate+nonhealthy.csv')
    print("数据:", readcsv.shape)
    print(readcsv.head(3))
    # print(readcsv.head(5))


    data = readcsv.iloc[:, 1:6]
    labels = readcsv.iloc[:, 6]

    """ x_train, x_test, y_train, y_test = train_test_split(data, labels, random_state=1, test_size=0.2)
    LR = LogisticRegression(solver='liblinear',multi_class = 'auto')
    LR.fit(x_train, y_train)
    score = LR.score(x_test, y_test)
    print(score)"""


    numb = 150

    LG_max = 0
    LG_max_i = 0
    LG_max_test_size = 0
    for i in range(numb):
        for j in range(3):
            k = (j + 1) / 10
            x_train, x_test, y_train, y_test = train_test_split(data, labels, random_state=i, test_size=k)
            LR = LogisticRegression(solver='liblinear',multi_class = 'auto')
            LR.fit(x_train, y_train)
            score = LR.score(x_test, y_test)
            # print("当前属于第", i ,"种划分，测试数据集为：", j/10  ,",测试结果为：",score)

            if (score > LG_max):
                LG_max_test_size = ((j + 1) / 10)
                LG_max = score
                LG_max_i = i

    print("测试数据的数量：", y_test.shape)
    print("LG max:", LG_max, ",LG_max_i:", LG_max_i, "LG_max_test_size:", LG_max_test_size)

#draw_fenbu()
LG()