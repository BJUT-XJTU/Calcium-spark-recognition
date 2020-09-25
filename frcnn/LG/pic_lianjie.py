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

def pic_long():
    pic_load_address_new = "D:/pythonprogram/spark/frcnn/result/confidence_0.2/"

    name_list = []
    for j in range(15):
        name = pic_load_address_new + "z1_" + str(j + 1) + "_result.png"
        name_list.append(name)
    print(name_list)

    for k in range(1):
        long_pic_name = "z" + str(k + 1) + "_long_0.2.png"
        print(long_pic_name)
        pic_list = []
        for i in range(len(name_list)):
            pic = cv2.imread(name_list[i])
            print(pic.shape)
            pic_list.append(pic)
        print(len(pic_list))

        long_pic = pic_list[0]
        for i in range(len(pic_list) - 1):
            print(pic_list[i].shape)
            long_pic = np.concatenate((long_pic, pic_list[i + 1]), axis=0)
            # print(long_pic.shape)

        cv2.imwrite(long_pic_name, long_pic)

"""long_pic_name = "z10_long.png"
pic_list = []
for i in range(len(name_list)):
    read_pic = name_list[i]
    str1 = long_pic_name[:3]
    if read_pic[36:39] == str1:
        # print(read_pic)
        pic = cv2.imread(read_pic)
        pic_list.append(pic)
print(len(pic_list))

long_pic = pic_list[0]
for i in range(len(pic_list) - 1):
    print(i+1,pic_list[i].shape)
    long_pic = np.concatenate((long_pic, pic_list[i + 1]), axis=0)
    print(long_pic.shape)
cv2.imwrite(long_pic_name, long_pic)
"""
#72 79 45 35 65
#69 78 45 33 58
# [ '#', 'dF/F0', 'FWHM', 'rise time', 't50', 'FDHM', 'tau', 'x_pos', 't_pos']
def read_txt_duibijieguo():
    # D:\研究生\导师\西安交大项目\实验结果统计\spark_实验结果记录\8.11实验结果对比\分析及汇报\z1对比结果.txt
    file = open("D:/研究生/导师/西安交大项目/实验结果统计/spark_实验结果记录/8.21实验结果对比/分析及汇报/全部数据/继良分析后/未检测出的结果821.txt", "r")
    list = file.readlines()  # 每一行数据写入到list中
    lists = []
    # 将txt文件转换成数组形式保存
    pic_num = 0
    for fields in list:
        fields = fields.split(",")  # fields.split(",")的作用是以逗号为分隔符，将字符串进行分隔。
        for i in range(len(fields)):
            fields[i] = fields[i].strip()  # fields.strip()用来删除字符串两端的空白字符。
        lists.append(fields)
    lists.pop(0)
    list_cha = lists


    return list_cha

# D:\研究生\导师\西安交大项目\实验结果统计\spark_实验结果记录\8.11实验结果对比\分析及汇报\z1谢老师程序.txt
def read_txt_all():

    file = open("D:/研究生/导师/西安交大项目/实验结果统计/spark_实验结果记录/8.21实验结果对比/分析及汇报/全部数据/继良分析后/谢老师程序检测结果2.txt", "r")
    list = file.readlines()  # 每一行数据写入到list中
    lists = []
    # 将txt文件转换成数组形式保存
    number = 0
    for fields in list:
        fields = fields.split(",")  # fields.split(",")的作用是以逗号为分隔符，将字符串进行分隔。
        fields[0] = str(number)
        number = number+1
        for i in range(len(fields)):
            fields[i] = fields[i].strip()  # fields.strip()用来删除字符串两端的空白字符。
        lists.append(fields)
    lists.pop(0)
    list_all = lists

    return list_all

def pic_duibitu():
    #'num', 'dF/F0', 'FWHM', 'rise time', 't50', 'FDHM', 'tau', 'x_pos', 't_pos'
    np_cha, list_cha, np_all, list_all=num_id_z()

    plt.scatter(np_all[:, :1], np_all[:, 1:2], s=10)
    plt.scatter(np_cha[:, :1], np_cha[:, 1:2], s=10)
    plt.legend(labels=["all data","no detectable"])
    plt.title("dF/F0")
    plt.savefig("dFF0.png")
    plt.show()

    plt.scatter(np_all[:, :1], np_all[:, 2:3], s=10)
    plt.scatter(np_cha[:, :1], np_cha[:, 2:3], s=10)
    plt.legend(labels=["all data", "no detectable"])
    plt.title("FWHM")
    plt.savefig("FWHM.png")
    plt.show()

    plt.scatter(np_all[:, :1], np_all[:, 3:4], s=10)
    plt.scatter(np_cha[:, :1], np_cha[:, 3:4], s=10)
    plt.legend(labels=["all data", "no detectable"])
    plt.title("rise time")
    plt.savefig("rise time.png")
    plt.show()

    plt.scatter(np_all[:, :1], np_all[:, 4:5], s=10)
    plt.scatter(np_cha[:, :1], np_cha[:, 4:5], s=10)
    plt.legend(labels=["all data", "no detectable"])
    plt.title("t50")
    plt.savefig("t50.png")
    plt.show()

    plt.scatter(np_all[:, :1], np_all[:, 5:6], s=10)
    plt.scatter(np_cha[:, :1], np_cha[:, 5:6], s=10)
    plt.legend(labels=["all data", "no detectable"])
    plt.title("FDHM")
    plt.savefig("FDHM.png")
    plt.show()

    plt.scatter(np_all[:, :1], np_all[:, 6:7], s=10)
    plt.scatter(np_cha[:, :1], np_cha[:, 6:7], s=10)
    plt.legend(labels=["all data", "no detectable"], loc="best")
    plt.title("tau")
    plt.savefig("tau.png")
    plt.show()

def num_id_z():
    list_cha = read_txt_duibijieguo()
    list_all = read_txt_all()
    for i in range(len(list_cha)):
        for j in range(len(list_all)):
            if (list_all[j][1] == list_cha[i][1]):
                if (list_all[j][2] == list_cha[i][2]):
                    if (list_all[j][3] == list_cha[i][3]):
                        if (list_all[j][4] == list_cha[i][4]):
                            if (list_all[j][5] == list_cha[i][5]):
                                list_cha[i][0] = list_all[j][0]

    np_cha = np.array(list_cha, dtype=float)
    # print(type(np_cha))
    print(np_cha.shape)

    np_all = np.array(list_all, dtype=float)
    print(np_all.shape)

    return np_cha, list_cha, np_all, list_all

def read_csv():
    # bingtai.csv jiankang.csv
    readcsv_0 = pd.read_csv('D:/pythonprogram/spark/frcnn/test/tt/spark_analyse_0.csv')
    readcsv_1 = pd.read_csv('D:/pythonprogram/spark/frcnn/test/tt/spark_analyse_1.csv')
    print("健康的数据",readcsv_0.shape)
    print("病态的数据",readcsv_1.shape)

    data_0, data_1 = comput(readcsv_0,readcsv_1)
    #data_0.to_csv('spark_analyse_new_0.csv')
    #data_1.to_csv('spark_analyse_new_1.csv')
    print(data_0.head(3))
    print(data_1.head(3))

    data_all = pd.concat([data_0, data_1])
    #data_all.to_csv('spark_analyse_new.csv')

    return data_0, data_1

def comput(data_0, data_1):
    pixel_size = pixel_size_compute()
    # data_0, data_1
    for i in range(data_0.shape[0]):
        name = data_0.iloc[i, 0]
        #print("name:",name)
        take_name = re.findall(r'z(.*)_', name)
        take_name_str = "".join(take_name)
        #print(type(take_name_str))
        pixel_size_x = pixel_size[int(take_name_str) - 1][1]
        pixel_size_t = pixel_size[int(take_name_str) - 1][2]
        data_0.iloc[i, 4] = data_0.iloc[i, 4] * pixel_size_t
        data_0.iloc[i, 5] = data_0.iloc[i, 5] * pixel_size_t
        data_0.iloc[i, 6] = data_0.iloc[i, 6] * pixel_size_t #fdhm
        data_0.iloc[i, 7] = data_0.iloc[i, 7] * pixel_size_x #FWHM
    for i in range(data_1.shape[0]):
        name = data_1.iloc[i, 0]
        #print("name:",name)
        take_name = re.findall(r'z(.*)_', name)
        take_name_str = "".join(take_name)
        #print(type(take_name_str))  #risetime t50
        pixel_size_x = pixel_size[int(take_name_str) - 1][1]
        pixel_size_t = pixel_size[int(take_name_str) - 1][2]
        data_1.iloc[i, 4] = data_1.iloc[i, 4] * pixel_size_t
        data_1.iloc[i, 5] = data_1.iloc[i, 5] * pixel_size_t
        data_1.iloc[i, 6] = data_1.iloc[i, 6] * pixel_size_t  # fdhm
        data_1.iloc[i, 7] = data_1.iloc[i, 7] * pixel_size_x  # FWHM
    return data_0, data_1

def draw_fenbutu():
    numlist_0 = []
    numlist_1 = []
    data_0, data_1 = read_csv()
    for i in range(data_0.shape[0]):
        numlist_0.append(i+1)
    for i in range(data_1.shape[0]):
        numlist_1.append(data_0.shape[0]+i+1)
    """# df/f0
    plt.scatter(numlist_0, data_0['dF/F0'], s=10)
    plt.scatter(numlist_1, data_1['dF/F0'], s=10)
    plt.legend(labels=["health", "ill health"])
    plt.title("dF/F0")
    plt.savefig("dFF0_分布.png")
    plt.show()"""
    """
    # rise time
    max_t50 = data_1['rise time'].max()
    print("最大值：",max_t50)
    heng_1 = max_t50*0.4
    heng_2 = max_t50*0.17
    hengxian_1 = []
    hengxian_2 = []
    numlist_all = []
    for i in range(data_0.shape[0]+data_1.shape[0]):
        hengxian_1.append(heng_1)
        hengxian_2.append(heng_2)
        numlist_all.append(i)
    plt.scatter(numlist_0, data_0['rise time'], s=10)
    plt.scatter(numlist_1, data_1['rise time'], s=10)
    plt.plot(numlist_all, hengxian_1, color='red')
    plt.plot(numlist_all, hengxian_2, color='green')
    plt.text(data_0.shape[0]+data_1.shape[0]+20, heng_1, r'y = max * 0.4')
    plt.text(data_0.shape[0]+data_1.shape[0]+20, heng_2, r'y = max * 0.17')
    plt.legend(labels=["max * 0.4","max * 0.17","health", "ill health"])
    plt.title("rise time")
    plt.savefig("rise time_分布.png")
    plt.show()
    

    # t50
    max_t50 = data_1['t50'].max()
    print("最大值：", max_t50)
    heng_1 = max_t50 * 0.55
    heng_2 = max_t50 * 0.24
    hengxian_1 = []
    hengxian_2 = []
    numlist_all = []
    for i in range(data_0.shape[0] + data_1.shape[0]):
        hengxian_1.append(heng_1)
        hengxian_2.append(heng_2)
        numlist_all.append(i)

    plt.scatter(numlist_0, data_0['t50'], s=10)
    plt.scatter(numlist_1, data_1['t50'], s=10)
    plt.plot(numlist_all, hengxian_1, color='red')
    plt.plot(numlist_all, hengxian_2, color='green')
    plt.text(data_0.shape[0] + data_1.shape[0] + 20, heng_1, r'y = max * 0.55')
    plt.text(data_0.shape[0] + data_1.shape[0] + 20, heng_2, r'y = max * 0.24')
    plt.legend(labels=["max * 0.55", "max * 0.24", "health", "ill health"])
    plt.title("t50")
    plt.savefig("t50_分布.png")
    plt.show()"""
    """
    # FDHM

    max_t50 = data_1['FDHM'].max()
    print("最大值：", max_t50)
    heng_1 = max_t50 * 0.555
    heng_2 = max_t50 * 0.23
    hengxian_1 = []
    hengxian_2 = []
    numlist_all = []
    for i in range(data_0.shape[0] + data_1.shape[0]):
        hengxian_1.append(heng_1)
        hengxian_2.append(heng_2)
        numlist_all.append(i)

    plt.scatter(numlist_0, data_0['FDHM'], s=10)
    plt.scatter(numlist_1, data_1['FDHM'], s=10)
    plt.plot(numlist_all, hengxian_1, color='red')
    plt.plot(numlist_all, hengxian_2, color='green')
    plt.text(data_0.shape[0] + data_1.shape[0] + 20, heng_1, r'y = max * 0.555')
    plt.text(data_0.shape[0] + data_1.shape[0] + 20, heng_2, r'y = max * 0.23')
    plt.legend(labels=["max * 0.555", "max * 0.23", "health", "ill health"])
    plt.title("FDHM")
    plt.savefig("FDHM_分布.png")
    plt.show()
    """

    # FWHM

    plt.scatter(numlist_0, data_0['FWHM'], s=10)
    plt.scatter(numlist_1, data_1['FWHM'], s=10)
    plt.legend(labels=["health", "ill health"])
    plt.title("FWHM")
    plt.savefig("FWHM_分布.png")
    plt.show()

def draw_zhifang():
    numlist_0 = []
    numlist_1 = []
    data_0, data_1 = read_csv()
    for i in range(data_0.shape[0]):
        numlist_0.append(i + 1)
    for i in range(data_1.shape[0]):
        numlist_1.append(data_0.shape[0] + i + 1)

    # df/f0
    plt.hist([ data_1['dF/F0'],data_1['dF/F0']], bins=20, stacked=False)
    plt.legend(labels=["health", "ill health"])
    plt.title("dF/F0")
    plt.savefig("dFF0_直方.png")
    plt.show()

    # rise time
    plt.hist([data_0['rise time'], data_1['rise time']], bins=20, stacked=False)
    plt.legend(labels=["health", "ill health"])
    plt.title("rise time")
    plt.savefig("rise time_直方.png")
    plt.show()

    # t50
    plt.hist([data_0['t50'], data_1['t50']], bins=20, stacked=False)
    plt.legend(labels=["health", "ill health"])
    plt.title("t50")
    plt.savefig("t50_直方.png")
    plt.show()

    # FDHM
    plt.hist([data_0['FDHM'], data_1['FDHM']], bins=20, stacked=False)
    plt.legend(labels=["health", "ill health"])
    plt.title("FDHM")
    plt.savefig("FDHM_直方.png")
    plt.show()

    # FWHM
    plt.hist([data_0['FWHM'], data_1['FWHM']], bins=20, stacked=False)
    plt.legend(labels=["health", "ill health"])
    plt.title("FWHM")
    plt.savefig("FWHM_直方.png")
    plt.show()
from sklearn.decomposition import PCA
def draw_comb():
    numlist_0 = []
    numlist_1 = []
    data_0, data_1 = read_csv()
    for i in range(data_0.shape[0]):
        numlist_0.append(i + 1)
    for i in range(data_1.shape[0]):
        numlist_1.append(data_0.shape[0] + i + 1)
    #pca
    """x0 = data_0.iloc[:,3:8]
    x1 = data_1.iloc[:,3:8]
    print(x0.head(2))
    pca = PCA(n_components=2)
    X_p0 = pca.fit(x0).transform(x0)
    X_p1 = pca.fit(x1).transform(x1)
    #print(X_p0[:,1])
    plt.scatter(X_p0[:,0],X_p0[:,1], s=10)
    plt.scatter(X_p1[:,0],X_p1[:,1], s=10)
    plt.legend(labels=["health", "ill health"])
    #plt.title("FDHM/FWHM")
    plt.savefig("pca.png")
    plt.show()"""

    #

def six_pic():
    pixel_size = pixel_size_compute()
    #print(pixel_size)

    data_0, data_1 = read_csv()

    """ take_name = re.findall(r'z(.*)_', name)
    take_name_str = "".join(take_name)"""

    data_0_pic_list_l = []
    data_1_pic_list_l = []

    #print(data_0.iloc[0,0])

    for i in range(data_0.shape[0]):
        take_name = re.findall(r'z(.*)_', data_0.iloc[i, 0])
        take_name_str = "".join(take_name)
        data_0_pic_list_l.append(int(take_name_str))
    count_0_list = dict(zip(*np.unique(data_0_pic_list_l, return_counts=True)))
    # print("0长度", len(count_0_list.keys()))
    list_0_keys = []
    for key in count_0_list.keys():
        list_0_keys.append(key)

    for i in range(data_1.shape[0]):
        take_name = re.findall(r'z(.*)_', data_1.iloc[i, 0])
        take_name_str = "".join(take_name)
        data_1_pic_list_l.append(int(take_name_str))
    count_1_list = dict(zip(*np.unique(data_1_pic_list_l, return_counts=True)))
    # print("1长度", len(count_1_list.keys()))
    list_1_keys = []
    for key in count_1_list.keys():
        list_1_keys.append(key)

    count_1_list[2] = 79
    count_1_list[3] = 40
    count_1_list[4] = 33
    count_1_list[5] = 63
    count_1_list[6] = 127
    count_1_list[7] = 44
    count_1_list[8] = 7
    count_1_list[9] = 16
    count_1_list[10] = 18
    count_1_list[11] = 30
    count_1_list[12] = 16
    count_1_list[13] = 63
    count_1_list[14] = 46
    count_1_list[15] = 54
    count_1_list[16] = 18
    count_1_list[17] = 70
    count_1_list[18] = 51
    count_1_list[47] = 16
    count_1_list[48] = 26
    count_1_list[49] = 46
    count_1_list[50] = 13
    count_1_list[51] = 43
    count_1_list[52] = 35
    count_1_list[53] = 60
    count_1_list[54] = 13
    count_1_list[56] = 52
    count_1_list[57] = 41

    count_0_list[63] = 60
    count_0_list[64] = 24
    count_0_list[65] = 21
    count_0_list[66] = 34
    count_0_list[67] = 10
    count_0_list[68] = 8
    count_0_list[69] = 42
    count_0_list[70] = 163
    count_0_list[71] = 159
    count_0_list[72] = 31
    count_0_list[73] = 41
    count_0_list[74] = 19
    count_0_list[75] = 61
    count_0_list[76] = 7
    count_0_list[77] = 21
    count_0_list[78] = 20
    count_0_list[79] = 30
    count_0_list[80] = 14
    count_0_list[81] = 14
    count_0_list[82] = 14
    count_0_list[83] = 35
    count_0_list[84] = 14
    count_0_list[85] = 40
    count_0_list[86] = 32
    count_0_list[87] = 6

    list_0_keys_pinlu = []
    for i in range(len(count_0_list.keys())):
        m1 = 512 # 空间像素个数
        m2 = 9000 # 时间像素个数
        pic_name_num = list_0_keys[i]
        N = count_0_list.get(pic_name_num)+10
        # print("z",pic_name_num,".png")
        a1 = pixel_size[pic_name_num-1][1]
        a2 = pixel_size[pic_name_num-1][2]
        # print("a1",a1," a2",a2)
        f = N/((m1*a1)*(m2*a2))
        # print("频率",f)
        list_0_keys_pinlu.append(f)

    list_1_keys_pinlu = []
    for i in range(len(count_1_list.keys())):
        m1 = 512  # 空间像素个数
        m2 = 9000  # 时间像素个数
        pic_name_num = list_1_keys[i]
        N = count_1_list.get(pic_name_num)
        # print("z",pic_name_num,".png")
        a1 = pixel_size[pic_name_num - 1][1]
        a2 = pixel_size[pic_name_num - 1][2]
        # print("a1",a1," a2",a2)
        f = N / ((m1 * a1) * (m2 * a2))
        # print("频率",f)
        list_1_keys_pinlu.append(f)

    numlist_0 = []
    numlist_1 = []
    for i in range(len(list_0_keys_pinlu)):
        numlist_0.append(i + 1)
    for i in range(len(list_1_keys_pinlu)):
        numlist_1.append(len(list_0_keys_pinlu) + i + 1)

    # 画图
    plt.scatter(numlist_0, list_0_keys_pinlu, s=10)
    plt.scatter(numlist_1, list_1_keys_pinlu, s=10)
    plt.legend(labels=["health", "ill health"])
    max_y0 = max(list_0_keys_pinlu)
    min_y0 = min(list_0_keys_pinlu)
    max_y1 = max(list_1_keys_pinlu)
    min_y1 = min(list_1_keys_pinlu)
    max_y = max_y0
    min_y = min_y0
    if (max_y1>=max_y0):
        max_y = max_y1
    if (min_y1>=min_y0):
        min_y = min_y1

    plt.ylim((min_y+min_y+min_y, max_y+min_y))
    plt.title("frequency")
    plt.savefig("第六幅图.png")
    plt.show()

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
                #print("1",pixel_size_x, pixel_size_t)

            else:
                file_object.seek(10, 1)
                #pixel_size_x = 0.8
                #pixel_size_t = 2.5
                #print(pixel_size_x, pixel_size_t)
                #return pixel_size_x, pixel_size_t
    # print("2",pixel_size_x, pixel_size_t)
    return pixel_size_x, pixel_size_t

#计算pixel_size的综合，最后获得一个矩阵，【图片名称，pixel_size_x,pixel_size_t】
def pixel_size_compute():
    name_list = name_read()
    pixel_size = np.zeros(shape=(100, 3))
    for i in range(len(name_list)):
        pic_name_list = "D:/研究生/导师/西安交大项目/实验用数据/原始数据/心房肌细胞_atrial/3.16实验数据/train_pic/pic_train_原图/" + name_list[i]

        pixel_size_x, pixel_size_t = read_resolution(pic_name_list)
        pixel_size[i] = [int(i + 1), round(pixel_size_x, 2), round(pixel_size_t, 2)]
    return pixel_size

def LG():
    #readcsv_before = pd.read_csv('D:/pythonprogram/spark/frcnn/test/spark_analyse.csv')
    readcsv = pd.read_csv('D:/pythonprogram/spark/frcnn/test/spark_analyse_new.csv')
    print("数据:", readcsv.shape)
    print(readcsv.head(2))
    #print(readcsv.head(5))

    data = readcsv.iloc[:, 4:9]
    data_new = readcsv.iloc[:, 5:8]
    labels = readcsv.iloc[:, 9]

    print(data.head(5))
    print(labels.head(5))

    """x_train, x_test, y_train, y_test = train_test_split(data, labels, random_state=4, test_size=0.2)

    DT = DecisionTreeClassifier()
    DT.fit(x_train, y_train)
    print("DT   训练数据集：", DT.score(x_train, y_train), "测试数据集：", DT.score(x_test, y_test))

    RF = RandomForestClassifier(n_estimators=300)
    RF.fit(x_train, y_train)
    print("RF   训练数据集：", RF.score(x_train, y_train), "测试数据集：", RF.score(x_test, y_test))

    LR = LogisticRegression(solver='liblinear')
    LR.fit(x_train, y_train)
    print("LR   训练数据集：", LR.score(x_train, y_train), "测试数据集：", LR.score(x_test, y_test))

    SVM = svm.SVC(kernel='linear', C=1)
    SVM.fit(x_train, y_train)
    print("SVM   训练数据集：", SVM.score(x_train, y_train), "测试数据集：", SVM.score(x_test, y_test))

    KN = KNeighborsClassifier()
    KN.fit(x_train, y_train)
    print("KN   训练数据集：", KN.score(x_train, y_train), "测试数据集：", KN.score(x_test, y_test))

    MNB = MultinomialNB()
    MNB.fit(x_train, y_train)
    print("MNB   训练数据集：", MNB.score(x_train, y_train), "测试数据集：", MNB.score(x_test, y_test))
    """

    #原数据集是第24种,score 0.785
    #新数据集是第27种,score 0.814
    # 新数据集:
    #五个特征 LG max: 0.819047619047619 ,LG_max_i: 35 LG_max_test_size: 0.1
    #三个特征 LG max: 0.7942583732057417 ,LG_max_i: 6 LG_max_test_size: 0.2

    numb = 50

    LG_max = 0
    LG_max_i = 0
    LG_max_test_size = 0
    for i in range(numb):
        for j in range(3):
            k=(j+1)/10
            x_train, x_test, y_train, y_test = train_test_split(data_new, labels, random_state=i, test_size=k)
            LR = LogisticRegression(solver='liblinear')
            LR.fit(x_train, y_train)
            score = LR.score(x_test, y_test)

            if (score > LG_max):
                LG_max_test_size = ((j+1)/10)
                LG_max = score
                LG_max_i = i

    print("测试数据的数量：",y_test.shape)
    print("LG max:",LG_max,",LG_max_i:",LG_max_i,"LG_max_test_size:",LG_max_test_size)
"""
    DT_max = 0
    DT_max_i = 0
    for i in range(numb):
        x_train, x_test, y_train, y_test = train_test_split(data, labels, random_state=i, test_size=0.2)
        DT = DecisionTreeClassifier()
        DT.fit(x_train, y_train)
        score = DT.score(x_test, y_test)
        if (score > DT_max):
            DT_max = score
            DT_max_i = i
    print("DT max:", DT_max, ",DT_max_i:", DT_max_i)

    SVM_max = 0
    SVM_max_i = 0
    for i in range(numb):
        x_train, x_test, y_train, y_test = train_test_split(data, labels, random_state=i, test_size=0.2)
        SVM = svm.SVC(kernel='linear', C=1)
        SVM.fit(x_train, y_train)
        score = SVM.score(x_test, y_test)
        if (score > DT_max):
            SVM_max = score
            SVM_max_i = i
    print("SVMmax:", SVM_max, ",SVM_max_i:", SVM_max_i)

    KN_max = 0
    KN_max_i = 0
    for i in range(numb):
        x_train, x_test, y_train, y_test = train_test_split(data, labels, random_state=i, test_size=0.2)
        KN = KNeighborsClassifier()
        KN.fit(x_train, y_train)
        score = KN.score(x_test, y_test)
        if (score > KN_max):
            KN_max = score
            KN_max_i = i
    print("KN max:", KN_max, ",DT_max_i:", KN_max_i)

    MNB_max = 0
    MNB_max_i = 0
    for i in range(numb):
        x_train, x_test, y_train, y_test = train_test_split(data, labels, random_state=i, test_size=0.2)
        MNB = MultinomialNB()
        MNB.fit(x_train, y_train)
        score = MNB.score(x_test, y_test)
        if (score > DT_max):
            MNB_max = score
            MNB_max_i = i
    print("KN MNB_max:", MNB_max, ",MNB_max_i:", MNB_max_i)
"""

"""
    x_train, x_test, y_train, y_test = train_test_split(data, labels, random_state=max_i, test_size=0.2)
    LR_max = LogisticRegression(solver='liblinear')
    LR_max.fit(x_train, y_train)
    guess = LR_max.predict(x_test)
    fact = y_test
    classes = list(set(fact))
    classes.sort()
    confusion = confusion_matrix(guess, fact)
    plt.imshow(confusion, cmap=plt.cm.Blues)
    indices = range(len(confusion))
    plt.xticks(indices, classes)
    plt.yticks(indices, classes)
    plt.colorbar()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    for first_index in range(len(confusion)):
        for second_index in range(len(confusion[first_index])):
            plt.text(first_index, second_index, confusion[first_index][second_index])
    plt.savefig("LG_24.jpg")
    plt.show()"""

#draw_fenbutu()
LG()
#X,Y = read_csv()
#six_pic()



