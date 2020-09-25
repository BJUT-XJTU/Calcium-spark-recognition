from PIL import Image
import numpy as np
import cv2
import cv2 as cv
import matplotlib.pyplot as plt   # 显示图片
import matplotlib.image as mpimg  # 读取图片
import os
from numpy import *
import os.path

import matplotlib.pyplot as plt
bgms=[]
bakeground=[]
pic_loads = "D:/研究生/导师/西安交大项目/实验用数据/原始数据/心房肌细胞_atrial/3.16实验数据/train_pic/pic_train_原图_标准化/"
pic_norm_load = "D:/研究生/导师/西安交大项目/实验用数据/原始数据/心房肌细胞_atrial/3.16实验数据/train_pic/pic_train_标准化/"
pic_load_jiequ_buchong = "D:/研究生/导师/西安交大项目/实验用数据/原始数据/心房肌细胞_atrial/3.16实验数据/train_pic/pic_train_原图_标准化_截取补充/"
pic_norm_load_3 = "D:/研究生/导师/西安交大项目/实验用数据/原始数据/心房肌细胞_atrial/3.16实验数据/train_pic/pic_train_原图_标准化_截取补充_3/"
pic_norm_load = "D:/研究生/导师/西安交大项目/实验用数据/原始数据/心房肌细胞_atrial/3.16实验数据/train_pic/pic_train_原图_标准化/"
pictest1="D:/pythonprogram/spark/frcnn/VOCdevkit/VOC2007/Annotations/"
pictest2="D:/pythonprogram/spark/frcnn/VOCdevkit/VOC2007/JPEGImages/"

def read_pic_name():
    L = []
    for root,dirs,files in os.walk(pictest1):
        for file in files:
            name = os.path.join(root,file)
            name = name[59:]
            #print(name)
            L.append(name)
    name_list1=L
    print("长度",len(name_list1))
    L = []
    for root, dirs, files in os.walk(pictest2):
        for file in files:
            name = os.path.join(root, file)
            name = name[58:]
            #print(name)
            L.append(name)
    name_list2 = L
    print("长度", len(name_list2))
    return name_list1,name_list2

def norm():
    name_list=read_pic_name()
    for num in range(1):
        pic_load=pic_load_jiequ_buchong+name_list[num]

        #1、读图片，换成数据格式，平滑处理
        image = Image.open(pic_load)
        img = np.array(image)  # 处理竖着的 41，39，64
        img = cv2.boxFilter(img, -1, (5, 5), normalize=1)

        #2、归一化处理
        #2.1、每一列取平均值
        mean_list = img.mean(axis=0)

        #2.2、判断平均值与其他数字大小 ;
        #2.3、背景点取平均值，作为基础背景值
        for i in range(len(mean_list)):
            for j in range(600):
                if img[j, i] <= mean_list[i]:
                    bakeground.append(img[j, i])
            bgm = mean(bakeground)
            bakeground.clear()
            bgms.append(bgm)

        #2.4、其他所有像素点除以该列的基础背景值
        img = img.astype(float)
        for i in range(len(mean_list)):
            for j in range(600):
                if img[j, i] != 0:
                    img[j, i] = img[j, i] / bgms[i]
                if img[j, i] == 0:
                    img[j, i] = 0
        max = img.max()
        for i in range(len(mean_list)):
            for j in range(600):
                img[j, i] = img[j, i] * 255 / max
        img = img.astype(int)

        # 3、转化为图片
        pic_new_name=pic_norm_load+name_list[num]

        cv2.imwrite(name_list[num], img)
        print("图片", num, "处理完成")
        #4、转化为3通道

def buchong_jiequ():
    # 补充+剪辑
    name_list = born_name()
    for i in range(len(name_list)):
        pic_address_old = pic_loads + name_list[i]  # 图片地址
        print(pic_address_old)
        im = Image.open(pic_address_old)  # 读图片，取大小
        img = np.array(im)
        # 计算大小
        width = im.size[0]
        height = im.size[1]
        # print("before:",width,height)
        cha_h = 9600 - height
        cha_w = 600 - width
        # print("需要补充：",cha_w,cha_h)
        # 上下左右
        a = cv2.copyMakeBorder(img, 1, cha_h - 1, 1, cha_w - 1, cv.BORDER_CONSTANT, value = 0)
        print(a.shape)
        j = 600
        k = 0
        for nn in range(15):
            pic_address_savejiancai_name = pic_load_jiequ_buchong+"z" + str(i+1) + "_" + str(nn+1) + ".png"
            region = a[k:j, 0:600]
            j = j + 600
            k = k + 600
            print(nn, region.shape)
            im1 = Image.fromarray(region)  # numpy 转 image类

            im1.save(pic_address_savejiancai_name)

def to_3():
    name_list = read_pic_name()
    for i in range(len(name_list)):
        name=pic_load_jiequ_buchong+name_list[i]
        save_name=pic_norm_load_3+name_list[i]
        image = Image.open(name)
        out = image.convert("RGB")  # 转三通道
        out.save(save_name)
        print("图片", i, "处理完成")

def born_name():
    namelist=[]
    i=1
    for i in range(100):
        if i <23:
            str1="z"+str(i+1)+"_norm.png"
            print(str1)
        if i>23:
            str1="z"+str(i+1)+"norm.png"
            print(str1)
        namelist.append(str1)
    return namelist


"""img = cv2.imread("STSCI_NGC4302_4298px1024.jpg")
print(img.shape)
print(img[1,1])
"""
"""
image = Image.open("z57_5.png")
out = image.convert("RGB")  # 转三通道
out.save("./t/z57_5.png")
print("图片", "处理完成")"""

"""image = Image.open("z1_1.png")
img = np.array(image)
print(img.shape)
print(img[1])

img = cv2.imread("z1_2.png")
print(img.shape)
print(img[1])"""

"""xmllist,piclist=read_pic_name()
for i in range(len(xmllist)):
    if (xmllist[i][:-3] == piclist[i][:-3]):
        print(xmllist[i], piclist[i], "一样")
    if(xmllist[i][:-3]!=piclist[i][:-3]):
        print("!!!!!!!!",xmllist[i],piclist[i],"不一样")
        break
"""