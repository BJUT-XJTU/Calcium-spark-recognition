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

from scipy.stats import norm

"""for i in range(31):
    if(i >= 10):
        j = i/10
        image1_new = image1.point(lambda p: p * j)
        name = "z33_5_"+str(j)+".png"
        image1_new.save(name)

"""



"""
pic_load_address_new="D:/pythonprogram/spark/frcnn/pic_longpic/"
L = []
for root,dirs,files in os.walk(pic_load_address_new):
    for file in files:
        name = os.path.join(root,file)
        #print(name)
        name1 = name[34:]
        print(name1)
        L.append(name1)
name_list=L
"""

#归一化
"""
image = Image.open("z1_1.png")

photo = np.array(image, dtype=np.float64)
print(photo.shape)
# 图片预处理，归一化
p = np.expand_dims(photo, 0)
print(p.shape)
print(p.size)
B = np.reshape(p,(600,600,3))
cv2.imwrite("p.png", B)

photo = preprocess_input(p)
print("photo")
print(photo.shape)
print(type(photo))


B = np.reshape(photo,(600,600,3))
print(B.shape)

cv2.imwrite("photo.png", B)
"""


def pic_data_analyse(left, top, right, bottom, pic_name):
    img = cv2.imread(pic_name)
    spark_box = img[top:bottom, left:right, 1]
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

    spark_space_means = []
    spark_space_mean_x = []
    spark_time_means = []
    spark_time_mean_x = []

    # 处理列的平均值
    for i in range(spark_space.shape[0]):
        spark_space_means.append(int(spark_space[i].mean()))
        spark_space_mean_x.append(i)

    # 处理行的平均值
    for i in range(spark_time.shape[1]):
        meanx = (int(spark_time[0, i]) + int(spark_time[1, i]) + int(spark_time[2, i])) / 3
        spark_time_means.append(int(meanx))
        spark_time_mean_x.append(i)

    amplitude, FDHM, T50, tRise = space_alldata_draw(spark_space_means, spark_space_mean_x, pic_name)
    plt.savefig(pic_space_name)
    plt.show()

    FWHM = time_alldata_draw(spark_time_mean_x, spark_time_means)
    #plt.savefig(pic_time_name)
    #plt.show()

    # FWHM, rise time, t50, FDHM, x_pos, t_pos
    x_pos = max_index_x+top
    y_pos = max_index_y+left

    print("幅度:", amplitude, "时间半宽度FDHM:", FDHM,  "空间半宽度FWHM:", FWHM)
    print("下降一半的时间T50:", T50, "上升相时间rise Rise:", tRise)
    print("x_pos:", x_pos, "y_pos:", y_pos)

# 高斯函数 时间
def gaussian(x, *param):
    return param[0] * np.exp(-np.power(x - param[1], 2.) / (2 * np.power(param[2], 2.)))

# 前半段函数 空间
def fitexp_sr1(x,*param):
    f = param[1]*(1-exp(-(1/param[2])*(x-param[3])))+param[0]
    num = 0
    for i in range(len(x)):
        if x[i] < param[3]:
            num = num + 1
    if num >= 1:
        for i in range(num - 1):
            f[i] = param[0]
    if num == 0:
        for i in range(num + 1):
            f[i] = param[0]
    return f

# 后半段函数 空间
def fitexp_sr2(x,*param):
    bx = exp(param[1]*x)
    return param[0]*bx+param[2]

# 尝试全过程使用一种函数 空间
def fitexp_all(x,*param):
    f = param[1]*(1-exp(-(1/param[2])*(x-param[3])))+param[0]
    g = param[4]*exp(-(1/param[5])*(x-param[6]))+param[7]
    fp = 0
    for i in range(len(x)):
        if x[i] < param[3]:
            fp = fp + 1
    if fp > 0:
        for i in range(fp-1):
            f[i] = param[0]
    gp = 0
    for i in range(len(f)):
        if f[i] <= g[i]:
            gp = gp+1
    if gp >= 0:
        for i in range(len(x)):
            if i == gp:
                f[i] = g[i]
    return f

# 空间 全部点画图
def space_alldata_draw(spark_space_means, spark_space_mean_x, pic_name):
    # x拟合
    spark_space_means_np = np.array(spark_space_means)
    max = spark_space_means_np.max()
    for i in range(bottom - top):
        if (spark_space_means_np[i] == max):
            max_index = i

    # max_index 作为分水岭
    y = spark_space_means_np
    X_y1 = np.array(spark_space_means_np[0:max_index])
    X_y2 = np.array(spark_space_means_np[max_index :])
    x = np.array(spark_space_mean_x)
    X_x1 = x[0:max_index]
    X_x2 = x[max_index:]

    plt.scatter(x, y, s=20, c="red", marker='o')
    plt.scatter(max_index, max, s=50, c="green", marker='o')

    N = norm(pic_name)

    # X 前半段画图 返回最后一个点
    x_last, y_last, y_qian= x_fitexp_sr1(X_x1, X_y1)

    # X 后半段画图 返回第一个点
    x_first, y_first, y_hou = x_fitexp_sr2(X_x2, X_y2, N)

    # X 分析
    y_quanbunihe = np.concatenate((y_qian,y_hou),axis=0)
    amplitude, FDHM, T50, tRise = pic_quxian_analyse_space(x,y_quanbunihe)

    # 两点链接画图
    first_last(x_last, y_last, x_first, y_first)

    # 整段画图
    # x_all_fitexp(x, y, max_index, max, N)
    return amplitude, FDHM, T50, tRise

# 时间 全部点画图
def time_alldata_draw(spark_time_mean_x, spark_time_means):
    x = np.array(spark_time_means)
    y = np.array(spark_time_mean_x)
    plt.scatter(y, x, s=20, c="red", marker='o')
    # Y 画图 高斯拟合
    FWHM = y_nihe(y, x)
    return FWHM

# x 前半段画图 拟合图
def x_fitexp_sr1(x, y):
    # 前半段数据分析
    # 1、将前半段的各个点求平均值M，
    # 2、然后对小于平均值M的各个点再求平均值M1，初始值a[0] = M1。
    # 3、然后再看前半段这些点中，从前向后，以第一个点的序号为0，找到最后一个小于M1的点的序号n，初始值a[3] = n。
    # 4、初始值a[1] = (前半段最后一个点的值 - M1）×1.5
    # 5、初始值a[2]=前半段的点的个数减去n。
    # plt.plot(x, y)
    y_mean = y.mean()
    list_min_mean = []
    for i in range(len(y)):
        if y[i] < y_mean:
            list_min_mean.append(y[i])
            n = i
    M1 = np.array(list_min_mean).mean()
    M4 = n
    M2 = (y[-2]-M1)*1.5
    M3 = len(y)-n
    popt, pcov = curve_fit(fitexp_sr1, x, y, p0=[[M1, M2, M3, M4]]) #   [193, 16, 29, 60]
    # print("前半段拟合参数",popt)
    y_nihe = fitexp_sr1(x, *popt)
    plt.plot(x, y_nihe)
    x_last = x[-1]
    y_last = y_nihe[-1]

    return x_last, y_last, y_nihe

# x 后半段画图 拟合图
def x_fitexp_sr2(x, y, N):
    # 初始值的设定：A[0]= 后半段第一个点的值-N, A[1]=-1.5/n1。其中n1为后半段的点的个数
    # print(max_index, max)
    # print(y[1])
    # print("N:", N , "后半段第一个值:", y[1])
    M1 = y[1]-N
    M2 = -1.5/(len(x)-1)
    M3 = N
    popt, pcov = curve_fit(fitexp_sr2, x, y, p0=[M1, M2, M3])
    # print("后半段拟合参数",popt)
    y_nihe = fitexp_sr2(x, *popt)
    plt.plot(x, y_nihe)
    x_first = x[0]
    y_first = y_nihe[0]

    return x_first, y_first, y_nihe

# y拟合 高斯 画图
def y_nihe(x, y):
    popt, pcov = curve_fit(gaussian, x, y, p0=[1, 1, 1])
    # print("高斯参数",popt)
    y_nihe = gaussian(x, *popt)
    plt.plot(x, y_nihe)
    # 分析曲线
    FWHM = pic_quxian_analyse_time(x,y_nihe)
    return FWHM

# 空间 全部点 拟合图
"""def x_all_fitexp(x, y, max_index, max, N):
    # 这样一共8个参数：a[0]~a[7]。
    # 前四个的初始值和上面a[0]~a[3]的一样即可。
    # 初始值a[7] = 上面说的那个N，
    # a[4] = 峰值 - N，
    # a[6] = 峰值对应的点的序号，
    # a[5] = -1.5 / n1。
    # 其中n1为峰值后面的点的个数
    y_mean = y.mean()
    list_min_mean = []
    for i in range(len(y)):
        if y[i] < y_mean:
            list_min_mean.append(y[i])
            n = i
    M1 = np.array(list_min_mean).mean()
    M4 = n
    M2 = (y[-2] - M1) * 1.5
    M3 = len(y) - n
    M5 = max - N
    M6 = -1.5 / (len(x)-max_index)
    M7 = max_index
    M8 = N
    popt, pcov = curve_fit(fitexp_all, x, y, p0=[M1,M2,M3,M4,M5,M6,M7,M8])
    print(popt)
    plt.plot(x, fitexp_all(x, *popt))
    print("全部拟合")"""

# 首尾链接画图
def first_last(x_last, y_last, x_first, y_first):
    x_lianjie = []
    y_lianjie = []
    x_lianjie.append(x_last)
    x_lianjie.append(x_first)
    y_lianjie.append(y_last)
    y_lianjie.append(y_first)
    plt.plot(x_lianjie, y_lianjie)

# 空间曲线分析
def pic_quxian_analyse_space(x, y_nihe):
    # dF / F0, FWHM, rise
    # time, t50, FDHM, x_pos, t_pos
    print("分析：")
    global Fpeak
    Fpeak = y_nihe.max()
    Fpeak_x = 0
    for i in range(len(x)):
        if y_nihe[i]==Fpeak:
            Fpeak_x=i
    F0 = (y_nihe[0]+y_nihe[-1])/2
    mudium = (Fpeak+F0)/2

    """f0=[]
    fpeak1=[]
    m=[]
    for i in range(len(x)):
        m.append(mudium)
        f0.append(F0)
        fpeak1.append(Fpeak)
    plt.plot(x,m)
    plt.plot(x, f0)
    plt.plot(x, fpeak1)"""

    amplitude = (Fpeak-F0)/F0   # df/d0 幅度
    amplitude = round(amplitude, 2)
    # print("幅度：",amplitude)
    for i in range(Fpeak_x):
        if y_nihe[i] <= mudium:
            x_xiao = i
            x_da = i+1
    x_index1 = (mudium-y_nihe[x_xiao])/(y_nihe[x_da] - y_nihe[x_xiao])
    x_index1 = x_xiao+x_index1
    for i in range(len(x)):
        if i > Fpeak_x:
            if y_nihe[i] >= mudium:
                x_da = i
                x_xiao = i + 1
    x_index2 = (mudium - y_nihe[x_xiao]) / (y_nihe[x_da] - y_nihe[x_xiao])
    x_index2 = x_da + x_index2
    FDHM = x_index2-x_index1
    FDHM = round(FDHM, 2)
    # print("FDHM", FDHM)  # 时间半函数
    T50 = x_index2-Fpeak_x  # 下降一半时间
    T50 = round(T50, 2)
    # print("T50", T50)
    tRise1=0
    for i in range(len(y_nihe)-1):
        if y_nihe[i] == y_nihe[i+1]:
            tRise1 = i
    tRise = Fpeak_x - 0
    tRise = round(tRise, 2)
    # print("tRise", tRise)
    return amplitude, FDHM, T50, tRise

# 时间曲线分析
def pic_quxian_analyse_time(x, y_nihe):
    # dF / F0, FWHM, rise
    # time, t50, FDHM, x_pos, t_pos
    # print("分析：")
    Fpeak= y_nihe.max()
    Fpeak_x = 0
    for i in range(len(x)):
        if y_nihe[i] == Fpeak:
            Fpeak_x = i
    F0 = (y_nihe[0] + y_nihe[-1]) / 2
    mudium = (Fpeak + F0) / 2
    # amplitude = (Fpeak - F0) / F0  # df/d0 幅度
    # print("幅度：", amplitude)
    for i in range(Fpeak_x):
        if y_nihe[i] <= mudium:
            x_xiao = i
            x_da = i + 1
    x_index1 = (mudium - y_nihe[x_xiao]) / (y_nihe[x_da] - y_nihe[x_xiao])
    x_index1 = x_xiao + x_index1
    for i in range(len(x)):
        if i >= Fpeak_x:
            if y_nihe[i] >= mudium:
                x_da = i
                x_xiao = i + 1
    x_index2 = (mudium - y_nihe[x_xiao]) / (y_nihe[x_da] - y_nihe[x_xiao])
    x_index2 = x_da + x_index2
    FWHM = x_index2 - x_index1
    FWHM = round(FWHM, 2)
    # print("FWHM", FWHM)  # 时间半函数
    #T50 = x_index2 - Fpeak_x  # 下降一半时间
    #print("T50", T50)
    #tRise1=0
    #tRise = Fpeak_x - tRise1
    #print("tRise", tRise)
    return FWHM

# 计算像素点
def norm(name):
    # 1、读图片，换成数据格式，平滑处理
    image = Image.open(name)
    img = np.array(image)  # 处理竖着的 41，39，64
    img = img[:, :, 1]
    img = cv2.boxFilter(img, -1, (5, 5), normalize=1)
    bgms = []
    bakeground = []
    # 2、归一化处理
    # 2.1、每一列取平均值
    mean_list = img.mean(axis=0)
    # 2.2、判断平均值与其他数字大小 ;
    # 2.3、背景点取平均值，作为基础背景值

    for i in range(len(mean_list)):
        for j in range(600):
            if img[j, i] <= mean_list[i]:
                bakeground.append(img[j, i])
        bgm = mean(bakeground)
        bakeground.clear()
        bgms.append(bgm)

    # 2.4、其他所有像素点除以该列的基础背景值
    img = img.astype(float)
    for i in range(len(mean_list)):
        for j in range(600):
            if img[j, i] != 0:
                img[j, i] = img[j, i] / bgms[i]
            if img[j, i] == 0:
                img[j, i] = 0
    max = img.max()

    return max
"""
z1_1,2,0.9171,155,43,197,149 高斯√ 前√ 后√
z1_1,3,0.9081,107,187,165,261 高斯√    后√
z1_1,4,0.8963,11,203,69,357 高斯√     后√
z1_1,5,0.881,123,491,165,581 高斯√ 前√ 后√
z1_1,6,0.8615,27,75,85,213 高斯√ 前√ 后√

left = 155
top = 43
right = 197
bottom = 149

left = 107
top = 187
right = 165
bottom = 261

left = 11
top = 203
right = 69
bottom = 357

left = 123
top = 491
right = 165
bottom = 581

left = 27
top = 75
right = 85
bottom = 213
"""
"""
left = 155
top = 43
right = 197
bottom = 149

pic_name = "z1_1.png"
pic_time_name = "z1_1_time.png"
pic_space_name = "z1_1_space.png"
pic_data_analyse(left, top, right, bottom, pic_name)
"""
# 左上角：（top,left）(43,155)
# 右下角（bottom,right）(149,197)
"""b=["姓名","性别","分数"]
for i in range(2):
    with open(r'统计.csv', 'a', newline='') as csvFile:
        csv.writer(csvFile).writerow(b)  # 给csv文件中插入一行
"""
"""x=[1,2,3,4,5]
y=[2,3,4,5,6]
plt.scatter(x,y)
plt.savefig("test.png")
plt.show()"""

