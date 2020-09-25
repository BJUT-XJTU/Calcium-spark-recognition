import csv
import cv2
import keras
from numpy import *
from scipy import exp
import numpy as np
import colorsys
from scipy.optimize import curve_fit
import os
import nets.frcnn as frcnn
from nets.frcnn_training import get_new_img_size
from keras import backend as K
from PIL import Image,ImageFont, ImageDraw
from utils.utils import BBoxUtility
from utils.anchors import get_anchors
from utils.config import Config
import copy
import math
class FRCNN(object):
    _defaults = {
        "model_path": 'logs/loss3.161-rpn2.194-roi0.967.h5',
        "classes_path": 'model_data/voc_classes.txt',
        "confidence": 0.7,
    }

    @classmethod
    def get_defaults(cls, n):
        if n in cls._defaults:
            return cls._defaults[n]
        else:
            return "Unrecognized attribute name '" + n + "'"

    #---------------------------------------------------#
    #   初始化faster RCNN
    #---------------------------------------------------#
    def __init__(self, **kwargs):
        self.__dict__.update(self._defaults)
        self.class_names = self._get_class()
        self.sess = K.get_session()
        self.config = Config()
        self.generate()
        self.bbox_util = BBoxUtility()
    #---------------------------------------------------#
    #   获得所有的分类
    #---------------------------------------------------#
    def _get_class(self):
        classes_path = os.path.expanduser(self.classes_path)
        with open(classes_path) as f:
            class_names = f.readlines()
        class_names = [c.strip() for c in class_names]
        return class_names

    #---------------------------------------------------#
    #   获得所有的分类
    #---------------------------------------------------#
    def generate(self):
        model_path = os.path.expanduser(self.model_path)
        assert model_path.endswith('.h5'), 'Keras model or weights must be a .h5 file.'
        
        # 计算总的种类
        self.num_classes = len(self.class_names)+1

        # 载入模型，如果原来的模型里已经包括了模型结构则直接载入。
        # 否则先构建模型再载入
        self.model_rpn,self.model_classifier = frcnn.get_predict_model(self.config,self.num_classes)
        self.model_rpn.load_weights(self.model_path,by_name=True)
        self.model_classifier.load_weights(self.model_path,by_name=True,skip_mismatch=True)
                
        print('{} model, anchors, and classes loaded.'.format(model_path))

        # 画框设置不同的颜色
        hsv_tuples = [(x / len(self.class_names), 1., 1.)
                      for x in range(len(self.class_names))]
        self.colors = list(map(lambda x: colorsys.hsv_to_rgb(*x), hsv_tuples))
        self.colors = list(
            map(lambda x: (int(x[0] * 255), int(x[1] * 255), int(x[2] * 255)),
                self.colors))
    
    def get_img_output_length(self, width, height):
        def get_output_length(input_length):
            # input_length += 6
            filter_sizes = [7, 3, 1, 1]
            padding = [3,1,0,0]
            stride = 2
            for i in range(4):
                # input_length = (input_length - filter_size + stride) // stride
                input_length = (input_length+2*padding[i]-filter_sizes[i]) // stride + 1
            return input_length
        return get_output_length(width), get_output_length(height) 
    
    #---------------------------------------------------#
    #   检测图片
    #---------------------------------------------------#
    #旧版本，直接显示图片，并标出来

    def detect_image(self,image_id,image):
        spark_num = 1
        #self.confidence = 0.05
        #f = open("./mAP-master/input/detection-results/"+image_id+".txt","a")
        image_shape = np.array(np.shape(image)[0:2])

        old_width = image_shape[1]
        old_height = image_shape[0]
        old_image = copy.deepcopy(image)
        width, height = get_new_img_size(old_width, old_height)

        image = image.resize([width, height])
        photo = np.array(image, dtype=np.float64)

        # 图片预处理，归一化
        # 新的预处理方式！
        #photo = preprocess_input(np.expand_dims(photo, 0))
        photo = np.expand_dims(photo, 0)

        preds = self.model_rpn.predict(photo)

        # 将预测结果进行解码
        anchors = get_anchors(self.get_img_output_length(width, height), width, height)


        rpn_results = self.bbox_util.detection_out(preds, anchors, 1, confidence_threshold=0)
        R = rpn_results[0][:, 2:]

        R[:, 0] = np.array(np.round(R[:, 0] * width / self.config.rpn_stride), dtype=np.int32)
        R[:, 1] = np.array(np.round(R[:, 1] * height / self.config.rpn_stride), dtype=np.int32)
        R[:, 2] = np.array(np.round(R[:, 2] * width / self.config.rpn_stride), dtype=np.int32)
        R[:, 3] = np.array(np.round(R[:, 3] * height / self.config.rpn_stride), dtype=np.int32)

        R[:, 2] -= R[:, 0]
        R[:, 3] -= R[:, 1]
        base_layer = preds[2]

        delete_line = []
        for i, r in enumerate(R):
            if r[2] < 1 or r[3] < 1:
                delete_line.append(i)
        R = np.delete(R, delete_line, axis=0)

        bboxes = []
        probs = []
        labels = []
        for jk in range(R.shape[0] // self.config.num_rois + 1):
            ROIs = np.expand_dims(R[self.config.num_rois * jk:self.config.num_rois * (jk + 1), :], axis=0)

            if ROIs.shape[1] == 0:
                break

            if jk == R.shape[0] // self.config.num_rois:
                # pad R
                curr_shape = ROIs.shape
                target_shape = (curr_shape[0], self.config.num_rois, curr_shape[2])
                ROIs_padded = np.zeros(target_shape).astype(ROIs.dtype)
                ROIs_padded[:, :curr_shape[1], :] = ROIs
                ROIs_padded[0, curr_shape[1]:, :] = ROIs[0, 0, :]
                ROIs = ROIs_padded

            [P_cls, P_regr] = self.model_classifier.predict([base_layer, ROIs])

            for ii in range(P_cls.shape[1]):
                if np.max(P_cls[0, ii, :]) < self.confidence or np.argmax(P_cls[0, ii, :]) == (P_cls.shape[2] - 1):
                    continue

                label = np.argmax(P_cls[0, ii, :])

                (x, y, w, h) = ROIs[0, ii, :]

                cls_num = np.argmax(P_cls[0, ii, :])

                (tx, ty, tw, th) = P_regr[0, ii, 4 * cls_num:4 * (cls_num + 1)]
                tx /= self.config.classifier_regr_std[0]
                ty /= self.config.classifier_regr_std[1]
                tw /= self.config.classifier_regr_std[2]
                th /= self.config.classifier_regr_std[3]

                cx = x + w / 2.
                cy = y + h / 2.
                cx1 = tx * w + cx
                cy1 = ty * h + cy
                w1 = math.exp(tw) * w
                h1 = math.exp(th) * h

                x1 = cx1 - w1 / 2.
                y1 = cy1 - h1 / 2.

                x2 = cx1 + w1 / 2
                y2 = cy1 + h1 / 2

                x1 = int(round(x1))
                y1 = int(round(y1))
                x2 = int(round(x2))
                y2 = int(round(y2))

                bboxes.append([x1, y1, x2, y2])
                probs.append(np.max(P_cls[0, ii, :]))
                labels.append(label)

        if len(bboxes) == 0:
            return old_image

        # 筛选出其中得分高于confidence的框
        labels = np.array(labels)
        probs = np.array(probs)
        boxes = np.array(bboxes, dtype=np.float32)
        boxes[:, 0] = boxes[:, 0] * self.config.rpn_stride / width
        boxes[:, 1] = boxes[:, 1] * self.config.rpn_stride / height
        boxes[:, 2] = boxes[:, 2] * self.config.rpn_stride / width
        boxes[:, 3] = boxes[:, 3] * self.config.rpn_stride / height
        results = np.array(
            self.bbox_util.nms_for_out(np.array(labels), np.array(probs), np.array(boxes), self.num_classes - 1, 0.4))

        top_label_indices = results[:, 0]
        top_conf = results[:, 1]
        boxes = results[:, 2:]
        boxes[:, 0] = boxes[:, 0] * old_width
        boxes[:, 1] = boxes[:, 1] * old_height
        boxes[:, 2] = boxes[:, 2] * old_width
        boxes[:, 3] = boxes[:, 3] * old_height

        font = ImageFont.truetype(font='model_data/simhei.ttf',
                                  size=np.floor(3e-2 * np.shape(image)[1] + 0.5).astype('int32'))

        thickness = (np.shape(old_image)[0] + np.shape(old_image)[1]) // width
        image = old_image
        #self.write_to_txt_title(image_id)
        #self.write_to_csv_title(image_id)
        for i, c in enumerate(top_label_indices):
            predicted_class = self.class_names[int(c)]
            score = top_conf[i]

            left, top, right, bottom = boxes[i]
            top = top - 5
            left = left - 5
            bottom = bottom + 5
            right = right + 5

            top = max(0, np.floor(top + 0.5).astype('int32'))
            left = max(0, np.floor(left + 0.5).astype('int32'))
            bottom = min(np.shape(image)[0], np.floor(bottom + 0.5).astype('int32'))
            right = min(np.shape(image)[1], np.floor(right + 0.5).astype('int32'))

            # 画框框
            label = '{} {} {:.2f}'.format(spark_num,predicted_class, score)
            draw = ImageDraw.Draw(image)
            label_size = draw.textsize(label, font)
            label = label.encode('utf-8')
            print(label)


            if top - label_size[1] >= 0:
                text_origin = np.array([left, top - label_size[1]])
            else:
                text_origin = np.array([left, top + 1])

            for i in range(thickness):
                draw.rectangle(
                    [left + i, top + i, right - i, bottom - i],
                    outline=self.colors[int(c)])
            draw.rectangle(
                [tuple(text_origin), tuple(text_origin + label_size)],
                fill=self.colors[int(c)])
            draw.text(text_origin, str(label, 'UTF-8'), fill=(0, 0, 0), font=font)
            del draw

            score = round(score, 4)
            score = str(score)

            # 分析钙火花
            self.spark_analyse(image_id , image , spark_num , score, left, top, right,bottom)
            spark_num = spark_num + 1
            #strr = str(spark_num) +" "+str(score)+" "+str(left)+" "+str(top)+" "+str(right)+" "+str(bottom)
            #print(strr)
            #f.write(strr)
            #f.write("%s %s %s %s %s %s\n" % (predicted_class, score, str(int(left)), str(int(top)), str(int(right)),str(int(bottom))))

        #f.close()
        return image

    def spark_analyse(self, image_id, image, spark_num, score, left, top, right,bottom):
        # 获取基础数据
        data_txt1 = str(image_id)+ ","+str(spark_num) + "," + str(score) + "," + str(left) \
               + "," + str(top) + "," + str(right) + "," + str(bottom)
        # 分析内容
        # pic_name ; num ; score ; spark_frame:left、top、right,bottom ; dF/F0 ; FWHM ; rise time ; t50 ; FDHM ; tau ; x_pos ; t_pos
        pic_name = "./VOCdevkit/VOC2007/JPEGImages/"+image_id+".png"
        num = spark_num
        # 分析函数
        amplitude, FDHM, T50, tRise, FWHM, x_pos, y_pos = self.pic_data_analyse(left, top, right, bottom, pic_name)
        # dF / F0;   FWHM; ris time;   t50;   FDHM;    x_pos;  t_pos
        data_txt2 = str(amplitude) + "," + str(FWHM) + "," + str(tRise) + "," + str(T50) + "," + str(FDHM) + "," + str(x_pos) + "," + str(y_pos)
        data_txt = data_txt1 + data_txt2
        # b = ["姓名", "性别", "分数"]
        data_csv = [image_id, num, score, amplitude, tRise, T50, FDHM, FWHM, x_pos, y_pos]
        # 分析结束 写入txt和csv
        #self.write_to_txt(data_txt, image_id)
        self.write_to_csv(data_csv, image_id)

        # 写入 csv

    def write_to_txt (self, data, image_id ):
        image_name ="./spark_analyse/"+image_id+".txt"
        result2txt = str(data)  # data是前面运行出的数据，先将其转为字符串才能写入
        with open(image_name, 'a') as file_handle:  # .txt可以不自己新建,代码会自动新建
            file_handle.write(result2txt)
            file_handle.write('\n')  # 有时放在循环里面需要自动转行，不然会覆盖上一条数据

    def write_to_txt_title (self, image_id):
        # print("写入标题")
        data = "pic_name ; num ; score ; spark_frame:left、top、right、bottom ; dF/F0 ; FWHM ; rise time ; t50 ; FDHM ; x_pos ; t_pos "
        image_txt_name ="./spark_analyse/"+image_id+".txt"
        result2txt = str(data)  # data是前面运行出的数据，先将其转为字符串才能写入
        with open(image_txt_name, 'a') as file_handle:  # .txt可以不自己新建,代码会自动新建
            file_handle.write(result2txt)
            file_handle.write('\n')  # 有时放在循环里面需要自动转行，不然会覆盖上一条数据

    def write_to_csv (self, data, image_id ):
        #image_csv_name = "./spark_analyse/" + image_id + ".csv"
        image_csv_name = "./spark_analyse/spark_analyse.csv"
        with open(image_csv_name, 'a', newline='') as csvFile:
            csv.writer(csvFile).writerow(data)  # 给csv文件中插入一行

    def write_to_csv_title(self,image_id):
        image_csv_name = "./spark_analyse/" + image_id + ".csv"
        data = ["pic_name", "num", u"score", "dF/F0", u"rise time", "t50", "FDHM", "FWHM", "x_pos", "t_pos"]
        with open(image_csv_name, 'a', newline='') as csvFile:
            csv.writer(csvFile).writerow(data)  # 给csv文件中插入一行

    def pic_data_analyse(self, left, top, right, bottom, pic_name):
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

        amplitude, FDHM, T50, tRise = self.space_alldata_draw(spark_space_means, spark_space_mean_x, pic_name, top, bottom)
        #plt.savefig(self.pic_space_name)
        #plt.show()

        FWHM = self.time_alldata_draw(spark_time_mean_x, spark_time_means)
        # plt.savefig(pic_time_name)
        # plt.show()

        # FWHM, rise time, t50, FDHM, x_pos, t_pos
        x_pos = max_index_x + top
        y_pos = max_index_y + left

        #print("幅度:", amplitude, "时间半宽度FDHM:", FDHM, "空间半宽度FWHM:", FWHM)
        #print("下降一半的时间T50:", T50, "上升相时间rise Rise:", tRise)
        #print("x_pos:", x_pos, "y_pos:", y_pos)

        return amplitude, FDHM, T50, tRise, FWHM, x_pos, y_pos

    # 高斯函数 时间
    def gaussian(self, x, *param):
        return param[0] * np.exp(-np.power(x - param[1], 2.) / (2 * np.power(param[2], 2.)))

    # 前半段函数 空间
    def fitexp_sr1(self, x, *param):
        f = param[1] * (1 - exp(-(1 / param[2]) * (x - param[3]))) + param[0]
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
    def fitexp_sr2(self, x, *param):
        bx = exp(param[1] * x)
        return param[0] * bx + param[2]

    # 尝试全过程使用一种函数 空间
    def fitexp_all(self, x, *param):
        f = param[1] * (1 - exp(-(1 / param[2]) * (x - param[3]))) + param[0]
        g = param[4] * exp(-(1 / param[5]) * (x - param[6])) + param[7]
        fp = 0
        for i in range(len(x)):
            if x[i] < param[3]:
                fp = fp + 1
        if fp > 0:
            for i in range(fp - 1):
                f[i] = param[0]
        gp = 0
        for i in range(len(f)):
            if f[i] <= g[i]:
                gp = gp + 1
        if gp >= 0:
            for i in range(len(x)):
                if i == gp:
                    f[i] = g[i]
        return f

    # 空间 全部点画图
    def space_alldata_draw(self, spark_space_means, spark_space_mean_x, pic_name, top, bottom):
        # x拟合
        spark_space_means_np = np.array(spark_space_means)
        max = spark_space_means_np.max()
        for i in range(bottom - top):
            if (spark_space_means_np[i] == max):
                max_index = i
        # max_index 作为分水岭
        y = spark_space_means_np
        X_y1 = np.array(spark_space_means_np[0:max_index])
        X_y2 = np.array(spark_space_means_np[max_index:])
        x = np.array(spark_space_mean_x)
        X_x1 = x[0:max_index]
        X_x2 = x[max_index:]

        #plt.scatter(x, y, s=20, c="red", marker='o')
        #plt.scatter(max_index, max, s=50, c="green", marker='o')

        N = self.norm_N(pic_name)

        # X 前半段画图 返回最后一个点
        x_last, y_last, y_qian = self.x_fitexp_sr1(X_x1, X_y1)

        # X 后半段画图 返回第一个点
        x_first, y_first, y_hou = self.x_fitexp_sr2(X_x2, X_y2, N)

        # X 分析
        y_quanbunihe = np.concatenate((y_qian, y_hou), axis=0)
        amplitude, FDHM, T50, tRise = self.pic_quxian_analyse_space(x, y_quanbunihe)

        # 两点链接画图
        self.first_last(x_last, y_last, x_first, y_first)

        return amplitude, FDHM, T50, tRise

    # 时间 全部点画图
    def time_alldata_draw(self, spark_time_mean_x, spark_time_means):
        x = np.array(spark_time_means)
        y = np.array(spark_time_mean_x)
        #plt.scatter(y, x, s=20, c="red", marker='o')
        # Y 画图 高斯拟合
        FWHM = self.y_nihe(y, x)
        return FWHM

    # x 前半段画图 拟合图
    def x_fitexp_sr1(self, x, y):
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
        M2 = (y[-2] - M1) * 1.5
        M3 = len(y) - n
        popt, pcov = curve_fit(self.fitexp_sr1, x, y, p0=[[M1, M2, M3, M4]])  # [193, 16, 29, 60]
        # print("前半段拟合参数",popt)
        y_nihe = self.fitexp_sr1(x, *popt)
        #plt.plot(x, y_nihe)
        x_last = x[-1]
        y_last = y_nihe[-1]

        return x_last, y_last, y_nihe

    # x 后半段画图 拟合图
    def x_fitexp_sr2(self, x, y, N):
        # 初始值的设定：A[0]= 后半段第一个点的值-N, A[1]=-1.5/n1。其中n1为后半段的点的个数
        # print(max_index, max)
        # print(y[1])
        # print("N:", N , "后半段第一个值:", y[1])
        M1 = y[1] - N
        M2 = -1.5 / (len(x) - 1)
        M3 = N
        popt, pcov = curve_fit(self.fitexp_sr2, x, y, p0=[M1, M2, M3])
        # print("后半段拟合参数",popt)
        y_nihe = self.fitexp_sr2(x, *popt)
        #plt.plot(x, y_nihe)
        x_first = x[0]
        y_first = y_nihe[0]

        return x_first, y_first, y_nihe

    # y拟合 高斯 画图
    def y_nihe(self, x, y):
        popt, pcov = curve_fit(self.gaussian, x, y, p0=[1, 1, 1])
        # print("高斯参数",popt)
        y_nihe = self.gaussian(x, *popt)
        #  plt.plot(x, y_nihe)
        # 分析曲线
        FWHM = self.pic_quxian_analyse_time(x, y_nihe)
        return FWHM

    # 首尾链接画图
    def first_last(self, x_last, y_last, x_first, y_first):
        x_lianjie = []
        y_lianjie = []
        x_lianjie.append(x_last)
        x_lianjie.append(x_first)
        y_lianjie.append(y_last)
        y_lianjie.append(y_first)
        #  plt.plot(x_lianjie, y_lianjie)

    # 空间曲线分析
    def pic_quxian_analyse_space(self, x, y_nihe):
        # dF / F0, FWHM, rise
        # time, t50, FDHM, x_pos, t_pos
        Fpeak = y_nihe.max()
        Fpeak_x = 0
        for i in range(len(x)):
            if y_nihe[i] == Fpeak:
                Fpeak_x = i
        F0 = (y_nihe[0] + y_nihe[-1]) / 2
        mudium = (Fpeak + F0) / 2
        amplitude = (Fpeak - F0) / F0  # df/d0 幅度
        amplitude = round(amplitude, 2)
        # print("幅度：",amplitude)
        x_xiao = 0
        x_da = 0
        for i in range(Fpeak_x):
            if y_nihe[i] <= mudium:
                x_xiao = i
                x_da = i + 1
        x_index1 = (mudium - y_nihe[x_xiao]) / (y_nihe[x_da] - y_nihe[x_xiao])
        x_index1 = x_xiao + x_index1
        for i in range(len(x)-1):
            if i > Fpeak_x:
                if y_nihe[i] >= mudium:
                    x_da = i
                    x_xiao = i + 1
        x_index2 = (mudium - y_nihe[x_xiao]) / (y_nihe[x_da] - y_nihe[x_xiao])
        x_index2 = x_da + x_index2
        FDHM = x_index2 - x_index1
        FDHM = round(FDHM, 2)
        # print("FDHM", FDHM)  # 时间半函数
        T50 = x_index2 - Fpeak_x  # 下降一半时间
        T50 = round(T50, 2)
        # print("T50", T50)
        tRise1 = 0
        for i in range(len(y_nihe) - 1):
            if y_nihe[i] == y_nihe[i + 1]:
                tRise1 = i
        tRise = Fpeak_x - 0
        tRise = round(tRise, 2)
        # print("tRise", tRise)
        return amplitude, FDHM, T50, tRise

    # 时间曲线分析
    def pic_quxian_analyse_time(self, x, y_nihe):
        # dF / F0, FWHM, rise
        # time, t50, FDHM, x_pos, t_pos
        # print("分析：")
        Fpeak = y_nihe.max()
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
        # T50 = x_index2 - Fpeak_x  # 下降一半时间
        # print("T50", T50)
        # tRise1=0
        # tRise = Fpeak_x - tRise1
        # print("tRise", tRise)
        return FWHM

    # 计算像素点
    def norm_N(self, name):
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

    def close_session(self):
        self.sess.close()
