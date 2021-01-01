import cv2
import sys
import numpy as np
import matplotlib.pyplot as plt


def zmMinFilterGray(src, r=7,iteration=1):
    """最小值滤波，r是滤波器半径"""
    #1.cv2.erode(src, kernel, iteration)
    #参数说明：src表示的是输入图片，kernel表示的是方框的大小，iteration表示迭代的次数
    #迭代次数越多，模糊程度（腐蚀程度）越大
    return cv2.erode(src, np.ones((2 * r + 1, 2 * r + 1)),iteration)

def zmMaxFilterGray(src, r=7,iteration=1):
    """最大值滤波，r是滤波器半径"""
    return cv2.dilate(src, np.ones((2 * r + 1, 2 * r + 1)),iteration)


def guidedfilter(I, p, r, eps):
    """导向滤波： https://blog.csdn.net/wsp_1138886114/article/details/84228939"""
    height, width = I.shape
    #boxFilter:  https://www.cnblogs.com/my-love-is-python/p/10391923.html
    #boxFilter第二个参数ddepth：int类型，输出图像的数据类型，往往使用-1，表示输出图像数据类型与输入图像一致；
    m_I = cv2.boxFilter(I, -1, (r, r))  #均值滤波
    m_p = cv2.boxFilter(p, -1, (r, r))
    m_Ip = cv2.boxFilter(I * p, -1, (r, r))
    cov_Ip = m_Ip - m_I * m_p

    m_II = cv2.boxFilter(I * I, -1, (r, r))
    var_I = m_II - m_I * m_I

    a = cov_Ip / (var_I + eps)
    b = m_p - a * m_I

    m_a = cv2.boxFilter(a, -1, (r, r))
    m_b = cv2.boxFilter(b, -1, (r, r))
    return m_a * I + m_b


def Defog(m, r, eps, w, maxV1):                 # 输入rgb图像，值范围[0,1]
    '''计算大气遮罩图像V1和光照值A, V1 = 1-t/A'''
    minFilterGray = np.min(m, 2)
    Dark_Channel = zmMinFilterGray(minFilterGray,5)  # 得到暗通道图像
    #cv2.imshow('20190708_Dark',Dark_Channel)    # 查看暗通道
    #cv2.imwrite('./images/output/'+imageName_prefix+'_Dark_Channel.'+imageName_suffix, Dark_Channel*255)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    V1 = guidedfilter(minFilterGray, Dark_Channel, r, eps)  # 使用引导滤波优化
    A=None
    bins = 2000
    ht = np.histogram(V1, bins)                  # 计算大气光照A
    d = np.cumsum(ht[0]) / float(V1.size)
    for lmax in range(bins - 1, 0, -1):
        if d[lmax] <= 0.999:
            A = np.mean(m, 2)[V1 >= ht[1][lmax]].max()
            break
    V1 = np.minimum(V1 * w, maxV1)  # 对值范围进行限制
    return minFilterGray,V1, A


def deHaze(m, r=81, eps=0.001, w=0.95, maxV1=0.80, bGamma=True,a=0,k=1):
    Y = np.zeros(m.shape)
    #Mask_img=A(1-t(x))，t(x)为透光率
    minFilterGray,Mask_img, A = Defog(m, r, eps, w, maxV1)             # 得到遮罩图像和大气光照
    t2=1-Mask_img/A
    for k in range(3):
        Y[:,:,k] = (m[:,:,k] - Mask_img)/t2  # 颜色校正

    Y = np.clip(Y, 0, 1)
    if bGamma:
        Y = Y ** (np.log(0.5) / np.log(Y.mean()))       # gamma校正,默认进行该操作
    return Y


def deHaze_improve(m, r=81, eps=0.001, w=0.95, maxV1=0.80, bGamma=True,a=0.5,k=5):
    Y = np.zeros(m.shape)
    #Mask_img=A(1-t(x))，t(x)为透光率
    minFilterGray,Mask_img, A = Defog(m, r, eps, w, maxV1)             # 得到遮罩图像和大气光照

    #--------改进1:缓解景深突变区域得光晕现象-----------
    t2=1-Mask_img/A
    t1=1-minFilterGray/A
    # 景深突变区域修正透射率: 对最小值像素暗通道和局部窗口暗通道
    # 分别求取的透射率t1,t2做加权运算.a为权重系数
    t=t1*a+t2*(1-a)
    # --------改进1:缓解景深突变区域得光晕现象-----------

    # --------改进2:缓解天空区域失真得问题-----------
    for k in range(3):
        tmp=m[:,:,k] - A
        tmp1=np.maximum(m[:,:,k] - A,A - m[:,:,k])
        tmp2 = np.maximum(k / tmp1, 1)
        Y[:, :, k]=tmp/np.minimum(tmp2*t,1)
    # --------改进2:缓解天空区域失真得问题-----------


    Y = np.clip(Y, 0, 1)
    if bGamma:
        Y = Y ** (np.log(0.5) / np.log(Y.mean()))       # gamma校正,默认进行该操作
    return Y

