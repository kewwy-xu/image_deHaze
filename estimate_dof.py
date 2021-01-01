import cv2
import sys
import numpy as np
import matplotlib.pyplot as plt
from guidedFilter import *



def getDOFChange(dark_Channel):
    """估计景深突变区域:在求暗通道图像时，使用的局部窗口内取最小值操作类似于形态学中腐蚀操作，
    假设对图像腐蚀后的结果再进行一次膨胀操作，并将两次运算结果做差则刚好对应了景深突变区域"""
    dark_Channel_erode=zmMaxFilterGray(dark_Channel)
    #做差得到景深突变区域
    dof_change=dark_Channel_erode-dark_Channel
    return dof_change



