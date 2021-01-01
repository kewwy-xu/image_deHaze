import cv2
import sys
import numpy as np
import matplotlib.pyplot as plt
from guidedFilter import *
from changeSaturation import *

if __name__ == '__main__':
    imageName='4.png'

    imageName_prefix = imageName.split('.', 1)[0]
    imageName_suffix = imageName.split('.', 1)[1]
    img_original=cv2.imread('./images/input/'+imageName)
    m = deHaze( img_original/ 255.0) * 255
    save_path1='./images/output/'+imageName
    cv2.imwrite(save_path1, m)

    #---------改进1：降低饱和度---------
    m=cv2.imread(save_path1)
    m = cv2.cvtColor(m, cv2.COLOR_BGR2RGB)
    img_lowSaturate = changeSaturation(m,-0.3)      #降低饱和度

    # ---------改进2：改进算法模型，解决光晕和天空色彩失真问题---------
    m_improve = deHaze(img_original / 255.0) * 255

    #原图
    plt.figure("img_original")
    plt.imshow(cv2.cvtColor(img_original, cv2.COLOR_BGR2RGB))
    plt.axis('off')

    # 去雾图
    plt.figure("img_deHaze")
    plt.imshow(m / 255.0)
    plt.axis('off')

    # 去雾图->降低饱和度
    plt.figure("img_lowSaturate")
    plt.imshow(img_lowSaturate)
    plt.axis('off')

    # 算法模型改进后的去雾图
    plt.figure("img_deHaze_improve")
    plt.imshow(m_improve / 255.0)
    plt.axis('off')

    plt.show()

