# CVLearning

本仓库记录了学习计算机视觉的作业与相关思考。

## week1

**1 basic_operations.ipynb**

练习了opencv的基本操作，包括：

（1）图像的读取与显示

（2）图像的色彩空间变换与通道分割

（3）图像裁剪、旋转、仿射变换与投影变换等

（4）图像亮度直方图均衡化

（5）图像的腐蚀和膨胀

**2 image_augmentation.py**

利用opencv实现了一些图像增强操作，包括：

（1）图像剪切 crop

（2）RGB通道的颜色偏移 color_shift

（3）图像旋转 rotation

（4）仿射变换 perspective_transform

**3 change_background.py**

改变图像背景色的demo。关键在于色域转换（RGB -> HSV）。

原图：

![image-20200201213300378](README.assets/image-20200201213300378.png)

改为绿色背景：

![image-20200201213409112](README.assets/image-20200201213409112.png)

改为红色背景：

![image-20200201213446676](README.assets/image-20200201213446676.png)