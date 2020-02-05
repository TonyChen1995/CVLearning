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



## Week2

本周学习了opencv的图像滤波与传统的特征提取（SIFT和HOG）算法。

### OpenCV中的图像滤波

opencv的图像滤波，实际上就是图像和卷积核做卷积，它和深度学习中的图像卷积**差异**有以下几点：

（1）卷积核是2D（单通道的），如果将卷积核作用于一个3D图像，那么会分别在图像的每个通道上独立的做相同的卷积，这意味着输入图像的通道数和输出的相等；而深度学习中的卷积核为3D的，它的通道数等于输入tensor的通道数。

（2）进行卷积前后，输入图像和输出图像的宽度、高度也都相同，也就是说，opecv的滤波函数会根据卷积核的大小，自动地调整padding的大小，使得宽高相等；而深度学习中，输入tensor和输出tensor的宽度和高度可以不同。

（3）卷积核中有一个所谓的[锚点](https://docs.opencv.org/2.4/modules/imgproc/doc/filtering.html#filter2d)(anchor point)，该锚点指原图像中进行滤波的像素点在卷积核中的相对位置，通常为卷积核的正中心（这也是卷积核长宽为什么为奇数的主要原因，因为奇数才有正中心）；但是深度学习中该概念并不明显。

（4）深度学习中的paddding一般只有一种（补0），但opencv中的padding有许多种，主要包括：

| Border types       | 示例                      | 说明                     |
| ------------------ | ------------------------- | ------------------------ |
| BORDER_REPLICATE   | aaaaaa\|abcdefgh\|hhhhhhh | 重复边界元素             |
| BORDER_REFLECT     | fedcba\|abcdefgh\|hgfedcb | 边界镜像（含边界元素）   |
| BORDER_REFLECT_101 | gfedcb\|abcdefgh\|gfedcba | 边界镜像（不含边界元素） |
| BORDER_CONSTANT    | iiiiii\|abcdefgh\|iiiiiii | 指定常数                 |

BORDER_REPLICATE是opecv中值滤波函数medianBlur采用的默认方式。BORDER_REFLECT与BORDER_REFLECT_101一般差异很小，只有在卷积核尺度接近图像尺度时差异才较大。BORDER_REFLECT_101应用十分广泛，是滤波函数filter2D，blur，GaussianBlur，bilateralFilter的默认处理方式。BORDER_CONSTANT常用于仿射变换与透视变换中。

卷积的物理意义主要有两点：

（1）对图像的像素值在空间范围上求导，例如prewith和sobel算子（体现一阶导数信息）和laplacian算子（体现二阶导数信息）。

（2）对图像的像素值按照卷积核进行加权平均，例如高斯滤波。