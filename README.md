# NeuralNetworkProject
LeNet-5 Project

# 已经把所有文件都添加入项目，以后只需上传自己修改过的文件
## 配置格式
输入训练集大小（组数）x，输入测试集大小y（y是0表示没有测试集）

先输入一个正整数n，表示神经网络的层数（不包括输入层，包括输出层）

输入两个整数h，w表示image的大小

接下来输入n组数，每组包括

type，表示神经网络该层的种类，1表示卷积层，2表示池化层，3表示全连接层，4表示输出层；最后一层可以且必须是唯一的输出层。

接下来一行包含三个整数，a表示该层的输入特征图大小（如果不是第一层需要与上一层匹配），b表示输入特征图个数（输入层必须是1；如果不是第一层需要与上一层的输出层特征图匹配），c表示输出特征图个数（如果是池化层需要和缩放比例匹配，输出层通常是10）

如果type=1，接下来输入b行长度为c的0/1字符串个数表示相连的邻接矩阵；另外还需要输入卷积核移动的步长d，和扩展边界的宽度e

如果type=2，则需输入缩放的比例f，其中f必须整除a
