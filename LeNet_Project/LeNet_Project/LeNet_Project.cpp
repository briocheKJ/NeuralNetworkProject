// LeNet_Project.cpp : 此文件包含 "main" 函数。程序执行将在此处开始并结束。
//

#include <cstdio>
#include <iostream>
using namespace std;

#include <stdlib.h>
#include <string>

#include "neuralnetwork.h"
#include "interaction.h" 

int state = 0;
int index = 0;
int cnt = 0;
char str1[2000]; char str2[2000]; char str3[2000]; char str4[2000]; char str5[2000];
int main()
{
    srand(time(NULL));

    while (1)
    {
        if (state == 0) {
            printf("\n\n1.确定网络结构\n2.exit\n");
            cin>>index;
            if (index == 1) {
                
                state = 1;
                system("cls");
            }
            else if (index == 2) {
                
                exit(0);
            }
            
        }
        else if (state == 1) {
            printf("\n\n是否需要进行测试？\n1.yes\n2.no\n");
            cin >> cnt;
            if (cnt == 1) {
                printf("请依次输入配置文件、训练集图片文件、训练集数据文件、测试集图片文件、测试集数据文件\n");
                
                cin >> str1; cin >> str2; cin >> str3; cin >> str4; cin >> str5;
                 string st1(str1);  string st2(str2);  string st3(str3);  string st4(str4);  string st5(str5);
                NeuralNetwork::getInstance()->initialize(&st1, &st2, &st3, &st4, &st5);
                 state = 2;
            }
            else if (cnt == 2) {
                printf("请依次输入配置文件、训练集图片文件、训练集数据文件\n");

                cin >> str1; cin >> str2; cin >> str3; 
                 string st1(str1);  string st2(str2);  string st3(str3); 
                NeuralNetwork::getInstance()->initialize(&st1, &st2, &st3);
                 state = 6;
            }
            
            
            
        }
        else if (state == 6) {
            printf("\n\n神经网络结构已经建立！\n1.重新组织神经网络\n2.进行训练\n");
            cin >> index;
            if (index == 1) {
                state = 1;
                NeuralNetwork::releaseInstance();
            }
            else if (index == 2) {
                int tempnum;
                printf("请输入Batch大小（必须整除训练集大小）：");
                cin >> tempnum;
                NeuralNetwork::getInstance()->trainBatch(tempnum);
                state = 4;
            }
           
        }
        else if (state == 2) {
            printf("\n\n神经网络结构已经建立！\n1.重新组织神经网络\n2.进行训练\n");
            cin >> index;
            if (index == 1) {
                state = 1;
                NeuralNetwork::releaseInstance();
            }
            else if (index == 2) {
                int tempnum;
                printf("请输入Batch大小（必须整除训练集大小）：");
                cin >> tempnum;
                NeuralNetwork::getInstance()->trainBatch(tempnum);
                state = 3;
            }
           
        }
        else if (state == 3) {
            printf("\n\n训练完毕！\n1.重新组织神经网络\n2.进行测试\n3.重新训练\n");
            cin >> index;
            if (index == 1) {
                state = 1;
                NeuralNetwork::releaseInstance();
            }
            else if (index == 2) {
                NeuralNetwork::getInstance()->testBatch();
                state = 4;
            }
            else if (index == 3) {
                state = 2;
            }
            
        }
        else if (state == 4) {
            printf("\n\n请选择\n1.重新组织神经网络\n2.实践应用\n3.重新训练\n");
            cin >> index;
            if (index == 1) {
                state = 1;
                NeuralNetwork::releaseInstance();
            }
            else if (index == 2) {
                state = 5;
                interaction::getInstance().Management();
            }
            else if (index == 3) {
                state = 2;
            }
        }
        else if (state == 5) {
            printf("\n\n是否继续实践？\n1.yes\n2.no\n");
            cin >> index;
            if (index == 1) {
                interaction::getInstance().Management();
                state = 5;
            }
            else if (index == 2) {
                state = 0;
                NeuralNetwork::releaseInstance();
            }
        }

    }
}

// 运行程序: Ctrl + F5 或调试 >“开始执行(不调试)”菜单
// 调试程序: F5 或调试 >“开始调试”菜单

// 入门使用技巧: 
//   1. 使用解决方案资源管理器窗口添加/管理文件
//   2. 使用团队资源管理器窗口连接到源代码管理
//   3. 使用输出窗口查看生成输出和其他消息
//   4. 使用错误列表窗口查看错误
//   5. 转到“项目”>“添加新项”以创建新的代码文件，或转到“项目”>“添加现有项”以将现有代码文件添加到项目
//   6. 将来，若要再次打开此项目，请转到“文件”>“打开”>“项目”并选择 .sln 文件
