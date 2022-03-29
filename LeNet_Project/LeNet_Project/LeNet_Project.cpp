// LeNet_Project.cpp : 此文件包含 "main" 函数。程序执行将在此处开始并结束。
//

#include <cstdio>
#include <stdlib.h>
#include <string>
#include "neuralnetwork.h"
#include "interaction.h" 
using namespace std;
int state = 0;
int index = 0;
int cnt = 0;
char str1[2000]; char str2[2000]; char str3[2000];
int main()
{
    while (1)
    {
        if (state == 0) {
            printf("\n\n1.确定网络结构\n2.exit\n");
            scanf("%d", &index);
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
            scanf("%d", &cnt);
            if (cnt == 1) {
                printf("请依次输入配置文件、训练集文件、测试文件\n");
                
                scanf("%s", str1); scanf("%s", str2); scanf("%s", str3);
                string st1(str1); string st2(str2); string st3(str3);
                //调用init函数
            }
            else if (cnt == 2) {
                printf("请依次输入配置文件、训练集文件\n");

                scanf("%s%s", str1, str2);
                string st1(str1); string st2(str2); 
                //调用init函数
            }
            
            state = 2;
            
        }
        else if (state == 2) {
            printf("\n\n神经网络结构已经建立！\n1.重新组织神经网络\n2.进行训练\n");
            scanf("%d", &index);
            if (index == 1) {
                state = 1;
                //调用清除函数
            }
            else if (index == 2) {
                //调用训练函数
                state = 3;
            }
            
        }
        else if (state == 3) {
            printf("\n\n训练完毕！\n1.重新组织神经网络\n2.进行测试\n");
                scanf("%d", &index);
            if (index == 1) {
                state = 1;
                //调用清除函数
            }
            else if (index == 2) {
                //调用测试函数
                state = 4;
            }
            
        }
        else if (state == 4) {
            printf("\n\n测试完毕\n1.重新组织神经网络\n2.实践应用\n");
            scanf("%d", &index);
            if (index == 1) {
                state = 1;
                //调用清除函数
            }
            else if (index == 2) {
                state = 5;
                //调用test函数
            }
            
        }
        else if (state == 5) {
            printf("\n\n是否继续实践？\n1.yes\n2.no\n");
            scanf("%d", & index);
            if (index == 1) {
                state = 5;
            }
            else if (index == 2) {
                state = 0;
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
