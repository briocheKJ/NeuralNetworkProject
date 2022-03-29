#include <cstdio>
using namespace std;
//读入opencv
#include "interaction.h"
#include "neuralnetwork.h"
interaction* interaction::instance = 0;
void interaction::InputPicture() {

}
void interaction::Management() {
	InputPicture();
	//调用神经网络的test()
}
