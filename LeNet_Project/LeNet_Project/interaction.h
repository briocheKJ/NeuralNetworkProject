#pragma once
              //读入opencv的相关库函数
class interaction {
public:
	
	static interaction& getInstance() {
		if (instance == nullptr) {
			instance = new interaction;
		}
		return *instance;
	}
	void Management();
	static void ReleaseInstance(){
		delete instance;
		instance = nullptr;
	}		
private:
	static interaction* instance;
	void	InputPicture();
	
	
};