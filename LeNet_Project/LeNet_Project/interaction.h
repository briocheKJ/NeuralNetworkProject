#pragma once
              //����opencv����ؿ⺯��
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