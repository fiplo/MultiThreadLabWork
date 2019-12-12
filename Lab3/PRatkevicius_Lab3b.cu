#include <iostream>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <fstream>

using namespace std;
using namespace thrust;

const int ARRAY_SIZE = 25;
const int CHAR_ARRAY_SIZE = 35;
const int STRING_SIZE = 254;

struct User {
    char Name[CHAR_ARRAY_SIZE];
    int Age;
    double Balance;
};


struct Addition_func{
__device__ User operator()(User accum, User src){
	int accLen, srcLen;
	for (accLen  = 0; accum.Name[accLen] != '\0'; accLen++);
	for (srcLen = 0; src.Name[srcLen] != '\0'; srcLen++)
	{
		accum.Name[accLen] = src.Name[srcLen];
		accLen++;
	}
	accum.Name[accLen] = '\0';
	accum.Age += src.Age;
	accum.Balance += src.Balance;
	return accum;
	}
};
void read(host_vector<User> &hostData, string fileName) {
	ifstream  ifs(fileName);
	string Name;
        int Age;
        double Balance;
	User item;

        if (ifs.fail()) {
                cout << "Error opening file " + fileName << endl;
                exit(1);
        }
        for(size_t i = 0; i <ARRAY_SIZE; i++){
           	ifs >> Name >> Age >> Balance;
                for (int j = 0; j < CHAR_ARRAY_SIZE;j++) {
                        if (Name[j] == 0) {
                                item.Name[j] = 0;
                                break;
                        }
                  	item.Name[j] = (char)Name[j];
                }
                item.Age = Age;
                item.Balance = Balance;
                hostData.push_back(item);
        }
}
int main() {
  string fileName = "../data/Paulius_Ratkevicius_L1_dat_1_ResPlain.txt";

	host_vector<User> hostData;
  read(hostData, fileName);
	device_vector<User> deviceData = hostData;
	User dest = {"", 0, 0.};
	User answer = reduce(deviceData.begin(), deviceData.end(), dest, Addition_func());
  cout << answer.Name << " "  << answer.Age << "\n" << answer.Balance << "\n";
	return 0;
}

