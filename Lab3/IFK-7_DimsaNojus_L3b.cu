#include <iostream>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <fstream>

using namespace std;
using namespace thrust;

const int ARRAY_SIZE = 25;
const int CHAR_ARRAY_SIZE = 10;
const int STRING_SIZE = 254;

struct Item {
        int intas;
        double doublas;
        char stringas[STRING_SIZE];
};
struct Addition_func{
__device__ Item operator()(Item accum, Item src){
	int accLen, srcLen;
	for (accLen  = 0; accum.stringas[accLen] != '\0'; accLen++);
	for (srcLen = 0; src.stringas[srcLen] != '\0'; srcLen++)
	{
		accum.stringas[accLen] = src.stringas[srcLen];
		accLen++;
	}
	accum.stringas[accLen] = '\0';
	accum.intas += src.intas;
	accum.doublas += src.doublas;
	return accum;
	}
};
void read(host_vector<Item> &hostData) {
	ifstream  ifs("IFK-7_NojusD_L3_dat.txt");
	//ifstream  ifs("IFK_NojusD_L3_dat.txt");
	string stringas;
        int intas;
        double doublas;
	Item item;

        if (ifs.fail()) {
                cout << "Error opening file (IFK-7_NojusD_L3_dat.txt)" << endl;
                exit(1);
        }
        for(size_t i = 0; i <ARRAY_SIZE; i++){
           	ifs >> stringas >> intas >> doublas;
                for (int j = 0; j < CHAR_ARRAY_SIZE;j++) {
                        if (stringas[j] == 0) {
        			item.stringas[j] = 0;
                                break;
                        }
                  	item.stringas[j] = (char)stringas[j];
                }
                item.intas = intas;
                item.doublas = doublas;
		hostData.push_back(item);
        }
}
int main() {
	host_vector<Item> hostData;
        read(hostData);
	device_vector<Item> deviceData = hostData;
	Item dest = {0, 0., ""};
	Item answer = reduce(deviceData.begin(), deviceData.end(), dest, Addition_func());
        cout << answer.intas << " "  << answer.doublas << "\n" << answer.stringas << "\n";
	return 0;
}

