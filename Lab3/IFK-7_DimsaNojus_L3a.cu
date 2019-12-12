#include <string>
#include "cuda_runtime.h"
#include <cuda.h>
#include <cstdio>
#include <iostream>
#include "device_launch_parameters.h"
#include <algorithm>
#include <fstream>

using namespace std;

const int ARRAY_SIZE = 27;//25;
const int CHAR_ARRAY_SIZE = 11;
const int STRING_SIZE = 260;

struct Item {
	char stringas[CHAR_ARRAY_SIZE];
	int intas;
	double doublas;
};
struct Answer {
	char stringas[STRING_SIZE];
	int intas;
	double doublas;
};
void read(Item items[]) {
	//ifstream  ifs("IFK-7_NojusD_L3_dat.txt");
	ifstream  ifs("IFK_NojusD_L3_dat.txt");
	string stringas;
	int intas;
	double doublas;

	if (ifs.fail()) {
		cout << "Error opening file (IFK-7_NojusD_L3_dat.txt)" << endl;
		exit(1);
	}
	for(size_t i = 0; i <ARRAY_SIZE; i++){
		ifs >> stringas >> intas >> doublas;
		for (int j = 0; j < CHAR_ARRAY_SIZE;j++) {
			if (stringas[j] == 0) {
				items[i].stringas[j] = 0;
				break;
			}
			items[i].stringas[j] = (char)stringas[j];
		}
		items[i].intas = intas;
		items[i].doublas = doublas;
	}

}
__global__ void addition(Item* data, Answer* sum, size_t n, int step)
{
	int index = threadIdx.x;
	int iterator = 0;
	for (int i = index; i < n; i += step) {
		sum[index].intas += data[i].intas;
		sum[index].doublas += data[i].doublas;
		for (int j = 0; j < CHAR_ARRAY_SIZE; j++) {
			if (data[i].stringas[j] == 0) {
				break;
			}
			sum[index].stringas[iterator] = data[i].stringas[j];
			//printf("%c", sum[thread_id].stringas[iterator]);
			iterator++;
		}
	}
}
int main() {
	int threads = 4;
	Item items[ARRAY_SIZE];
	read(items);
	Item* hostData = items;
	Item* deviceData;
	Answer* hostA = new Answer[ARRAY_SIZE];
	Answer* deviceA;// = new Answer[ARRAY_SIZE];
	cudaMalloc(&deviceData, ARRAY_SIZE * sizeof(Item));
	cudaMalloc(&deviceA, ARRAY_SIZE * sizeof(Answer));
	cudaMemcpy(deviceData, items, ARRAY_SIZE * sizeof(Item), cudaMemcpyHostToDevice);
	addition << <1, threads >> > (deviceData, deviceA, ARRAY_SIZE, threads);
	//cudaMemcpy(hostData, deviceData, ARRAY_SIZE * sizeof(Item), cudaMemcpyDeviceToHost);
	cudaMemcpy(hostA, deviceA, ARRAY_SIZE * sizeof(Answer), cudaMemcpyDeviceToHost);
	for (int i = 0; i < threads; i++) {
		cout << hostA[i].intas << " " << hostA[i].doublas << " " << hostA[i].stringas << "\n";
	}
	return 0;
}

