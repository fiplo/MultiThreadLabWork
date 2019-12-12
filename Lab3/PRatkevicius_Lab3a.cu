#include <string>
#include "cuda_runtime.h"
#include <cuda.h>
#include <cstdio>
#include <iostream>
#include "device_launch_parameters.h"
#include <algorithm>
#include <fstream>

using namespace std;

const int ARRAY_SIZE = 50;
const int CHAR_ARRAY_SIZE = 11;


struct User {
    char Name;
    int Age;
    double Balance;
};


void read(User users[], string fileName, int &count) {
	ifstream  ifs(fileName);
	string Name;
	int Age;
	double Balance;
  int i = 0;

	if (ifs.fail()) {
		cout << "Error opening file" + fileName << endl;
		exit(1);
	}
	while(!ifs.fail()){
		ifs >> Name >> Age >> Balance;
		users[i].Age = Age;
		users[i].Balance = Balance;
    for (int j = 0; j < CHAR_ARRAY_SIZE;j++) {
			if (Name[j] == 0) {
				users[i].Name[j] = 0;
				break;
			}
			users[i].Name[j] = (char)Name[j];
		}
    if( ifs.eof() ) break;
    i++;
	}
  count = i;

}
__global__ void addition(User* input, User* output, size_t n, int step)
{
	int index = threadIdx.x;
	for (int i = index; i < n; i += step) {
    int iterator = 0;
		output[index].Age += input[i].Age;
		output[index].Balance += input[i].Balance;
    for (int j = 0; j < CHAR_ARRAY_SIZE; j++) {
			if (input[i].Name[j] == 0) {
				break;
			}
			output[index].Name[iterator] = input[i].Name[j];
			iterator++;
		}
	}
}
int main() {

  string fileName = "../data/Paulius_Ratkevicius_L1_dat_1_ResPlain.txt";

  int counter;

	int threads = 4;
	User input[ARRAY_SIZE];
	read(input, fileName, counter);
	User* hostData = input;
	User* deviceData;
	User* hostA = new User[counter];
	User* deviceA;// = new User[ARRAY_SIZE];
	cudaMalloc(&deviceData, counter * sizeof(User));
	cudaMalloc(&deviceA, counter * sizeof(User));
	cudaMemcpy(deviceData, input, counter * sizeof(User), cudaMemcpyHostToDevice);
	addition << <1, threads >> > (deviceData, deviceA, counter, threads);
	//cudaMemcpy(hostData, deviceData, ARRAY_SIZE * sizeof(User), cudaMemcpyDeviceToHost);
	cudaMemcpy(hostA, deviceA, counter * sizeof(User), cudaMemcpyDeviceToHost);
	for (int i = 0; i < threads; i++) {
		cout << hostA[i].Name << " " << hostA[i].Age << " " << hostA[i].Balance << "\n";
	}
	return 0;
}

