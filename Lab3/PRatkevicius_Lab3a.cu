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
const int STRING_SIZE = 260;
struct User {
    string Name;
    int Age;
    double Balance;
};


void read(Item users[], string fileName, int &count) {
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
    users[i].Name = Name;
		users[i].Age = Age;
		users[i].Balance = Balance;
    if( ifs.eof() ) break;
    i++;
	}
  count = i;

}
__global__ void addition(User* input, User* output, size_t n, int step)
{
	int index = threadIdx.x;
	for (int i = index; i < n; i += step) {
		output[index].Age += input[i].Age;
		output[index].Balance += input[i].Balance;
    output[index].Name += input[i].Name;
	}
}
int main() {

  string fileName = "../data/Paulius_Ratkevicius_L1_dat_1_ResPlain.txt";
  regex outputdir(".txt");
  string outputFile = regex_replace(fileName, outputdir, "Res.txt");

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

