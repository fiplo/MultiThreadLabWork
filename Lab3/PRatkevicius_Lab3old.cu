#include <string>
#include "cuda_runtime.h"
#include <cuda.h>
#include <cstdio>
#include <iostream>
#include "device_launch_parameters.h"
#include <algorithm>
#include <fstream>
#include "json.hpp"

using namespace std;
using json = nlohmann::json;

const int ARRAY_SIZE = 27;//25;

namespace ns{
    struct User {
        string Name;
        int Age;
        double Balance;
    };


    void to_json(json& j, const User& p){
        j = json{{"name", p.Name}, {"age", p.Age}, {"balance", p.Balance}};
    }

    void from_json(const json& j, User& p){
        j.at("name").get_to(p.Name);
        j.at("age").get_to(p.Age);
        j.at("balance").get_to(p.Balance);
    }

}

ns::User* ParseJson(string fileName){
    ifstream fileStream(fileName);
    if (fileStream.fail()) {
        cout << "Error while opening stream of input file: " << fileName << endl;
        exit(1);
    }
    static ns::User users[ARRAY_SIZE];
    int counter = 0;
    json j = json::parse(fileStream);
    for (auto it: j.items() )
    {
        json &user = it.value();
        users[counter++] = user.get<ns::User>();
    }
    return users;
}
void OutputJson(string fileName, ns::User users[]){
    json output;
    int i = 0;
    while(users[i].Age != 0 && i < ARRAY_SIZE){
        output.push_back(users[i]);
        i++;
    }
    ofstream results(fileName);
    results << output.dump(1, '\t') << endl;
}

__global__ void addition(ns::User* data, ns::User* sum, size_t n, int step)
{
	int index = threadIdx.x;
	for (int i = index; i < n; i += step) {
		sum[index].Age += data[i].Age;
		sum[index].Balance += data[i].Balance;	
    sum[index].Name += data[i].Name[j];
	}
}
int main() {
  string fileName = "../data/Paulius_Ratkevicius_L1_dat_1.json";
  regex outputdir(".json");
  string outputFile = regex_replace(fileName, outputdir, "_Res3.json");

	int threads = 4;
  ns::User input = ParseJson(fileName);
  ns::User* hostData = input;
  ns::User* deviceData;
	Answer* hostA = new ns::User[ARRAY_SIZE];
	Answer* deviceA;// = new Answer[ARRAY_SIZE];
	cudaMalloc(&deviceData, ARRAY_SIZE * sizeof(ns::User));
	cudaMalloc(&deviceA, ARRAY_SIZE * sizeof(ns::User));
	cudaMemcpy(deviceData, input, ARRAY_SIZE * sizeof(ns::User), cudaMemcpyHostToDevice);
	addition << <1, threads >> > (deviceData, deviceA, ARRAY_SIZE, threads);
	//cudaMemcpy(hostData, deviceData, ARRAY_SIZE * sizeof(Item), cudaMemcpyDeviceToHost);
	cudaMemcpy(hostA, deviceA, ARRAY_SIZE * sizeof(ns::User), cudaMemcpyDeviceToHost);
	for (int i = 0; i < threads; i++) {
		cout << hostA[i].Name << " " << hostA[i].Age << " " << hostA[i].Balance << "\n";

	}
	return 0;
}

