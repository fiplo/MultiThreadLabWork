#include <iostream>
#include <fstream>
#include <string>
#include <omp.h>
#include <vector>
#include "json.hpp"

using namespace std;
using json = nlohmann::json;

class User {
    public:
		string Name;
		int Age;
		double Balance;

	User(string name, int age, double balance)
	{
		Name = name;
        Age = age;
        Balance = balance;
	}
	User(){}

	string getName() { return Name; }
	int getAge() { return Age; }
	double getBalance() { return Balance; }
};

vector<User> ParseJson(string fileName);

int main(int, char**) {
    string fileName = "Paulius_Ratkevicius_L1_dat_1.json";
    vector<User> users = ParseJson(fileName);
    
    return 0;
}

vector<User> ParseJson(string fileName){
    ifstream jsonFile(fileName);
    vector<User> users;
    json j = json::parse(jsonFile);
    for (auto it: j["users"].items() )
    {
        json &user = it.value();
        User entry(user["name"], user["age"], user["balance"]);
        users.push_back(entry);
    }
    return users;
}
