#include <iostream>
#include <fstream>
#include <string>
#include <omp.h>
#include "json.hpp"
#include <unistd.h>
#include <time.h>
#include <regex>
#include <cmath>

using namespace std;
using json = nlohmann::json;
static const int ARRAY_SIZE = 60;


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
        users[counter] = user.get<ns::User>();
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

int main(int, char**) {
    string fileName = "../data/Paulius_Ratkevicius_L1_dat_1.json";
    regex outputdir(".json");
    string outputFile = regex_replace(fileName, outputdir, "_Res3.json");
    ns::User* users = ParseJson(fileName);
    OutputJson(outputFile, users);
    return 0;
}
