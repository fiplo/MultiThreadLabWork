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
static const int MAX_USERS = 10;
static const int USER_AGE = 30;
static const struct timespec waittime = {0, 1000000};


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

class Users{
    private:
        vector<ns::User> users;
    public:
        Users(){}
        void addUser(ns::User user){
            users.push_back(user);
            std::sort(users.begin(), users.end(), [](ns::User a, ns::User b) { return a.Balance < b.Balance; });
        }
        ns::User    getUser(int id) { return users.at(id); }
        int     getLength() { return users.size(); }
        ns::User    getLastRemove() {
            ns::User back = users.back();
            users.pop_back();
            return back;
        }
};

class Monitor{
    private:
        Users users;
        int index = 0;
        omp_lock_t mux;
        int maxMonitor;
    public:
        bool done = false;
        Monitor() {
            omp_init_lock(&mux);
        }

        ~Monitor() {
            omp_destroy_lock(&mux);
        }

        void setSize(int size){
            maxMonitor = size;
        }

        ns::User take(){
            omp_set_lock(&mux);
            while (index == 0) {
                if (done) {
                    omp_unset_lock(&mux);
                    ns::User empty;
                    empty.Balance = nan("");
                    return empty;
                }
                omp_unset_lock(&mux);
                nanosleep(&waittime, NULL);
                omp_set_lock(&mux);
            }
            ns::User ret = users.getLastRemove();
            index--;
            omp_unset_lock(&mux);
            return ret;
        }

        void place(ns::User user){
            omp_set_lock(&mux);
            while (users.getLength() >= maxMonitor){
                omp_unset_lock(&mux);
                nanosleep(&waittime, NULL);
                omp_set_lock(&mux);
            }
            users.addUser(user);
            index++;
            omp_unset_lock(&mux);
        }

        Users &TakeEntireList(){
            return users;
        }
};
        

Users ParseJson(string fileName);
void Work(Monitor& in, Monitor& out);
void OutputJson(string filename, Users users);

int main(int, char**) {
    string fileName = "../data/Paulius_Ratkevicius_L1_dat_3.json";
    regex outputdir(".json");
    string outputFile = regex_replace(fileName, outputdir, "_Res.json");
    Users users = ParseJson(fileName);
    Monitor in;
    Monitor out;
    in.setSize(10);
    out.setSize(users.getLength());


#pragma omp parallel num_threads(4)
    {
        if(omp_get_thread_num() == 0) {
            while(users.getLength() > 1){
                in.place(users.getLastRemove());
            }
            in.done = true;
        } else {
            Work(in, out);
        }
#pragma omp barrier
    }
    users = out.TakeEntireList();
    OutputJson(outputFile, users);
    return 0;
}

Users ParseJson(string fileName){
    ifstream fileStream(fileName);
    if (fileStream.fail()) {
        cout << "Error while opening stream of input file: " << fileName << endl;
        exit(1);
    }
    Users users;
    json j = json::parse(fileStream);
    for (auto it: j["users"].items() )
    {
        json &user = it.value();
        users.addUser(user.get<ns::User>());
    }
    return users;
}

void Work(Monitor& in, Monitor& out){
    ns::User user;
    bool work = true;
    while(work){
        user = in.take();
        if(isnan(user.Balance)) work = false;
        if(user.Age > USER_AGE && !isnan(user.Balance)){
            user.Balance = user.Balance * 1.1;
            out.place(user);
        }
    }
}

void OutputJson(string fileName, Users users){
    json output;
    while(users.getLength() > 1){
        output.push_back(users.getLastRemove());
    }
    ofstream results(fileName);
    results << output.dump(1, '\t') << endl;
}
