//
// main.go
// Copyright (C) 2019 fiplo <fiplo@weebtop>
//
// Distributed under terms of the MIT license.
//

package main

import (
	"encoding/json"
	"fmt"
	"io"
	"io/ioutil"
	"log"
	"os"
	"regexp"
	"sort"
	"sync"
	"time"
)

var (
	ageReq int
)

func init() {
	ageReq = 30
}

func work(input *Monitor, output *Monitor, wg *sync.WaitGroup) {
	var work bool = true
	defer wg.Done()
	var user User
	for work {
		user = input.takeEntry()
		if Math.IsNaN(user.Balance) {
			work = false
		}
		if user.Age > ageReq && !IsNaN(user.Balance) {
			user.Balance = user.Balance * 1.1
			output.addEntry(user)
		}
	}
}

func main() {
	var inputFile = "../../data/Paulius_Ratkevicius_L1_dat_1.json"
	var users = ParseJSON(inputFile)
	var users2 Users
	var wg sync.WaitGroup
	var tCount int
	tCount = 4 // Count of filter threads
	wg.Add(tCount)

	var inputM Monitor
	inputM.setSize(10)
	var outputM Monitor
	outputM.setSize(users.Len())

	for i := 0; i < tCount; i++ {
		go work(&inputM, &outputM, &wg)
	}

	for i := 0; i < len(users.Users); i++ {
		inputM.addEntry(users.removeLast())
	}

	wg.Wait()
	var outputFile string
	reg := regexp.MustCompilePOSIX(".json")
	outputFile = reg.ReplaceAllString(inputFile, "Res.json")

	output, _ := json.MarshalIndent(users2, " ", "		")
	outputString := string(output)

	err := WriteToFile(outputFile, outputString)
	if err != nil {
		log.Fatal(err)
	}

}

//ParseJSON Parses Json information and returns it in a Users sturct format
func ParseJSON(inputFile string) Users {
	jsonFile, err := os.Open(inputFile)

	if err != nil {
		fmt.Println(err)
	}

	fmt.Println("Opened json file")

	defer jsonFile.Close()

	byteValue, _ := ioutil.ReadAll(jsonFile)
	var users Users

	json.Unmarshal(byteValue, &users)

	return users
}

//WriteToFile Writes given data string to a file located in filename.
func WriteToFile(outputFile string, data string) error {
	file, err := os.Create(outputFile)
	if err != nil {
		return err
	}
	defer file.Close()

	_, err = io.WriteString(file, data)
	if err != nil {
		return err
	}
	return file.Sync()
}

//Users struct that holds array of User
type Users struct {
	Users []User `json:"users"`
}

//User basic structure of an user
type User struct {
	Name    string  `json:"name"`
	Balance float64 `json:"balance"`
	Age     int     `json:"age"`
}

func (a Users) Len() int { return len(a.Users) }
func (a Users) Less(i, j int) bool {
	return a.Users[i].Age < a.Users[j].Age
}
func (a Users) Swap(i, j int) {
	a.Users[i], a.Users[j] = a.Users[j], a.Users[i]
}

func (a Users) addUser(user User) Users {
	a.Users = append(a.Users, user)
	sort.Sort(Users(a))
	return a
}

func (a Users) removeLast() User {
	var returnValue User
	if a.Len() > 0 {
		returnValue = a.Users[a.Len()-1]
		a.Users = a.Users[:a.Len()-1]
	}
	return returnValue
}

type Monitor struct {
	users   Users
	mutex   sync.Mutex
	maxSize int
	index   int
	done    bool
}

func (a Monitor) setSize(size int) {
	a.maxSize = size
	a.done = false
	a.index = 0
}

func (a Monitor) addEntry(user User) {
	a.mutex.Lock()
	for a.users.Len() >= a.maxSize {
		a.mutex.Unlock()
		time.Sleep(2 * time.Millisecond)
		a.mutex.Lock()
	}
	a.users.addUser(user)
	a.index++
	a.mutex.Unlock()
}

func (a Monitor) takeEntry() User {
	var user User
	a.mutex.Lock()
	for a.index == 0 {
		if a.done {
			a.mutex.Unlock()
			user.Balance = NaN()
			return user
		}
		a.mutex.Unlock()
		time.Sleep(2 * time.Millisecond)
		a.mutex.Lock()
	}
	user = a.users.removeLast()
	a.index--
	a.mutex.Unlock()
	return user
}

func (a Monitor) returnUsers() Users {
	return a.users
}
