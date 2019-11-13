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
)

const (
	ageReq      int = 30
	maxSize     int = 50
	threadCount int = 4
)

func main() {

	var inputFile = "../../data/Paulius_Ratkevicius_L1_dat_1.json"
	var users Users = ParseJson(inputFile)

	inputBuff := make(chan User)
	inputToMiddle := make(chan User)
	middleToOutput := make(chan User)
	resultBuff := make(chan Users)
	LITERALLYWORKERGROUPS := make(chan bool)

	go inputControl(inputBuff, inputToMiddle)
	go outputControl(middleToOutput, resultBuff)

	for i := 0; i < threadCount; i++ {
		go worker(inputToMiddle, middleToOutput, LITERALLYWORKERGROUPS)
	}

	for users.Len() > 0 {
		inputBuff <- users.removeLast()
	}
	close(inputBuff)
	i := 0
	for i < threadCount {
		<-LITERALLYWORKERGROUPS
		i++
	}
	close(middleToOutput)

	users2 := <-resultBuff

	output, _ := json.MarshalIndent(users2, " ", "		")
	outputString := string(output)

	reg := regexp.MustCompilePOSIX(`dat_.\.json`)

	outputFile := reg.ReplaceAllString(inputFile, "resL2.json")

	err := WriteToFile(outputFile, outputString)
	if err != nil {
		log.Fatal(err)
	}

}

// Parses Json information and returns it in a Users sturct format
func ParseJson(inputFile string) Users {
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

// Writes given data string to a file located in filename.
func WriteToFile(filename string, data string) error {
	file, err := os.Create(filename)
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

//// This is struct of Users

type Users struct {
	Users []User `json:"users"`
}

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

func (a *Users) addUser(user User) {
	a.Users = append(a.Users, user)
	sort.Sort(Users(*a))
}

func (a *Users) removeLast() User {
	var returnValue User
	if a.Len() > 0 {
		returnValue = a.Users[a.Len()-1]
		a.Users = a.Users[:a.Len()-1]
	}
	return returnValue
}

//// This is func of input controller

func inputControl(input <-chan User, output chan<- User) {
	var arr Users
	var done bool = false
	var readDone bool = false
	for !done {
		//Keeps reading until fills up to MaxSize/(Threadcount*2) or channel is closed and empty.
		for arr.Len() < maxSize/(threadCount*2) && !readDone {
			user, innerDone := <-input
			if innerDone {
				arr.addUser(user)
			}
			readDone = !innerDone
		}
		//Unloads data
		for arr.Len() > 0 {
			output <- arr.removeLast()
		}
		//If channel is closed, stops the loop.
		done = readDone
	}
	close(output)
}

//// This is a func of output controller

func outputControl(input <-chan User, output chan<- Users) {
	var arr Users
	var done bool = false
	//Reads until channel is closed and empty.
	for !done {
		user, innerDone := <-input
		if innerDone {
			arr.addUser(user)
		}
		done = !innerDone
	}
	output <- arr
}

//// This is a func of worker thread

func worker(input <-chan User, output chan<- User, catch chan<- bool) {
	var done bool
	for !done {
		user, innerDone := <-input
		if user.Age >= ageReq && innerDone {
			output <- user
		}
		done = !innerDone
	}
	catch <- true
}
