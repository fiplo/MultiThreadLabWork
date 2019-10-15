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

func main() {

	var inputFile = "../../data/Paulius_Ratkevicius_L1_dat_1.json"
	var users = ParseJson(inputFile)

	inputBuff := make(chan User, 10)
	exportBuff := make(chan User, 10)
	resultBuff := make(chan Users)

	var wg sync.WaitGroup

	var tCount int
	tCount = 4 // Count of filter threads

	wg.Add(tCount)
	for i := 0; i < tCount; i++ {
		go FilterData(inputBuff, exportBuff, 30, &wg)
	}

	go FetchFromChan(exportBuff, resultBuff)

	for i := 0; i < len(users.Users); i++ {
		inputBuff <- users.Users[i]
	}
	close(inputBuff)

	wg.Wait()
	var users2 = <-resultBuff

	close(exportBuff)

	output, _ := json.MarshalIndent(users2, " ", "		")
	outputString := string(output)

	reg := regexp.MustCompilePOSIX(`dat_.\.json`)

	outputFile := reg.ReplaceAllString(inputFile, "res.json")

	err := WriteToFile(outputFile, outputString)
	if err != nil {
		log.Fatal(err)
	}

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

// Filters data from inputBuff and exports it to exportBuff
// Filter picks user entries that are above given age.
func FilterData(inputBuff <-chan User, exportBuff chan<- User, ageReq int, wg *sync.WaitGroup) {
	defer wg.Done()
	var ok bool = true
	for ok {
		user, status := <-inputBuff
		if !status { //A bit hacky, I actually don't know what happens when buffer is closed and there's still data in it (supposedly status checks for Closed AND Empty (?))
			ok = false
		}
		if user.Age > ageReq {
			exportBuff <- user
		}
	}

}

// Takes data from inputBuff channel and processes it into struct Users,
// which is exported into exportBuff channel
func FetchFromChan(inputBuff <-chan User, exportBuff chan<- Users) {
	var ok bool = true
	var users2 Users
	for ok {
		select {
		case user := <-inputBuff:
			users2 = users2.addUser(user)
		case <-time.After(5 * time.Second): //need to replace this with some other proper way to close thread
			ok = false
		}
	}
	exportBuff <- users2
}

type Users struct {
	Users []User `json:"users"`
}

type User struct {
	Name    string  `json:"name"`
	Balance float64 `json:"balance"` //Why do you struggle to read numbers????? (Apparently float numbers can't be in parenthesis, eventhough that's default for json???)
	Age     int     `json:"age"`
}

func (a Users) Len() int { return len(a.Users) }
func (a Users) Less(i, j int) bool {
	return a.Users[i].Age < a.Users[j].Age
}
func (a Users) Swap(i, j int) {
	a.Users[i], a.Users[j] = a.Users[j], a.Users[i]
}

func (e Users) addUser(user User) Users {
	e.Users = append(e.Users, user)
	sort.Sort(Users(e))
	return e
}
