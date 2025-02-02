// stream_sender.go
package main

import (
	"bufio"
	"fmt"
	"log"
	"net"
	"os"
	"time"
)

func main() {
	ip := "172.30.248.16"
	port := "9999"
	// Connect to the streaming receiver on localhost:9999
	conn, err := net.Dial("tcp", ip+":"+port)
	if err != nil {
		log.Fatal("Error connecting:", err)
	}
	defer conn.Close()

	// Open the CSV file
	file, err := os.Open("../iris_extended_stream.csv")
	if err != nil {
		log.Fatal("Error opening file:", err)
	}
	defer file.Close()

	scanner := bufio.NewScanner(file)

	// Skip the header line
	if scanner.Scan() {
		header := scanner.Text()
		fmt.Println("Skipping header:", header)
	}

	// Read and send each record with a delay
	for scanner.Scan() {
		line := scanner.Text()
		// Send the line over the socket followed by a newline
		fmt.Fprintf(conn, "%s\n", line)
		fmt.Println("Sent:", line)
		// Wait 1 second before sending the next record
		time.Sleep(1 * time.Second)
	}

	if err := scanner.Err(); err != nil {
		log.Fatal("Error reading file:", err)
	}

	fmt.Println("All data sent.")
}
