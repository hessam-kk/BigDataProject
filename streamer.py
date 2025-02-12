import socket
import time
import csv
from random import randint
# Configuration
host = "192.168.134.131"  # Spark Streaming will listen on this IP (update if necessary)
host = "localhost"  # Spark Streaming will listen on this IP (update if necessary)
port = 9999        # Port to send the data to (same as Spark configuration)
csv_file = "iris_extended_stream.csv"  # The CSV file to stream

# Create a socket connection
with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
    print(f"Connecting to {host}:{port}...")
    s.connect((host, port))
    print("Connected.")

    # Open the CSV file and start streaming
    with open(csv_file, 'r') as file:
        reader = csv.reader(file)
        header = next(reader)  # Skip header
        print("Skipping header:", header)

        for row in reader:
            # Send each row as a comma-separated string followed by a newline
            message = ','.join(row) + "\n"
            s.sendall(message.encode('utf-8'))
            print("Sent:", message.strip())
            
            # Wait for 1 second between rows to simulate streaming
            time.sleep(randint(0, 10)/10)

print("All data sent.")
