import socket

# Define the server's IP address and port
# Get ip address of host
s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
s.connect(("8.8.8.8",80))
server_ip = s.getsockname()[0]
#server_ip = '127.0.0.1'  # localhost
server_port = 9001

print("server_ip: " + server_ip + ": ")

# Create a socket object
client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
#client_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

# Connect to the server
client_socket.connect((server_ip, server_port))

# Send data to the server
message = "Hello, server!"
client_socket.send(message.encode('utf-8'))

# Receive a response from the server
response = client_socket.recv(1024).decode('utf-8')
print(f"Received response from server: {response}")

# Close the socket
client_socket.close()
