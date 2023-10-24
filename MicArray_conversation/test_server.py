import socket

# Define the server's IP address and port
# Get ip address of host
s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
s.connect(("8.8.8.8",80))
server_ip = s.getsockname()[0]
#server_ip = '127.0.0.1'  # localhost
server_port = 9001

# Create a socket object
server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

# Bind the socket to the IP address and port
server_socket.bind((server_ip, server_port))

# Listen for incoming connections
server_socket.listen(5)
print(f"Server is listening on {server_ip}:{server_port}")

# Accept a client connection
client_socket, client_address = server_socket.accept()
print(f"Connection from {client_address}")

# Receive data from the client
data = client_socket.recv(1024).decode('utf-8')
print(f"Received data from client: {data}")

# Send a response to the client
response = "Hello, client!"
client_socket.send(response.encode('utf-8'))

# Close the sockets
client_socket.close()
server_socket.close()