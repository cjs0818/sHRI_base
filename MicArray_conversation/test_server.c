#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <sys/socket.h>
#include <netinet/in.h>

int main() {
    // Define the server's IP address and port
    char *server_ip = "127.0.0.1";  // localhost
    int server_port = 12345;

    // Create a socket
    int server_socket = socket(AF_INET, SOCK_STREAM, 0);
    if (server_socket == -1) {
        perror("Socket creation failed");
        exit(EXIT_FAILURE);
    }

    // Bind the socket to the IP address and port
    struct sockaddr_in server_address;
    server_address.sin_family = AF_INET;
    server_address.sin_addr.s_addr = inet_addr(server_ip);
    server_address.sin_port = htons(server_port);

    if (bind(server_socket, (struct sockaddr *)&server_address, sizeof(server_address)) < 0) {
        perror("Binding failed");
        exit(EXIT_FAILURE);
    }

    // Listen for incoming connections
    if (listen(server_socket, 5) < 0) {
        perror("Listening failed");
        exit(EXIT_FAILURE);
    }

    printf("Server is listening on %s:%d\n", server_ip, server_port);

    // Accept a client connection
    int client_socket = accept(server_socket, NULL, NULL);
    if (client_socket < 0) {
        perror("Accepting connection failed");
        exit(EXIT_FAILURE);
    }

    printf("Connection established with the client\n");

    // Receive data from the client
    char buffer[1024];
    memset(buffer, 0, sizeof(buffer));
    read(client_socket, buffer, sizeof(buffer));
    printf("Received data from client: %s\n", buffer);

    // Send a response to the client
    char *response = "Hello, client!";
    write(client_socket, response, strlen(response));

    // Close the sockets
    close(client_socket);
    close(server_socket);

    return 0;
}