#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <sys/socket.h>
#include <netinet/in.h>

int main() {
    // Define the server's IP address and port
    //char *server_ip = "127.0.0.1";  // localhost
    //int server_port = 12345;
    char *server_ip = "192.168.1.10";  // localhost
    int server_port = 9000;

    // Create a socket
    int client_socket = socket(AF_INET, SOCK_STREAM, 0);
    if (client_socket == -1) {
        perror("Socket creation failed");
        exit(EXIT_FAILURE);
    }

    // Connect to the server
    struct sockaddr_in server_address;
    server_address.sin_family = AF_INET;
    server_address.sin_port = htons(server_port);
    server_address.sin_addr.s_addr = inet_addr(server_ip);

    if (connect(client_socket, (struct sockaddr *)&server_address, sizeof(server_address)) < 0) {
        perror("Connection to server failed");
        exit(EXIT_FAILURE);
    }

    printf("Connected to the server\n");

    // Send data to the server
    char *message = "Hello, server!";
    write(client_socket, message, strlen(message));

    // Receive a response from the server
    char buffer[1024];
    memset(buffer, 0, sizeof(buffer));
    read(client_socket, buffer, sizeof(buffer));
    printf("Received response from server: %s\n", buffer);

    // Close the socket
    close(client_socket);

    return 0;
}