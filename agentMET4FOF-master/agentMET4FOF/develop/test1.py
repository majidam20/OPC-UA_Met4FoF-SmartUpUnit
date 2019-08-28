
import socket
import messages_pb2


UDP_IP = "192.168.2.100"
UDP_PORT = 7000
sock = socket.socket(socket.AF_INET,  # Internet
                                  socket.SOCK_DGRAM)  # UDP
# self.sock.bind((UDP_IP, UDP_PORT))

HEADERSIZE = 200

def main():
    while True:
        full_msg = ''
        new_msg = True
        while True:
             data, addr = sock.recvfrom(1024)
             if new_msg:
                print("new msg len:", data[:HEADERSIZE])
                msglen = data[:HEADERSIZE]
                new_msg = False

             print(f"full message length: {msglen}")

             ProtoData = messages_pb2.DataMessage()
             ProtoData.ParseFromString(data)
             full_msg += str(ProtoData.Data_01)

             print(len(full_msg))

        if len(full_msg) - HEADERSIZE == msglen:
            print("full msg recvd")
            print(full_msg[HEADERSIZE:])
            return full_msg[HEADERSIZE:]
            new_msg = True
            full_msg = ""