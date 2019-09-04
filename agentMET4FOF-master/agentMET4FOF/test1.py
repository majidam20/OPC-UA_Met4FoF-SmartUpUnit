
import socket
import messages_pb2
import numpy as np
from numpy_ringbuffer import RingBuffer
import collections


UDP_IP = "192.168.2.100"
UDP_PORT = 7000
sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)  # UDP
sock.bind((UDP_IP, UDP_PORT))


counter=0
d=[]
while True:
      data, addr = sock.recvfrom(1024)
      ProtoData = messages_pb2.DataMessage()
      ProtoData.ParseFromString(data)
      #print(ProtoData.Data_01)
      #r = RingBuffer(capacity=1000, dtype=np.float)
      #r.append(ProtoData.Data_01)
      #d = collections.deque(maxlen=10)
      d.append(ProtoData.Data_01)
      counter += 1
      if counter == 5:
            #print(dict(gps_data=r.popleft()))
          print(dict(gps_data=d))
          counter=0





