#!/usr/bin/env python
# coding: utf-8

# In[2]:


import os
#os.getcwd()


# In[7]:


#os.chdir("C:\\Users\\aminia01\\Desktop\\IOT\\agentMET4FOF-master\\agentMET4FOF-master")


# In[1]:


import google.protobuf as pf
import sys
import time
import socket
import messages_pb2
import threading
import pandas as pd
import csv
from datetime import datetime

import numpy as np
from numpy_ringbuffer import RingBuffer


#from skmultiflow.data.file_stream import FileStream

from agentMET4FOF.agents import AgentMET4FOF, AgentNetwork, MonitorAgent
from agentMET4FOF.streams import SensorGenerator
 
#Here we define a new agent SineGeneratorAgent, and override the functions : init_parameters & agent_loop
#init_parameters() is used to setup the data stream and necessary parameters
#agent_loop() is an infinite loop, and will read from the stream continuously,
#then it sends the data to its output channel via send_output
#Each agent has internal current_state which can be used as a switch by the AgentNetwork

class SensorGeneratorAgent(AgentMET4FOF):

    def init_parameters(self):
            self.stream = SensorGenerator()
            self.stream.prepare_for_use()
            self.counter = 0
            self.start =0
            self.r=[]
    def agent_loop(self):
        if self.current_state == "Running":

            sensor_data2 = self.stream.next_sample()
            #return self.send_output({'gps_data':sensor_data2})
            #r = RingBuffer(capacity=1000, dtype=np.float)
            #r.append( sensor_data2 )
            self.r.append(sensor_data2)
            self.counter += 1
            if self.counter == 10:
               #return \
               self.send_output(dict(gps_data=self.r))
               self.r=[]
               self.counter=0




def main():
    #start agent network server
    agentNetwork = AgentNetwork()

    #init agents by adding into the agent network
    gen_agent = agentNetwork.add_agent(agentType= SensorGeneratorAgent)
   # gen_agent.init_agent_loop(loop_wait=.001)
    monitor_agent = agentNetwork.add_agent(agentType= MonitorAgent)

    #connect agents by either way:
    # 1) by agent network.bind_agents(source,target)
    agentNetwork.bind_agents(gen_agent, monitor_agent)

    # 2) by the agent.bind_output()
    gen_agent.bind_output(monitor_agent)

    # set all agents states to "Running"
    agentNetwork.set_running_state()

    # allow for shutting down the network after execution
    return agentNetwork


if __name__ == '__main__':
   main()
'''
if __name__ == '__main__':
   agt_net = AgentNetwork(connect=True)
   monitor_agent = agt_net.get_agent("MonitorAgent_1")
   memory = monitor_agent.get_attr("memory")  # your collected sensor data should be here
   print(memory)
'''




