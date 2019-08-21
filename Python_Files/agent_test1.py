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
#from skmultiflow.data.file_stream import FileStream


# In[4]:




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
       
    def agent_loop(self):
        if self.current_state == "Running":
            sensor_data1,sensor_data2 = self.stream.next_sample() #dictionary
            self.send_output({'time':sensor_data1, 'gps_data':sensor_data2})


# In[8]:


def main():
    #start agent network server
    agentNetwork = AgentNetwork()

    #init agents by adding into the agent network
    gen_agent = agentNetwork.add_agent(agentType= SensorGeneratorAgent)
    gen_agent.init_agent_loop(loop_wait=1)
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


# In[ ]:




