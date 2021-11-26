[![Documentation Status](https://readthedocs.org/projects/agentmet4fof/badge/?version=latest)](https://agentmet4fof.readthedocs.io/en/latest/?badge=latest)
[![Codecov Badge](https://codecov.io/gh/bangxiangyong/agentMet4FoF/branch/master/graph/badge.svg)](https://codecov.io/gh/bangxiangyong/agentMet4FoF)

# Multi-Agent System for Metrology for Factory of the Future (Met4FoF) Code
This Git repository is supported by European Metrology Programme for Innovation and Research (EMPIR) under the project Metrology for the Factory of the Future (Met4FoF), project number 17IND12. (https://www.ptb.de/empir2018/met4fof/home/)

About
---
 - How can metrological data be incorporated into an agent-based system for addressing the uncertainty of machine learning in future manufacturing?
 - Includes agent-based simulation and implementation
 - Readthedocs documentation is available at (https://agentmet4fof.readthedocs.io)

Get started
---
First, please clone the repository to your local machine as described
[here](https://help.github.com/en/articles/cloning-a-repository). To get started
with your present *Anaconda* installation, please go to *Anaconda
prompt*, navigate to your local clone
```
cd /your/local/folder/agentMet4FoF
```
and execute
```
conda env create --file environment.yml 
```
This command will create an *Anaconda* virtual environment with all dependencies
satisfied. If you don't have *Anaconda* installed already, follow [this guide
](https://docs.conda.io/projects/continuumio-conda/en/latest/user-guide/install/download.html)
Firstly, then create the virtual environment as stated above and then proceed.

First take a look at the [tutorials](agentMET4FOF-master/tutorials/tutorial_1_generator_agent.py) and [examples](agentMET4FOF-master/examples)
Alternatively, start hacking the [main_agent_network.py](agentMET4FOF-master/agentMET4FOF/main_agent_network.py) if you are already
familiar with agentMet4FoF and want to customize your agents' network.

Updates
---
 - Implemented base class AgentMET4FOF with built-in agent classes DataStreamAgent, MonitorAgent
 - Implemented class AgentNetwork to start or connect to a agent server
 - Implemented with ZEMA prognosis of Electromechanical cylinder data set as use case [![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.1326278.svg)](https://doi.org/10.5281/zenodo.1326278)
 - Implemented interactive web application with user interface


## Screenshot of web visualization
![Web Screenshot](agentMET4FOF-master/docs/screenshot_met4fof.png)

Note
---
 - In the case of agents not terminating cleanly, run ```taskkill /f /im python.exe /t``` in Windows Command Prompt to remove all background python processes.
