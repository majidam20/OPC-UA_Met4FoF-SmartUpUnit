
# _agentMet4FoF_ use case anomaly detection based on machine-learning

This Git repository is supported by European Metrology Programme for Innovation and Research (EMPIR)
under the project
[Metrology for the Factory of the Future (Met4FoF)](https://met4fof.eu), project number
17IND12.

## Anomaly detection

With the provided code, we showcase an agent-based machine learning approach for
 online anomaly detection of (in our case simulated) sensor readings.
  
## Getting started

If you are using PyCharm, you will already find proper run configurations at the
appropriate place in the IDE. It expects that you have prepared and defined a default
interpreter.

If you are not using PyCharm, of course, you can run the script files as usual.

If you have any questions, please get in touch with
[the author](https://github.com/majidam20).

### Dependencies

To install all dependencies in a virtual environment based on Python version 3.7, first
install `pip-tools` and afterward use our prepared `requirements.txt` to get
everything ready.

### Create a virtual environment on Windows

In your Windows command prompt, execute the following to set up a virtual environment
in a folder of your choice.

```shell
> python -m venv my_anomaly_detection_use_case_env
> my_anomaly_detection_use_case_env\Scripts\activate.bat
(my_anomaly_detection_use_case_env) > pip install --upgrade pip setuptools pip-tools
(my_anomaly_detection_use_case_env) > pip-sync
```

### Create a virtual environment on Mac and Linux

In your terminal, execute the following to set up a virtual environment in a folder of
 your choice.

```shell
$ python3.7 -m venv my_anomaly_detection_use_case_env
$ source my_anomaly_detection_use_case_env/bin/activate
(my_anomaly_detection_use_case_env) $ pip install --upgrade pip setuptools pip-tools
(my_anomaly_detection_use_case_env) $ pip-sync
```

### Scripts

The interesting parts you find in the file.

- `agentMET4FOF_anomaly_detection/anomaly_detection.py`

### Orphaned processes

In the event of agents not terminating cleanly, you can end all Python processes
running on your system (caution: the following commands affect **all** running Python
 functions, not just those that emerged from the agents).

#### Killing all Python processes in Windows

In your Windows command prompt, execute the following to terminate all python processes.

```shell
> taskkill /f /im python.exe /t
>
```

#### Killing all Python processes on Mac and Linux

In your terminal, execute the following to terminate all python processes.

```shell
$ pkill python
$
```

## References

For details about the agents, refer to the
[upstream repository _agentMET4FOF_](https://github.com/bangxiangyong/agentMET4FOF)

## Screenshot of web visualization
![Web Screenshot](https://github.com/bangxiangyong/agentMET4FOF/raw/master/docs/screenshot_met4fof.png)

## Developing

For development and testing, you should as well install the development dependencies
provided in the dev-requirements.txt for Python 3.8 as well.
 
```python
$ source my_anomaly_detection_use_case_env/bin/activate
(my_anomaly_detection_use_case_env) $ pip-sync dev-requirements.txt requirements.txt
```

You will find another run configuration for the test suite in your PyCharm IDE.
