#! Author is Mr.Majid Aminian(majidam66@gmail.com)
#
# Contributors:
#
# - "Maximilian Gruber" <maximilian.gruber@ptb.de>,
# - "Bang Xiang Yong" <bxy20@cam.ac.uk>,
# - "Björn Ludwig" <bjoern.ludwig@ptb.de>,

import copy
import gc
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objs as go
import torch
from agentMET4FOF.agents import AgentMET4FOF, AgentNetwork, MonitorAgent
from agentMET4FOF.streams import DataStreamMET4FOF
from numpy.random import seed
from scipy import stats
from torch import nn

########################################################################################
# Defined random_seed because of willing same raw data in every running.
random_seed = 42
np.random.seed(random_seed)
torch.manual_seed(random_seed)

# Definition of process engine of pytorch model, so if some machine does not have GPU
# it could be used CPU instead.
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = torch.device("cpu")


class SineGeneratorAgent(AgentMET4FOF):
    # Generating raw data
    def init_parameters(self, scale_amplitude=1):
        # Setup a sine stream as in the original version written for agentMET4FOF 0.2.0.
        sine_stream = np.sin(np.arange(0, 3.142 * 1000000, 0.5))
        stream = DataStreamMET4FOF()
        stream.set_data_source(quantities=sine_stream)

        self.stream = stream
        self.scale_amplitude = scale_amplitude

    def agent_loop(self):
        if self.current_state == "Running":
            sine_data = self.stream.next_sample()

            current_time = time.time()

            sine_data = {
                "time": current_time,
                "y1": sine_data["quantities"] * self.scale_amplitude,
            }

            self.buffer.store(agent_from=self.name, data=sine_data)
            if self.buffer.buffer_filled(self.name):
                self.send_output(self.buffer[self.name])
                self.buffer.clear()


########################################################################################
# Auto encoder consisted by two classes Encoder and Decoder
# There are two types of Encoder and Decoder that names withLSTM and withoutLSTM
# Definition of three important variables in pytorch model
"""
seq_len means number of columns
n_seq means number of rows
n_features means each tensor that is one
"""


# Each of Encoder_withLSTM and Decoder_withLSTM has two layers of LSTM that is called
# rnn1 and rnn2
class Encoder_withLSTM(nn.Module):
    def __init__(self, seq_len, n_features, embedding_dim=64):
        super(Encoder_withLSTM, self).__init__()

        self.seq_len, self.n_features = seq_len, n_features
        self.embedding_dim, self.hidden_dim = embedding_dim, 2 * embedding_dim

        self.rnn1 = nn.LSTM(
            input_size=n_features,
            hidden_size=self.hidden_dim,
            num_layers=1,
            batch_first=True,
        )

        self.rnn2 = nn.LSTM(
            input_size=self.hidden_dim,
            hidden_size=embedding_dim,
            num_layers=1,
            batch_first=True,
        )

    def forward(self, x):
        x = x.reshape((1, self.seq_len, self.n_features))

        x, (_, _) = self.rnn1(x)
        x, (hidden_n, _) = self.rnn2(x)

        return hidden_n.reshape((self.n_features, self.embedding_dim))


#########################################################################
class Decoder_withLSTM(nn.Module):
    def __init__(self, seq_len, input_dim=64, n_features=1):
        super(Decoder_withLSTM, self).__init__()

        self.seq_len, self.input_dim = seq_len, input_dim
        self.hidden_dim, self.n_features = 2 * input_dim, n_features

        self.rnn1 = nn.LSTM(
            input_size=input_dim, hidden_size=input_dim, num_layers=1, batch_first=True
        )

        self.rnn2 = nn.LSTM(
            input_size=input_dim,
            hidden_size=self.hidden_dim,
            num_layers=1,
            batch_first=True,
        )

        self.output_layer = nn.Linear(self.hidden_dim, n_features)

    def forward(self, x):
        x = x.repeat(self.seq_len, self.n_features)
        x = x.reshape((self.n_features, self.seq_len, self.input_dim))

        x, _ = self.rnn1(x)
        x, _ = self.rnn2(x)
        x = x.reshape((self.seq_len, self.hidden_dim))

        return self.output_layer(x)


#########################################################################
# Implementation of the encoder network
# Each of Encoder_withoutLSTM and Decoder_withoutLSTM has two layers that is called L1
# and L2
class Encoder_withoutLSTM(nn.Module):
    def __init__(self, seq_len, n_features, embedding_dim=64):
        super(Encoder_withoutLSTM, self).__init__()

        self.seq_len, self.n_features = seq_len, n_features
        self.embedding_dim, self.hidden_dim = embedding_dim, 2 * embedding_dim

        self.encoder_L1 = nn.Linear(self.n_features, self.hidden_dim, bias=True)
        self.encoder_R1 = nn.ReLU(True)

        self.encoder_L2 = nn.Linear(self.hidden_dim, self.embedding_dim, bias=True)
        self.encoder_R2 = nn.ReLU(True)

    def forward(self, x):
        x = self.encoder_L1(x)
        x = self.encoder_L2(x)

        return x


########################################################################################
# implementation of the decoder network
class Decoder_withoutLSTM(nn.Module):
    def __init__(self, seq_len, input_dim=64, n_features=1):
        super(Decoder_withoutLSTM, self).__init__()

        self.seq_len, self.input_dim = seq_len, input_dim
        self.hidden_dim, self.n_features = 2 * input_dim, n_features

        self.decoder_L1 = nn.Linear(
            in_features=self.input_dim, out_features=self.hidden_dim
        )  # add linearity
        self.decoder_R1 = nn.ReLU(True)  # add non-linearity according to [2]

        self.decoder_L2 = nn.Linear(
            in_features=self.hidden_dim, out_features=self.n_features, bias=True
        )  # add linearity
        self.decoder_R2 = nn.ReLU(True)  # add non-linearity according to [2]

    def forward(self, x):
        # define forward pass through the network
        x = self.decoder_L1(x)
        x = self.decoder_L2(x)

        return x


##########################################################################


class RecurrentAutoencoder(nn.Module):
    def __init__(self, seq_len, n_features, embedding_dim=64, model_type="withoutLSTM"):
        super(RecurrentAutoencoder, self).__init__()

        if model_type == "withoutLSTM":
            self.encoder = Encoder_withoutLSTM(seq_len, n_features, embedding_dim).to(
                device
            )
            self.decoder = Decoder_withoutLSTM(seq_len, embedding_dim, n_features).to(
                device
            )

        elif model_type == "withLSTM":
            self.encoder = Encoder_withLSTM(seq_len, n_features, embedding_dim).to(
                device
            )
            self.decoder = Decoder_withLSTM(seq_len, embedding_dim, n_features).to(
                device
            )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)

        return x


########################################################################################
# Aggregator class aggregates three different raw sensors data to one output sensor data
class Aggregator(AgentMET4FOF):
    def on_received_message(self, message):
        self.buffer.store(message)
        self.log_info(f"message:{message}")
        self.log_info("self.buffer:" + str(self.buffer))

        if (
            "Sensor1" in self.buffer.keys()
            and "Sensor2" in self.buffer.keys()
            and "Sensor3" in self.buffer.keys()
        ):
            # Time selected from Sensor1_1, so Time of Sensor2_1 and Sensor3_1 are
            # omitted
            a = pd.DataFrame(self.buffer["Sensor1"])
            b = pd.DataFrame(self.buffer["Sensor2"]["y1"])
            c = pd.DataFrame(self.buffer["Sensor3"]["y1"])

            # Concatenate there different sensor data and remove duplicated time
            # columns, so there is one dataframe with one time column and three
            # different sensor data in three columns
            agg_df = pd.concat([a, b, c], axis=1)
            agg_df.columns = ["time", "y1", "y2", "y3"]

            self.buffer.clear()
            self.log_info(f"agg_df:{agg_df}")
            self.send_output(agg_df.to_dict("list"))


# Disturbance class receives data from Aggregator and add anomalies to received data
class Disturbance(AgentMET4FOF):
    def on_received_message(self, message):
        self.log_info(f"agg_message:{message}")
        now = datetime.now()

        # Generate every 5 seconds random anomaly data and multiple to each sensor
        # data, random anomaly data multiplied with fixed decimal number because
        # purpose is different anomalies for each sensor
        if now.second % 5 == 0:
            # added Anomalies value for checking that data is anomaly or not
            message["data"].update({"Anomalies": [True]})
            self.log_info(f"message_Anomalies:{message}")

            random_anomaly = np.random.uniform(-2, 2, size=1)
            self.log_info(f"r:{random_anomaly}")
            X_test_df = pd.DataFrame(
                [
                    message["data"]["time"],
                    message["data"]["y1"] * random_anomaly,
                    message["data"]["y2"] * random_anomaly * 0.3,
                    message["data"]["y3"] * random_anomaly * 0.6,
                    message["data"]["Anomalies"],
                ]
            )

            X_test_df = X_test_df.T
            X_test_df.columns = ["time", "y1", "y2", "y3", "Anomalies"]
            self.log_info(f"abnormal_test:{X_test_df}")

        # Generate normal data during every 5 seconds
        if now.second % 5 != 0:
            X_test_df = pd.DataFrame(
                [
                    message["data"]["time"],
                    message["data"]["y1"],
                    message["data"]["y2"],
                    message["data"]["y3"],
                ]
            )
            X_test_df = X_test_df.T

            X_test_df.columns = ["time", "y1", "y2", "y3"]
            self.log_info(f"normal_test:{X_test_df}")

        self.log_info(f"Disturbance_X_test_df:{X_test_df}")
        self.send_output(X_test_df.to_dict("list"))


# Predictor class is the most important class that could train, predict and calculate
# loss, uncertainties, threshold and p_value, so all these calculations will be
# done in function "on_received_message" in every iteration
class Trainer_Predictor(AgentMET4FOF):
    def init_parameters(self, train_size=10, model_type="withLSTM"):
        self.model = None
        self.counter = 0
        self.X_train_std = 0
        self.n_epochs = 5
        self.train_size = train_size
        self.X_train_df = pd.DataFrame()

        self.model_type = model_type
        self.path = str(Path(__file__).parent) + "/saved_trained_models/"
        self.train_loss_best = []
        self.best_epoch = None
        self.best_model_wts = []

        self.var_Xtest = 0
        self.loss = 0
        self.uncertainty_loss_der_square = 0
        self.uncertainty_loss = 0
        self.threshold = 0
        self.z_scores = 0
        self.p_values = 0

    # Function "create_dataset" receives a dataframe parameter and convert it to list
    # of tensors that pytorch model could read it as input data
    def create_dataset(self, df):
        sequences = df.astype(np.float32).to_numpy().tolist()
        dataset = [torch.tensor(s).unsqueeze(1).float() for s in sequences]
        n_seq, seq_len, n_features = torch.stack(dataset).shape
        return dataset, seq_len, n_features

    def on_received_message(self, message):
        self.log_info("trainer_Start")
        self.log_info(f"Disturbance_message:{message}")

        # stream data comes from disturbance and because disturbance generates normal
        # and abnormal data and we just want to train our model with normal data we
        # omit data that has anomalies value
        if "Anomalies" not in message["data"]:
            X_train_df_temp = pd.DataFrame(
                [message["data"]["y1"], message["data"]["y2"], message["data"]["y3"]]
            )
            X_train_df_temp = X_train_df_temp.T
            self.log_info(f"X_train_df_temp:{X_train_df_temp}")

        # Fill a X_train_df in amount of train_size by considering in every iteration
        # Receive X_train_df_temp that it's size is amount of buffer size and will be
        # added to X_train_df in amount of train size
        if self.counter < self.train_size and "Anomalies" not in message["data"]:
            self.counter += len(X_train_df_temp)

            self.X_train_df = self.X_train_df.append(
                X_train_df_temp.head(self.train_size)
            )
            self.X_train_df = self.X_train_df.reset_index(drop=True)
            self.log_info(f"X_train_df:{self.X_train_df}")

        ################################################################################
        # Filled X_train_df in amount of train_size, so start to train model with
        # X_train_df self.counter>=self.train_size means X_train_df filled in amount
        # of train_size self.model == None means our model did not train yet and is
        # empty
        if self.counter >= self.train_size and self.model is None:
            self.log_info(f"counter1:{self.counter}")
            self.X_train_std, self.model = self.Train_func(self.X_train_df)

        # self.model == None means model tarined and is ready to predict new input
        # streams raw data
        elif self.model is not None:
            self.log_info(f"counter2:{self.counter}")
            self.send_output(self.Prediction_func(message, self.model))

        else:
            self.log_info("train_model not available!!!")

    def Train_func(self, X_train_df):
        self.log_info(f"full_X_train_df:{X_train_df}")

        X_train_std = X_train_df.std(axis=0)

        # Calculate standard deviation of train data that in next steps will be puted
        # in threshold
        self.log_info(f"X_train_std:{X_train_std}")

        # Pass X_train_df to function create_dataset for generating list of tensors
        # that will be read in pytorch model as input data
        X_train_df, seq_len, n_features = self.create_dataset(X_train_df)

        self.log_info(f"seq_len:{seq_len}")
        self.log_info(f"n_features:{n_features}")

        # Initialize train model with defining pytorch model hyperparameters
        model = RecurrentAutoencoder(seq_len, n_features, 64, self.model_type)
        model = model.to(device)

        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        criterion = nn.L1Loss(reduction="sum").to(device)

        self.best_model_wts = copy.deepcopy(model.state_dict())
        self.log_info(f"state_dict:{model.state_dict()}")

        # Train model with some number of epochs and calculate loss of train data
        for epoch in range(1, self.n_epochs + 1):
            model = model.train()

            train_losses = []
            for seq_true in X_train_df:
                optimizer.zero_grad()

                seq_true = seq_true.to(device)
                seq_pred = model(seq_true)

                loss = criterion(seq_pred, seq_true)

                loss.backward()
                optimizer.step()

                train_losses.append(loss.item())

            train_loss = np.mean(train_losses)
            self.train_loss_best.append(train_loss)
            self.log_info(f"self.model.state_dict():{model.state_dict()}")

            # Put state_dict of best model(means each model that has minimum loss) in
            # best_model_wts variable. In addition, train_loss_best[0] has minimum
            # train loss value
            if train_loss < self.train_loss_best[0]:
                self.train_loss_best[0] = train_loss
                self.best_model_wts = []
                self.best_model_wts.append(model.state_dict())
                self.log_info(f"best_model_wts:{self.best_model_wts}")
                self.best_epoch = epoch

            self.log_info(
                f"Epoch:{self.model_type} {epoch}: train loss {np.round(train_loss, 6)}"
            )

        now = datetime.now()
        # Save best_model_wts in hard disk
        torch.save(
            self.best_model_wts,
            self.path
            + now.strftime("%Y-%m-%d___%H_%M_%S")
            + "_train_epoch{}.pth".format(self.best_epoch),
        )

        self.log_info(f"train_loss_best:{np.round(self.train_loss_best, 6)}")
        self.log_info(f"best_model_wts:{self.best_model_wts}")

        self.log_info("trainer_End")

        return X_train_std, model

    ####################################################################################
    def Prediction_func(self, message, model):
        X_test_df = 0
        self.log_info(f"message_test:{message}")
        self.log_info("X_test_df_begin:")

        X_test_df = pd.DataFrame(
            [message["data"]["y1"], message["data"]["y2"], message["data"]["y3"]]
        )
        X_test_df = X_test_df.T
        X_test_df.columns = ["y1", "y2", "y3"]
        self.log_info(f"X_test_df:{X_test_df}")

        X_test_torch, seq_len, n_features = self.create_dataset(X_test_df)

        # Pass every stream blocks of test data(in amount of buffer size) to model
        # for prediction
        predictions, losses = [], []
        with torch.no_grad():
            model = model.eval()
            for seq_true in X_test_torch:
                seq_true = seq_true.to(device)
                seq_pred = model(seq_true)

                predictions.append(seq_pred.cpu().numpy().flatten())

        self.log_info(f"X_test_torch_3dim: {X_test_torch}")

        # Convert 3dim test list data to pandas dataframe
        X_test_df = pd.DataFrame(np.squeeze(torch.stack(X_test_torch).numpy()))

        self.log_info(f"X_test_df: {X_test_df}")

        X_test_arr = np.array(X_test_df)
        self.log_info(f"X_test_arr:{X_test_arr}")

        # Convert 3dim list prediction data to pandas dataframe
        self.log_info(f"predictions: {predictions}")
        X_pred_df = pd.DataFrame(predictions)

        self.log_info(f"X_pred_df: {X_pred_df}")

        X_pred_arr = np.array(X_pred_df)
        self.log_info(f"X_pred_arr:{X_pred_arr}")

        ################################################################################
        # scored dataframe consisted by all calculation's results that will be showed
        # in monitor_agent_2 in final output web page
        scored = pd.DataFrame()

        # Delete scored dataframe after each iteration for preventing overflow in memory
        if not scored.empty:
            del [scored]
            gc.collect()

        self.var_Xtest = X_test_df.var(axis=0)
        self.log_info(f"var_Xtest: {self.var_Xtest}")

        # Calculate loss value that used mean square error formula for calculating of
        # loss
        self.loss = (1 / len(X_test_df.columns)) * np.sum(
            ((X_test_df - X_pred_df) ** 2), axis=0
        )
        self.log_info(f"MSE: {self.loss}")

        # Calculate derivative of loss for using in standard uncertainty
        self.uncertainty_loss_der_square = pd.DataFrame(
            ((X_test_arr - X_pred_arr) / 2) ** 2
        )
        self.log_info(
            f"uncertainty_loss_der_square: {self.uncertainty_loss_der_square}"
        )

        # Calculate standard uncertainty
        self.uncertainty_loss = np.sum(
            self.uncertainty_loss_der_square * self.var_Xtest, axis=0
        )
        self.log_info(f"uncertainty_loss: {self.uncertainty_loss}")

        # Define threshold that is calculated by mean of standard deviations of train
        # data for three sensors
        self.log_info(f"X_train_std: {self.X_train_std}")
        self.threshold = [np.mean(self.X_train_std) * 2]  # 95.4% confidence interval
        self.log_info(f"threshold: {self.threshold}")

        self.z_scores = (self.threshold - self.loss) / np.sqrt(
            self.uncertainty_loss.values
        )
        self.log_info(f"z_scores: {self.z_scores}")

        # Calculate probability of each test data
        self.p_values = stats.norm.cdf(self.z_scores)
        self.log_info(f"p_values: {self.p_values}")

        # Fill scored dataframe with all calculations that achieved above
        # Consider mean of loss,uncertainty_loss and p_values because output must
        # have one output for there different sensors
        scored["loss"] = [np.mean(self.loss)]

        scored["threshold"] = self.threshold
        self.log_info(f"threshold:{scored['threshold']}")

        scored["upper_uncertainty_loss"] = np.mean(
            [self.loss + self.uncertainty_loss.values]
        )
        scored["below_uncertainty_loss"] = np.mean(
            [self.loss - self.uncertainty_loss.values]
        )

        scored["p_values"] = np.mean([1 - self.p_values])

        # Select first time of message(stream data came from aggregator)
        scored["time"] = message["data"]["time"][0]
        self.log_info(f"scored:{scored.T}")

        scored_dict = scored.to_dict("list")
        self.log_info(f"scored_dict:{scored_dict}")

        return scored_dict


# custom_create_monitor_graph_raw_data could modify appearance of final graph that
# will be showed in output web page
def custom_create_monitor_graph_raw_data(data, sender_agent):
    """
    Parameters
    ----------
    data : dict or np.ndarray
        The data saved in the MonitorAgent's memory, for each Inputs (Agents) it is
        connected to.

    sender_agent : str
        Name of the sender agent

    **kwargs
        Custom parameters
    """

    x = pd.to_datetime(data["time"], unit="s")
    y1 = data["y1"]
    y2 = data["y2"]
    y3 = data["y3"]

    all_go = [
        go.Scatter(x=x, y=y1, mode="lines", name="Sensor1", line=dict(color="red")),
        go.Scatter(x=x, y=y2, mode="lines", name="Sensor2", line=dict(color="green")),
        go.Scatter(x=x, y=y3, mode="lines", name="Sensor3", line=dict(color="blue")),
    ]
    return all_go


# custom_create_monitor_graph_calculation could modify appearance of final graph that
# will be showed in output web page
def custom_create_monitor_graph_calculation(data, sender_agent):
    """
    Parameters
    ----------
    data : dict or np.ndarray
        The data saved in the MonitorAgent's memory, for each Inputs (Agents) it is
        connected to.

    sender_agent : str
        Name of the sender agent

    **kwargs
        Custom parameters
    """

    x = pd.to_datetime(data["time"], unit="s")
    loss = data["loss"]
    threshold = data["threshold"]
    upper_uncertainty_loss = data["upper_uncertainty_loss"]
    below_uncertainty_loss = data["below_uncertainty_loss"]
    p_values = data["p_values"]

    all_go = [
        go.Scatter(x=x, y=loss, mode="lines", name="Loss", line=dict(color="#3399FF")),
        go.Scatter(
            x=x, y=threshold, mode="lines", name="Threshold", line=dict(color="yellow")
        ),
        go.Scatter(
            x=x,
            y=upper_uncertainty_loss,
            mode="lines",
            name="Upper_uncertainty_loss",
            line=dict(color="#CCE5FF"),
        ),
        go.Scatter(
            x=x,
            y=below_uncertainty_loss,
            mode="lines",
            name="Below_uncertainty_loss",
            line=dict(color="#CCE5FF"),
        ),
        go.Scatter(
            x=x, y=p_values, mode="lines", name="p_values", line=dict(color="#FF66B2")
        ),
    ]
    return all_go


########################################################################################


def run_detection():
    # start agent network server
    agentNetwork = AgentNetwork()

    gen_agent_test1 = agentNetwork.add_agent(
        name="Sensor1", agentType=SineGeneratorAgent, log_mode=False, buffer_size=5,
    )
    gen_agent_test2 = agentNetwork.add_agent(
        name="Sensor2", agentType=SineGeneratorAgent, log_mode=False, buffer_size=5,
    )
    gen_agent_test3 = agentNetwork.add_agent(
        name="Sensor3", agentType=SineGeneratorAgent, log_mode=False, buffer_size=5,
    )

    aggregator_agent = agentNetwork.add_agent(agentType=Aggregator)
    disturbance_agent = agentNetwork.add_agent(agentType=Disturbance)
    Trainer_Predictor_agent = agentNetwork.add_agent(agentType=Trainer_Predictor)

    monitor_agent_1 = agentNetwork.add_agent(
        agentType=MonitorAgent, buffer_size=100, log_mode=False
    )
    monitor_agent_2 = agentNetwork.add_agent(
        agentType=MonitorAgent, buffer_size=300, log_mode=False
    )

    gen_agent_test1.init_agent_loop(loop_wait=0.01)

    gen_agent_test2.init_parameters(scale_amplitude=0.3)
    gen_agent_test2.init_agent_loop(loop_wait=0.01)

    gen_agent_test3.init_parameters(scale_amplitude=0.6)
    gen_agent_test3.init_agent_loop(loop_wait=0.01)

    Trainer_Predictor_agent.init_parameters(
        model_type="withLSTM"
    )  # define train_size and machine learning model types(
    # withLSTM or withoutLSTM)

    monitor_agent_1.init_parameters(
        custom_plot_function=custom_create_monitor_graph_raw_data,
    )
    monitor_agent_2.init_parameters(
        custom_plot_function=custom_create_monitor_graph_calculation,
    )

    # bind agents
    agentNetwork.bind_agents(gen_agent_test1, aggregator_agent)
    agentNetwork.bind_agents(gen_agent_test2, aggregator_agent)
    agentNetwork.bind_agents(gen_agent_test3, aggregator_agent)

    agentNetwork.bind_agents(aggregator_agent, disturbance_agent)
    agentNetwork.bind_agents(disturbance_agent, Trainer_Predictor_agent)
    agentNetwork.bind_agents(aggregator_agent, monitor_agent_1)
    agentNetwork.bind_agents(Trainer_Predictor_agent, monitor_agent_2)

    # set all agents states to "Running"
    agentNetwork.set_running_state()

    # allow for shutting down the network after execution
    return agentNetwork


if __name__ == "__main__":
    run_detection()

# Done and finish
