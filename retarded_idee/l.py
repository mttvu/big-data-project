# Pandas imports
import pandas as pd

# Numpy imports
import numpy as np
from numpy import fft

# Sklearn imports
from sklearn.model_selection import train_test_split

# Statsmodels imports
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.holtwinters import SimpleExpSmoothing
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.holtwinters import Holt
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.regime_switching.markov_autoregression import MarkovAutoregression
from statsmodels.tsa.regime_switching.markov_regression import MarkovRegression

# Pmdarima imports
#from pmdarima import auto_arima

# Plot import
import plotly.express as px
import matplotlib.pyplot as plt

# Other imports
import pickle
import datetime
import sys, os
from tqdm import tqdm
import traces
import warnings

# Torch imports
import torch
import torch.nn as nn

# REEEEEEEEEEEEEEEEEEEEEEEEE
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler


class LSTM(nn.Module):
    def __init__(self, device, input_size=1, hidden_layer_size=100, output_size=1):
        super().__init__()
        self.hidden_layer_size = hidden_layer_size

        self.lstm = nn.LSTM(input_size, hidden_layer_size).to(device)

        self.linear = nn.Linear(hidden_layer_size, output_size).to(device)

        self.hidden_cell = (torch.zeros(1,1,self.hidden_layer_size).to(device),
                            torch.zeros(1,1,self.hidden_layer_size).to(device))

    def forward(self, input_seq):
        lstm_out, self.hidden_cell = self.lstm(input_seq.view(len(input_seq) ,1, -1), self.hidden_cell)
        predictions = self.linear(lstm_out.view(len(input_seq), -1))
        return predictions[-1]

class thing:
    def __init__(self, DATA, N_FORECAST=10000, StartingTimeDelta=datetime.timedelta(minutes=10), newTimeDelta=datetime.timedelta(minutes=60), timeCol='time', dataCol='waterDepth'):
        if timeCol == 'index':
            DATA['time'] = DATA.index
            timeCol = 'time'

        # Change the collum names
        self.data = DATA[[timeCol, dataCol]].rename(columns={timeCol: 'Time', dataCol: 'Data'})
        self.data['Time'] = pd.to_datetime(self.data['Time'], format="%d-%m-%Y %H:%M:%S")

        self.data.index = self.data['Time']
        self.data = self.data[['Data']]

        # Change the time frequenty to speed up the training proces
        if StartingTimeDelta != newTimeDelta:
            self.data = self.data.resample(newTimeDelta, origin='start').mean()
            self.data.fillna(method='ffill')
        else:
            self.data = self.data.resample(StartingTimeDelta, origin='start').mean()
            self.data.fillna(method='ffill')

        self.timcol, self.datacol = timeCol, dataCol

        self.data.interpolate('linear', inplace=True)

        # Select how many datapoints u want to predict
        self.N_FORECAST = N_FORECAST

        # Can be done with pd.Daterange
        self.ForewardTimedata = pd.date_range(start=self.data.index.max(), end=self.data.index.max() + ((self.N_FORECAST-1)*newTimeDelta), freq=newTimeDelta)

        self.device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
        # print("Using", self.device)

        # Normalize the data for the model
        self.scaler = MinMaxScaler(feature_range=(-1, 1))
        self.data_normalized = self.scaler.fit_transform(np.array(self.data).reshape(-1, 1))
        self.data_normalized = torch.FloatTensor(self.data_normalized).view(-1).to(self.device)

        # X Y generation
        self.train_window = 30
        self.train_inout_seq = self.create_inout_sequences(self.data_normalized, self.train_window)

        self.predictions = []

    def create_inout_sequences(self, input_data, tw):
        inout_seq = []
        L = len(input_data)
        for i in range(L-tw):
            train_seq = input_data[i:i+tw]
            train_label = input_data[i+tw:i+tw+1]
            inout_seq.append((train_seq ,train_label))
        return inout_seq

    def train(self, epochs=150, model=None):
        if model == None:
            model = LSTM(device=self.device, hidden_layer_size=100).to(self.device)

        loss_function = nn.MSELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

        for i in tqdm(range(epochs)):
            for seq, labels in self.train_inout_seq:
                optimizer.zero_grad()
                model.hidden_cell = (torch.zeros(1, 1, model.hidden_layer_size).to(self.device),
                                        torch.zeros(1, 1, model.hidden_layer_size).to(self.device))

                y_pred = model(seq)

                single_loss = loss_function(y_pred, labels)
                single_loss.backward()
                optimizer.step()

            # if i%25 == 1:
            #     print(f'epoch: {i:3} loss: {single_loss.item():10.8f}')

        # print(f'epoch: {i:3} loss: {single_loss.item():10.10f}')

        return model

    def predict(self, model):
        fut_pred = self.N_FORECAST

        test_inputs = self.data_normalized[-self.train_window:].tolist()

        model.eval()

        for i in range(fut_pred):
            seq = torch.cuda.FloatTensor(test_inputs[-self.train_window:])
            with torch.no_grad():
                model.hidden = (torch.zeros(1, 1, model.hidden_layer_size).to(self.device),
                                torch.zeros(1, 1, model.hidden_layer_size).to(self.device))
                test_inputs.append(model(seq).item())

        self.foreCast = self.scaler.inverse_transform(np.array(test_inputs[self.train_window:] ).reshape(-1, 1))
        self.foreCast = self.foreCast.reshape(-1)

        return pd.DataFrame({self.timcol: self.ForewardTimedata, self.datacol: self.foreCast})

    def plot(self):
        fig = plt.figure()
        ax = fig.add_subplot(111)

        # Plot the Origional and interpolated data
        ax.plot(self.data.index, self.data['Data'], label="Data")

        # Plot the predictions
        ax.plot(self.ForewardTimedata, self.foreCast, label="LSTM")

        plt.legend()
        plt.show()

    def loadPickleModel(self, Path):
        with open(Path, 'rb') as f:
            return pickle.load(f)


    def saveModel(self, Path, model):
        with open(Path, 'wb') as f:
            pickle.dump(model, f)

if __name__ == '__main__':
    # Test with mongodb
    sys.path.append(os.getcwd() + '/..')
    from covadem.data.rws_data import RWSData

    rws = RWSData()
    start = datetime.datetime(2018, 1, 1)
    end = datetime.datetime(2019, 12, 31)

    # Meetpunten plus coordinaten
    meetpunten = rws.get_coordinates_meetpunten()
    print(meetpunten['name'])

    # Gebruikte meetpunt
    n = 'Amerongen boven'
    meetpunt = meetpunten[meetpunten['name'] == 'Amerongen boven']
    # waterlevel_df = rws.get_waterlevel(start, end, n)

    #! Joining all of them in one df might be bad because of continuaty of timeseries stuff
    # for naam in (range(len(meetpunten['name']))):
    #     try:
    #         if naam == 0:
    #             waterlevel_df = rws.get_waterlevel(start, end, meetpunten['name'][naam])
    #         else:
    #             waterlevel_df = pd.concat([waterlevel_df, rws.get_waterlevel(start, end, meetpunten['name'][naam])])
    #     except:
    #         continue
    # print(waterlevel_df.shape)

    # waterlevel_df = rws.get_waterlevel(start, end, meetpunten['name'][naam])


    # waterlevel_df = rws.get_waterlevel(start, end, 'Amerongen boven')
    # predicting = thing(DATA=waterlevel_df, N_FORECAST=(300), StartingTimeDelta=datetime.timedelta(minutes=10), newTimeDelta=datetime.timedelta(days=1), timeCol='time', dataCol='NUMERIEKEWAARDE')
    # model = predicting.train(epochs=750)
    # predicting.saveModel("machine_learning//MODELS//Waterlevel_test_comb_.pkl", model)
    # predicting.predict(model)
    # predicting.plot()

    #! batch train all points
    # for naam in tqdm(range(len(meetpunten['name'])), desc='All points'):
    #     try:
    #         if naam == 0:
    #             waterlevel_df = rws.get_waterlevel(start, end, meetpunten['name'][naam])
    #             predicting = thing(DATA=waterlevel_df, N_FORECAST=(300), StartingTimeDelta=datetime.timedelta(minutes=10), newTimeDelta=datetime.timedelta(days=1), timeCol='time', dataCol='NUMERIEKEWAARDE')
    #             model = predicting.train(epochs=500)
    #             predicting.saveModel("machine_learning//MODELS//Waterlevel_general_waterLevel_10min_.pkl", model)
    #         else:
    #             waterlevel_df = rws.get_waterlevel(start, end, meetpunten['name'][naam])
    #             predicting = thing(DATA=waterlevel_df, N_FORECAST=(300), StartingTimeDelta=datetime.timedelta(minutes=10), newTimeDelta=datetime.timedelta(days=1), timeCol='time', dataCol='NUMERIEKEWAARDE')
    #             model = predicting.train(epochs=500, model=model)
    #             predicting.saveModel("machine_learning//MODELS//Waterlevel_general_waterLevel_10min_.pkl", model)
    #     except:
    #         continue

    #! Test the points
    for naam in tqdm(range(len(meetpunten['name'])), desc='All points'):
        try:
            waterlevel_df = rws.get_waterlevel(start, end,  meetpunten['name'][naam])
            predicting = thing(DATA=waterlevel_df, N_FORECAST=(300), StartingTimeDelta=datetime.timedelta(minutes=10), newTimeDelta=datetime.timedelta(days=1), timeCol='time', dataCol='NUMERIEKEWAARDE')
            model = predicting.train(epochs=300)
            predicting.saveModel("machine_learning//MODELS//waterLevel//" + str(waterlevel_df['MEETPUNT_IDENTIFICATIE'][0]) + "_waterLevel_V2.pkl", model)
        except Exception as e: 
            print(e)
            continue

    predicting.predict(predicting.loadPickleModel("machine_learning//MODELS//waterLevel//" + str(waterlevel_df['MEETPUNT_IDENTIFICATIE'][0]) + "_waterLevel_V2.pkl"))
    predicting.plot()

    
    
    
    
 
