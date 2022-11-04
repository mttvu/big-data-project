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



# RREEEEEEEEEEEEEEEEEEEEEE
from pandas import DataFrame
from pandas import Series
from pandas import concat
from pandas import read_csv
# from pandas import datetime
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler
#from keras.models import Sequential
#from keras.layers import Dense
#from keras.layers import LSTM
from math import sqrt
from matplotlib import pyplot
import numpy
 

# Own imports
# sys.path.append(os.getcwd() + '/..')
# from covadem.data.rws_data import RWSData

# Algoritm performs per 10 mins, but very slow.

class TimeSeriesPrediction:
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
            self.data = self.data.resample(newTimeDelta).mean()
            self.data.fillna(method='ffill')
        else:
            self.data = self.data.resample(StartingTimeDelta, origin='start').mean()
            self.data.fillna(method='ffill')
        # There is still nan data on missing values
        # print(self.data["Data"].isnull().sum())

        self.timcol, self.datacol = timeCol, dataCol

        self.data.interpolate('linear', inplace=True)

        # Select how many datapoints u want to predict
        self.N_FORECAST = N_FORECAST

        # Can be done with pd.Daterange
        self.ForewardTimedata = pd.date_range(start=self.data.index.max(), end=self.data.index.max() + ((self.N_FORECAST-1)*newTimeDelta), freq=newTimeDelta)

        self.predictions = []
    
    # frame a sequence as a supervised learning problem
    def timeseries_to_supervised(self, data, lag=1):
        df = DataFrame(data)
        columns = [df.shift(i) for i in range(1, lag+1)]
        columns.append(df)
        df = concat(columns, axis=1)
        df.fillna(0, inplace=True)
        return df
    
    # create a differenced series
    def difference(self, dataset, interval=1):
        diff = list()
        for i in range(interval, len(dataset)):
            value = dataset[i] - dataset[i - interval]
            diff.append(value)
        return Series(diff)
    
    # invert differenced value
    def inverse_difference(self, history, yhat, interval=1):
        return yhat + history[-interval]
    
    # scale train and test data to [-1, 1]
    def scale(self, train, test):
        # fit scaler
        scaler = MinMaxScaler(feature_range=(-1, 1))
        scaler = scaler.fit(train)
        # transform train
        train = train.reshape(train.shape[0], train.shape[1])
        train_scaled = scaler.transform(train)
        # transform test
        test = test.reshape(test.shape[0], test.shape[1])
        test_scaled = scaler.transform(test)
        return scaler, train_scaled, test_scaled
    
    # inverse scaling for a forecasted value
    def invert_scale(self, scaler, X, value):
        new_row = [x for x in X] + [value]
        array = numpy.array(new_row)
        array = array.reshape(1, len(array))
        inverted = scaler.inverse_transform(array)
        return inverted[0, -1]
    
    # fit an LSTM network to training data
    def fit_lstm(self, train, batch_size, nb_epoch, neurons):
        X, y = train[:, 0:-1], train[:, 1]
        X = X.reshape(X.shape[0], 1, X.shape[1])
        model = Sequential()
        model.add(LSTM(neurons, batch_input_shape=(batch_size, X.shape[1], X.shape[2]), stateful=True))
        model.add(Dense(1))
        model.compile(loss='mean_squared_error', optimizer='adam')
        for i in tqdm(range(nb_epoch)):
            model.fit(X, y, epochs=1, batch_size=batch_size, verbose=0, shuffle=False)
            model.reset_states()
        return model
    
    # make a one-step forecast
    def forecast_lstm(self, model, batch_size, X):
        X = X.reshape(1, 1, len(X))
        yhat = model.predict(X, batch_size=batch_size)
        return yhat[0,0]
    
    def trainNN(self):
        # load dataset
        series = self.data
        
        # transform data to be stationary
        raw_values = series['Data']
        print(raw_values.shape)
        # diff_values = self.difference(raw_values, 1)
        
        # transform data to be supervised learning
        supervised = self.timeseries_to_supervised(raw_values, 1)
        supervised_values = supervised.values
        print(supervised_values.shape)
        
        # split data into train and test-sets
        train, test = supervised_values[0:int(len(supervised_values)/2)], supervised_values[int(len(supervised_values)/2):]
        print(train.shape, test.shape)
        
        # transform the scale of the data
        scaler, train_scaled, test_scaled = self.scale(train, test)
        print(train_scaled.shape, test_scaled.shape)
        complete = np.concatenate((train_scaled, test_scaled))
        print(complete.shape)

        # fit the model
        lstm_model = self.fit_lstm(supervised_values, 1, 1000, 1)
        # self.saveModel("machine_learning\\MODELS\\tensorFL.pkl", lstm_model)
        # print(self.forecast_lstm(lstm_model, 1, complete[-1, 0:-1]))
        
        foreCast = list()
        for i in tqdm(range(self.N_FORECAST)):
            # make one-step forecast
            if i == 0:
                X = supervised_values[-1, 0:-1]
                olX = [X,]
            else:
                X = np.array([yhat,])

            # yhat = lstm_model.predict(X)
            yhat = self.forecast_lstm(lstm_model, 1, X)
            # invert scaling
            # yhat = self.invert_scale(scaler, X, yhat)
            # # invert differencing
            # yhat = self.inverse_difference(raw_values, yhat, 1)
            # store forecast
            foreCast.append(yhat)

        print(foreCast)
        self.predictions.append(foreCast)
        return pd.DataFrame({self.timcol: self.ForewardTimedata, self.datacol: foreCast})
        


    def train(self, n_harmonics=25):
        # Fit the prediction models
        # result = seasonal_decompose(self.data)
        # print(result.seasonal)
        # result.plot()
        # plt.show()
        # self.sesper = seasonal_periods
        # seasonal_periods=seasonal_periods, seasonal='multiplicative', use_boxcox=True, optimized=True
        self.model = ExponentialSmoothing(self.data['Data'], seasonal_periods=30, seasonal='additive').fit(use_boxcox=True, optimized=True)
        # self.sim = self.model.simulate(8, repetitions=100, error='mul')

        # x = np.array(self.data['Data'])
        # n = x.size
        # n_harm = n_harmonics            # number of harmonics in model
        # t = np.arange(0, n)
        # p = np.polyfit(t, x, 1)         # find linear trend in x
        # x_notrend = x - p[0] * t        # detrended x
        # x_freqdom = fft.fft(x_notrend)  # detrended x in frequency domain
        # f = fft.fftfreq(n)              # frequencies
        # indexes = list(range(n))
        # # sort indexes by frequency, lower -> higher
        # indexes.sort(key = lambda i: np.absolute(f[i]))
    
        # t = np.arange(0, n + self.N_FORECAST)
        # restored_sig = np.zeros(t.size)
        # for i in indexes[:1 + n_harm * 2]:
        #     ampli = np.absolute(x_freqdom[i]) / n   # amplitude
        #     phase = np.angle(x_freqdom[i])          # phase
        #     restored_sig += ampli * np.cos(2 * np.pi * f[i] * t + phase)
        # foreCast = (restored_sig + p[0] * t)[-self.N_FORECAST:]



        # Append the results to an array
        foreCast = self.model.forecast(self.N_FORECAST)
        print(self.model.forecast(self.N_FORECAST))
        self.predictions.append(foreCast)
        return pd.DataFrame({self.timcol: self.ForewardTimedata, self.datacol: foreCast})

    def predict(self, pointName):
        model = self.loadPickleModel('Machine_learning//MODELS//' + pointName + '.pkl')
        self.predictions.append(self.model.predict(self.N_FORECAST))
        self.plot()

    def plot(self):
        # Todo make train test plottable
        # Plot the data with the predictions
        fig = plt.figure()
        ax = fig.add_subplot(111)

        # Plot the Origional and interpolated data
        ax.plot(self.data.index, self.data['Data'], label="Data")

        # Plot the predictions
        for p in range(len(self.predictions)):
            ax.plot(self.ForewardTimedata, self.predictions[p], label="Model " + str(p))
            # self.simulations.plot(ax=ax, style='-', alpha=0.05, color='grey', legend=False)

        plt.legend()
        plt.show()

    def checkAccuracy(self, testSize=0.2):
        self.trainx, self.testx = train_test_split(self.data, test_size=testSize, shuffle=False)

        fig = plt.figure()
        ax = fig.add_subplot(111)

        # Plot the Origional and interpolated data
        ax.plot(self.trainx.index, self.trainx['Data'], label="x_train")
        ax.plot(self.testx.index, self.testx['Data'], label="x_test")

        # seasonal_periods=self.sesper, seasonal='multiplicative'
        prediction = ExponentialSmoothing(self.trainx['Data'], ).fit(use_boxcox=True, optimized=True).forecast(int(len(self.testx)))

        # prediction = ARIMA(self.trainx['Data'], order=(2,1,3), seasonal_order=(0,0,0,30)).fit().forecast(int(len(self.testx)))
        ax.plot(prediction.index, prediction, label="Predicted")

        plt.legend()
        fig.show()
        
    def fixDataset(self, df, timeDelta=datetime.timedelta(minutes=10)):
        # Make a timeseries object
        timeseries = traces.TimeSeries()

        # Fill the timeseries object
        for x in range(len(df['Time'])):
            timeseries[df['Time'][x]] = df['Time'][x]

        # Fix the missing data
        return timeseries.sample(sampling_period=timeDelta, start=df['Time'].min(), end=df['Time'].max(), interpolate='previous')

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

    # Gebruikte meetpunt
    meetpunt = meetpunten[meetpunten['name'] == 'Amerongen boven']

    # Waterdiepte van covadem per coordinate
    # for naam in meetpunten['name']:
    #     try:
    # waterdiepte = rws.get_covadem_waterdepth(start, end, 'Amerongen boven', 0.1)
    waterlevel_df = rws.get_waterlevel(start, end, 'Amerongen boven')

    predict = TimeSeriesPrediction(DATA=waterlevel_df, N_FORECAST=(6), StartingTimeDelta=datetime.timedelta(minutes=10), newTimeDelta=datetime.timedelta(minutes=10), timeCol='time', dataCol='NUMERIEKEWAARDE')
    predict.trainNN()
    # predict.train(n_harmonics=15)
    # predict.checkAccuracy(testSize=0.1)
    predict.plot()
        # except:
        #     continue

    # Get rws waterlevel
    # waterlevel = rws.get_waterlevel(start, end, meetpunt)
    # print(waterlevel)




''' Legacy comments
# Zig Zag to speed up training? Done with datetime freq change but look into.
        # self.model = ExponentialSmoothing(self.data['Data'], seasonal_periods=seasonal_periods, trend='add', seasonal='add').fit()
        # print(seasonal_decompose(self.data)._seasonal)
        # auto_arima(self.data, seasonal=True, trace=True, parallel=True, stationary=False, n_jobss=-1, m=2, stepwise=True).summary()
        # self.model = ARIMA(self.data['Data'], order=(1,1,1), seasonal_order=(2,0,2,2)).fit()

        # # # ! dublicate data check volgende 3 lines
    # for entrie in data['MEETPUNT_IDENTIFICATIE'].unique():
    #     bier = data[data['MEETPUNT_IDENTIFICATIE'] == entrie]
    #     print(entrie, len(bier['WAARNEMINGDATUMTIJD'].unique()), "/", len(bier['WAARNEMINGDATUMTIJD']))

    # for x in a:
    #     predict = TimeSeriesPrediction(DATA=data[data['MEETPUNT_IDENTIFICATIE'] == x], N_FORECAST=(30), StartingTimeDelta=datetime.timedelta(minutes=10), newTimeDelta=datetime.timedelta(days=1))
    #     predict.train(seasonal_periods=(30))
    #     predict.checkAccuracy(testSize=0.1)
    #     predict.plot()


 # Suppress future warnings
    # warnings.simplefilter(action='ignore', category=FutureWarning)

    # Get the data and construct the object
    sys.path.append(os.getcwd() + '/..')
    data = pickle.load(open(os.path.dirname(__file__) + "/../data/" + "rws_data.pkl", 'rb'))

    a = ['Doorslag', 'Amerongen boven']
'''





    


