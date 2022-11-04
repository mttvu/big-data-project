# Pandas imports
import pandas as pd

# Numpy imports
import numpy as np

# Sklearn imports
from sklearn.model_selection import train_test_split

# Statsmodels imports
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.holtwinters import SimpleExpSmoothing
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.holtwinters import Holt

# Plot import
import plotly.express as px
import matplotlib.pyplot as plt

# Other imports
import pickle
import sys, os
import datetime
from tqdm import tqdm
import traces

class bodemVoorspelling:
    def __init__(self, point_name='Amerongen beneden', N_FORECAST=10000, DATA="rws_data.pkl"):
        # Load the pickle file
        sys.path.append(os.getcwd() + '/..')
        f = open(os.path.dirname(__file__) + "/../data/" + DATA, 'rb')
        data = pickle.load(f)

        self.point = point_name

        # Select the measuring point u want
        self.meetPuntData = data[data['MEETPUNT_IDENTIFICATIE'] == point_name]
        del data

        # Fix the missing data
        self.timeseries = pd.DataFrame(self.fixDataset(self.meetPuntData))

        # Select how many datapoints u want to predict
        self.N_FORECAST = N_FORECAST
        self.forewardTime = [(self.meetPuntData['WAARNEMINGDATUMTIJD'].max() + x*datetime.timedelta(minutes=10)) for x in range(self.N_FORECAST)]

        self.predictions = []

    def train(self):
        # Fit the prediction models
        # 8784 because of montly seasonal in 10 min timeframe (31*24*6)
        # Zig Zag to speed up training?
        model = ExponentialSmoothing(self.timeseries[1], seasonal_periods=8784, trend='add', seasonal='add', use_boxcox=True, initialization_method='estimated').fit()
        self.saveModel('Machine_learning//MODELS//' + self.point + '.pkl', model)

        # Append the results to an array
        self.predictions.append(model.forecast(self.N_FORECAST))

    def predict(self, pointName):
        model = self.loadPickleModel('Machine_learning//MODELS//' + pointName + '.pkl')
        self.predictions.append(model.forecast(self.N_FORECAST))
        self.plot()


    def plot(self):
        # Plot the data with the predictions
        fig = plt.figure()
        ax = fig.add_subplot(111)

        # Plot the Origional and interpolated data
        ax.scatter(self.meetPuntData['WAARNEMINGDATUMTIJD'], self.meetPuntData['NUMERIEKEWAARDE'], s=0.5, color='lime', label="Original Data")
        ax.plot(self.timeseries[0], self.timeseries[1], label="Intrepolated Data", alpha=0.7)

        # Facy fukin colors
        colors = ['red', 'green', 'purple']
        description = ["ExponentialSmoothing", "SimpleExpSmoothing", "Holt"]

        # Plot the predictions
        for p in range(len(self.predictions)):
            ax.plot(self.forewardTime, self.predictions[p], color=colors[p], label=description[p])

        plt.legend()
        plt.show()


    def checkAccuracy(self, testSize=0.2):
        return 0

    def fixDataset(self, df):
        # Make a timeseries object
        timeseries = traces.TimeSeries()

        # Fill the timeseries object
        for x in range(len(df['WAARNEMINGDATUMTIJD'])):
            timeseries[df['WAARNEMINGDATUMTIJD'][x]] = df['NUMERIEKEWAARDE'][x]

        # Fix the missing data
        return timeseries.sample(sampling_period=datetime.timedelta(minutes=10), start=df['WAARNEMINGDATUMTIJD'].min(), end=df['WAARNEMINGDATUMTIJD'].max(), interpolate='previous')

    def loadPickleModel(self, Path):
        with open(Path, 'rb') as f:
            return pickle.load(f)

    def saveModel(self, Path, model):
        with open(Path, 'wb') as f:
            pickle.dump(model, f)
    

if __name__ == '__main__':
    # Current code run takes ~2909.494 seconds
    buttom = bodemVoorspelling(N_FORECAST=4000)
    # buttom.train()
    buttom.predict('Amerongen beneden')
    # buttom.plot()


    


