import pandas as pd
import numpy as np
import pickle
import sys, os

f = open(os.path.dirname(__file__) + '\\rws_data.pkl', 'rb')
df = pickle.load(f)


for x in df:
    print(x)
# print((df['MEETPUNT_IDENTIFICATIE'].unique()))