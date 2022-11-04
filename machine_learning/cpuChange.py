# Pandas imports
import pandas as pd

# Other imports
import pickle
from tqdm import tqdm

# Torch imports
import torch

# Numpy imports
import numpy as np

# System imports
import sys, os

if __name__ == "__main__":
    sys.path.append(os.getcwd() + '/..')
    from covadem.data.rws_data import RWSData
    from covadem.machine_learning.SnellePyTorch import LSTM, thing

    rws = RWSData()

    # Meetpunten plus coordinaten
    meetpunten = rws.get_coordinates_meetpunten()

    # for x in tqdm(range(len(meetpunten['name'])), desc='WaterDepth'):
    #     try:
    #         name = "machine_learning//MODELS//WaterDepth//" + str(meetpunten['name'][x]) + "_waterDiepte_V1.pkl"
    #         with open(name, 'rb') as f:
    #             model = pickle.load(f)
    #         torch.save(model.state_dict(), "machine_learning//MODELS//WaterDepth//" + str(meetpunten['name'][x]) + "_waterDiepte_V1.pt")
    #     except Exception as e:
    #         print(e)
    #         continue

    # for x in tqdm(range(len(meetpunten['name'])), desc='BodelLigging'):
    #     try:
    #         name = "machine_learning//MODELS//BodelLigging//" + str(meetpunten['name'][x]) + "_bodemligging_V1.pkl"
    #         with open(name, 'rb') as f:
    #             model = pickle.load(f)
    #         torch.save(model.state_dict(), "machine_learning//MODELS//BodelLigging//" + str(meetpunten['name'][x]) + "_bodemligging_V1.pt")
    #     except Exception as e:
    #         print(e)
    #         continue

    # for x in tqdm(range(len(meetpunten['name'])), desc='WaterLevel'):
    #     try:
    #         name = "machine_learning//MODELS//WaterLevel//" + str(meetpunten['name'][x]) + "_waterLevel_V2.pkl"
    #         with open(name, 'rb') as f:
    #             model = pickle.load(f)
    #         torch.save(model.state_dict(), "machine_learning//MODELS//WaterLevel//" + str(meetpunten['name'][x]) + "_waterLevel_V2.pt")
    #     except Exception as e:
    #         print(e)
    #         continue

    a = (torch.load("machine_learning//MODELS//WaterLevel//" + str("Amerongen beneden") + "_waterLevel_V2.pt"))
    model = LSTM('cuda:0')
    model.load_state_dict(a)
    print(model.eval())

       

