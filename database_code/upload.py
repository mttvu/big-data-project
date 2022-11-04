import pandas as pd
import requests
import mysql.connector
from sqlalchemy import types, create_engine

USER = ""
PASS = ""
HOST = "oege.ie.hva.nl"
PORT = "3306"
DATABASE = "zruijten4"
FILE = "D:\HVA\JAAR 3\Big data project\Data\waterdepth-nl-area2-2018-2019-20200723.csv"

engine = create_engine('mysql+mysqlconnector://'+USER+':'+PASS+'@'+HOST+':'+PORT+'/'+DATABASE, echo=False)


for chunks in pd.read_csv(FILE, chunksize=100000):
    chunks.to_sql(name='covadem_nl_rivieren', con=engine, if_exists='append', index=False)

print(engine)