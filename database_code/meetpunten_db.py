import pandas as pd
import requests
import mysql.connector
from sqlalchemy import create_engine, text
import time
from datetime import datetime


class MeetpuntenDB:
    # Constructor makes a connection to the database
    def __init__(self, USER, PASS, HOST="oege.ie.hva.nl", PORT="3306", DATABASE="zruijten4"):
        self.engine = create_engine('mysql+mysqlconnector://'+USER+':'+PASS+'@'+HOST+':'+PORT+'/'+DATABASE, echo=False,
                                    connect_args={'auth_plugin': 'mysql_native_password'})

    # Make a plain text SQL call (VUNERABLE TO INJECTION (probably))
    def sqlCall(self, query):
        assert isinstance(query, str)
        with self.engine.connect() as connection:
            result = connection.execute(query)

        return result
    
    def get_meetpunt(self, meetpunt):
        with self.engine.connect() as connection:
            s = text("""
                select *
                from meetpunten
                where MEETPUNT_IDENTIFICATIE = :meetpunt;""")
            result = connection.execute(s,meetpunt=meetpunt)
            meetpunt = pd.DataFrame(result)
            meetpunt.columns = result.keys()
            return meetpunt


    def get_meetpunten(self):
        result = self.sqlCall("SELECT * FROM meetpunten")
        meetpunten_df = pd.DataFrame(result)
        meetpunten_df.columns = result.keys()
        return meetpunten_df

    def update_meetpunt_waterdepth_status(self, meetpunt, status, waterdepth):
        with self.engine.connect() as connection:
            s = text("""
                update meetpunten
                set waterdepthStatus = :status, waterDepth = :waterdepth, lastUpdated = :last_updated
                where MEETPUNT_IDENTIFICATIE = :meetpunt;""")
            connection.execute(s,
                               status=status, meetpunt=meetpunt, waterdepth=waterdepth, last_updated=datetime.now())

    def update_meetpunt_waterlevel_status(self, meetpunt, status, waterlevel):
        with self.engine.connect() as connection:
            s = text("""
                update meetpunten
                set waterlevelStatus = :status, waterLevel = :waterlevel, lastUpdated = :last_updated
                where MEETPUNT_IDENTIFICATIE = :meetpunt;""")
            connection.execute(s,
                               status=status, meetpunt=meetpunt, waterlevel=waterlevel, last_updated=datetime.now())

    def update_locatie_code(self, meetpunt, code):
        with self.engine.connect() as connection:
            s = text("""
                update meetpunten
                set locatieCode = :code
                where MEETPUNT_IDENTIFICATIE = :meetpunt;""")
            connection.execute(s,meetpunt=meetpunt, code=code)

    def upload_meetpunten(self, df):
        df.to_sql(name='meetpunten', con=self.engine, if_exists='replace', index=False)

    # MAKE A PROCEDURE CALL
    def procedureCall(self, procedure, **list):
        with self.engine.connect() as connection:
            return connection.execute(procedure, **list)

    # Changes the sql result to a pandas data frame (Should be worked on more probably)
    def SQLtoPandasDF(self, sqlreturn):
        return pd.DataFrame(sqlreturn)


if __name__ == "__main__":
    data = MeetpuntenDB(USER="ruijten4", PASS="UeSjK97IKd6aLU")

    result = data.sqlCall("SELECT * FROM covadem_nl_rivieren WHERE ukc = 1.44")
    columns = result.keys()

    df = data.SQLtoPandasDF(result)
    df.columns = columns