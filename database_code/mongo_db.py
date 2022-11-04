from pymongo import MongoClient
import pymongo
import pandas as pd
import numpy as np


class MongoRWS:
    def __init__(self):
        client = MongoClient('localhost', 27017)
        self.collection = client.covadem.waterlevel

    def get_by_meetpunt(self, meetpunt, start, end):
        """

        :param meetpunt:
        :param start:
        :param end:
        :return: waterhoogte data van de meegegeven meetpunt binnen de start en eind datum
        """
        result = self.collection.find(
            {"MEETPUNT_IDENTIFICATIE": meetpunt,
             'time': {'$lt': end, '$gte': start}}
        )
        df = pd.DataFrame(list(result))
        df.NUMERIEKEWAARDE = df.NUMERIEKEWAARDE.astype(int)
        df = df[df['NUMERIEKEWAARDE'] != 999999999]
        df['NUMERIEKEWAARDE'] = df['NUMERIEKEWAARDE'] / 100
        return df

    def convert_string(self):
        self.collection.update_many({},
                                    [{'$set': {
                                        'time': {
                                            '$dateFromString': {
                                                'dateString': '$time',
                                                'format': '%d-%m-%Y %H:%M:%S'}}
                                    }}])

    def get_last_record_meetpunten(self):
        meetpunten = pd.read_csv('../data/meetpunten.csv')
        waterlevel_list = []
        for i, row in meetpunten.iterrows():
            result = self.collection.aggregate([
                {
                    '$sort': {
                        'time': -1
                    }
                },
                {
                    '$match': {
                        'MEETPUNT_IDENTIFICATIE': row['MEETPUNT_IDENTIFICATIE']
                    }
                },
                {
                    '$limit': 1
                }
            ])
            result = list(result)
            if result:
                waterlevel_list.append(result[0]['NUMERIEKEWAARDE'])
                print(row['name'] + ' ' + result[0]['NUMERIEKEWAARDE'] + ' ' + str(result[0]['time']))
            else:
                waterlevel_list.append(np.nan)
                print(row['name'] + 'nan')
        meetpunten['waterLevel'] = waterlevel_list
        meetpunten.waterLevel = meetpunten.waterLevel.astype(int)
        meetpunten = meetpunten[meetpunten['waterLevel'] != 999999999]
        meetpunten['waterLevel'] = meetpunten['waterLevel'] / 100
        return meetpunten

class MongoCovadem:
    def __init__(self):
        client = MongoClient('localhost', 27017)
        self.collection = client.covadem.waterdepth

    # om te kunnen filteren op radius moet dit formaat gebruikt worden voor coordinaten
    def create_location_field(self):
        self.collection.update_many({},
                                    [{'$set': {
                                        'location': {
                                            'type': 'Point',
                                            'coordinates': [
                                                {'$toDecimal': '$lng'},
                                                {'$toDecimal': '$lat'}]}}}])

    # index maken zodat de geospatial features van mongodb gebruikt kunnen worden
    def create_location_index(self):
        self.collection.create_index([('location', pymongo.GEOSPHERE)], name='_location')

    def convert_time(self):
        self.collection.update_many({},
                                    [{'$set': {
                                        'time': {
                                            '$dateFromString': {
                                                'dateString': '$time'}}
                                    }}])

    def create_time_index(self):
        self.collection.create_index([('time', pymongo.ASCENDING)], name='_time')

    def get_by_radius(self, start, end, lat, lng, radius):
        """ 
            get covadem meetpunten binnen een radius van de aangegeven coordinaten
        """
        query = {
            'time': {'$lt': end, '$gte': start},
            'location': {
                '$geoWithin': {
                    # 6371 is de radius van de aarde
                    '$centerSphere': [[lng, lat], radius / 6371]
                }
            }}
        result = self.collection.find(query)
        df = pd.DataFrame(list(result))
        # string to correct data type
        df.lat = df.lat.astype(float)
        df.lng = df.lng.astype(float)
        df.waterDepth = df.waterDepth.astype(float)

        # return necessary columns only
        return df[['time', 'lat', 'lng', 'waterDepth']]

    def get_last_record_meetpunten(self):
        meetpunten = pd.read_csv('../data/meetpunten.csv')
        radius = 1
        waterdepth_list = []
        last_updated_list = []
        for i, row in meetpunten.iterrows():
            lng = row['lng']
            lat = row['lat']
            result = self.collection.aggregate([
                {
                    '$sort': {
                        'time': -1
                    }
                },
                {
                    '$match': {
                        'location': {
                            '$geoWithin': {
                                '$centerSphere': [[lng, lat], radius / 6371]

                            }
                        }
                    }
                },
                {
                    '$limit': 1
                }
            ])
            result = list(result)
            if result:
                waterdepth_list.append(result[0]['waterDepth'])
                last_updated_list.append(result[0]['time'])
                print(row['name'] + ' ' + result[0]['waterDepth'] + ' ' + str(result[0]['time']))
            else:
                waterdepth_list.append(np.nan)
                last_updated_list.append(np.nan)
                print(row['name'] + 'nan')

        meetpunten['waterDepth'] = waterdepth_list
        meetpunten['lastUpdated'] = last_updated_list
        meetpunten.lat = meetpunten.lat.astype(float)
        meetpunten.lng = meetpunten.lng.astype(float)
        meetpunten.waterDepth = meetpunten.waterDepth.astype(float)
        return meetpunten


if __name__ == "__main__":
    db = MongoRWS()
    db.get_last_record_meetpunten()