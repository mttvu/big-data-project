# import sys
# sys.path.append("..")
from covadem.database_code.mongo_db import MongoCovadem, MongoRWS
from covadem.database_code.meetpunten_db import MeetpuntenDB
import requests
import pandas as pd
import re


def get_waterdepth_status(waterdepth):
    """aan de hand van de waterdiepte bepalen wat de status is van een meetpunt"""
    if waterdepth < 2.80:
        return 'kritisch'
    else:
        return 'normaal'

def get_waterlevel_status(meetpunt, waterlevel):
    zones = get_zones_waterlevel(meetpunt)
    print(meetpunt)
    if zones is not None:
        for i, zone in zones.iterrows():
            p = re.compile(r'\([^)]*\)')
            label = re.sub(p, '', zone['label']).rstrip()

            if pd.isna(zone['from']):
                if waterlevel < zone['to']:
                    return label

            if pd.isna(zone['to']):
                if waterlevel > zone['from']:
                    return label

            if (waterlevel > zone['from']) and (waterlevel < zone['to']):
                return label
    return 'Geen klasse-indeling'


def get_locatie_code(meetpunt):
    try:
        sqldb = MeetpuntenDB('ruijten4', 'r6ehAdkPGJ92JUwM')
        result = sqldb.get_meetpunt(meetpunt)
        print(meetpunt)
        return result['locatieCode'][0]
    except Exception:
        print(meetpunt)


def get_zones_waterlevel(meetpunt):
    locatie_code = get_locatie_code(meetpunt)
    if locatie_code:
        result = requests.get('https://waterinfo.rws.nl/api/chart?mapType=waterhoogte&locationCode='+locatie_code+'&values=-72,0')
        limits = pd.DataFrame(result.json()['limits'])
        limits['to'] = limits['to'] / 100
        limits['from'] = limits['from'] / 100
        return limits

def get_zones_legenda():
    result = requests.get('https://waterinfo.rws.nl/api/legend?mapType=waterhoogte&user=publiek{parameters}')
    zones = pd.DataFrame(result.json()[0]['items'])
    return zones


def get_bedlevel_status(bedlevel):
    return ''



def set_waterdepth_status_meetpunten():
    """pakt de laatste update van elk meetpunt en bepaalt van die waterdiepte de status van het meetpunt
    en slaat het op in de database"""
    mongodb = MongoCovadem()
    sqldb = MeetpuntenDB('ruijten4', 'r6ehAdkPGJ92JUwM')

    last_records = mongodb.get_last_record_meetpunten()
    last_records['waterdepthStatus'] = last_records.apply(lambda x: get_waterdepth_status(x.waterDepth), axis=1)

    sqldb.upload_meetpunten(last_records)


def set_waterlevel_status_meetpunten():
    mongodb = MongoRWS()
    sqldb = MeetpuntenDB('ruijten4', 'r6ehAdkPGJ92JUwM')
    last_records = mongodb.get_last_record_meetpunten()
    print(last_records.head())
    last_records['waterlevelStatus'] = last_records.apply(lambda x: get_waterlevel_status(x.MEETPUNT_IDENTIFICATIE, x.waterLevel), axis=1)

    for i, row in last_records.iterrows():
        sqldb.update_meetpunt_waterlevel_status(row['MEETPUNT_IDENTIFICATIE'], row['waterlevelStatus'], row['waterLevel'])



def update_status(meetpunt, waterdepth):
    db = MeetpuntenDB('ruijten4', 'r6ehAdkPGJ92JUwM')
    waterdepth_status = get_waterdepth_status(waterdepth)
    db.update_meetpunt_status(meetpunt, waterdepth_status, waterdepth)


if __name__ == "__main__":
    # locatie codes van de meetpunten op halen via de api van rws
    collect_catalogus = ('https://waterwebservices.rijkswaterstaat.nl/' +
                         'METADATASERVICES_DBO/' +
                         'OphalenCatalogus/')

    request = {
        "CatalogusFilter": {
            "Eenheden": True,
            "Grootheden": True,
            "Hoedanigheden": True,
            "Compartimenten": True
        }
    }

    resp = requests.post(collect_catalogus, json=request)
    result = resp.json()

    df_locations = pd.DataFrame(result['LocatieLijst'])
    df_locations['Naam_met_streep'] = df_locations.apply(lambda x: x.Naam.title().replace(' ', '-'), axis=1)
    df_locations['locatie_code'] = df_locations.apply(lambda x: x.Naam_met_streep + '('+x.Code+')', axis=1)

    sqldb = MeetpuntenDB('ruijten4', 'r6ehAdkPGJ92JUwM')
    meetpunten = sqldb.get_meetpunten()
    meetpunten_codes = []
    for i, meetpunt in meetpunten.iterrows():
        for index, row in df_locations.iterrows():
            if row['Naam'] == meetpunt['MEETPUNT_IDENTIFICATIE']:
                code = row['locatie_code']
                result = requests.get('https://waterinfo.rws.nl/api/chart?mapType=waterhoogte&locationCode='+code+'&values=-72,0')
                if result.status_code == 200:
                    print(code)
                    sqldb.update_locatie_code(meetpunt['MEETPUNT_IDENTIFICATIE'], code)

    result = requests.get('https://waterinfo.rws.nl/api/chart?mapType=waterhoogte&locationCode=Westervoort-IJsseldijkerw(WESTV)&values=-72,0')
    limits =pd.DataFrame(result.json()['limits'])
