from covadem.database_code.mongo_db import MongoCovadem
from bson.json_util import dumps
from math import radians, cos, sin, asin, sqrt
from covadem.database_code.meetpunten_db import MeetpuntenDB
from covadem.machine_learning.kritische_zone import update_status
import pprint

sql_db = MeetpuntenDB(USER="ruijten4", PASS="r6ehAdkPGJ92JUwM")
mongodb = MongoCovadem()
meetpunten = sql_db.get_meetpunten()

def within_radius(lat, lng, meetpunt_lat, meetpunt_lng, radius):
    """

    :param lat: latitude van de nieuwe data waarvan je wil weten of het binnen de meegegeven radius is van het meetpunt
    :param lng: longitude van de nieuwe data waarvan je wil weten of het binnen de meegegeven radius is van het meetpunt
    :param meetpunt_lat: latitude van het meetpunt
    :param meetpunt_lng: longitude van het meetpunt
    :param radius: de radius om het meetpunt in km
    :return: of de afstand tussen de gegeven coordinaten kleiner is dan de radius
    """
    lng, lat, meetpunt_lng, meetpunt_lat = map(radians, [lng, lat, meetpunt_lng, meetpunt_lat])

    # haversine formula
    distance_lon = meetpunt_lng - lng
    distance_lat = meetpunt_lat - lat
    a = sin(distance_lat / 2) ** 2 + cos(lat) * cos(meetpunt_lat) * sin(distance_lon / 2) ** 2
    c = 2 * asin(sqrt(a))
    r = 6371
    return (c * r) < radius


def check_update(change):
    """
    kijkt bij elk meetpunt of de nieuwe data binnen het radius zit van een meetpunt.
    Zo ja, dan wordt die geupdate in de SQL database
    :param change: nieuwe waterdiepte data in mongodb in mongodb
    """
    for i, row in meetpunten.iterrows():
        if within_radius(float(change['lat']), float(change['lng']), row['lat'], row['lng'], 0.5):
            print('yes')
            update_status(row['MEETPUNT_IDENTIFICATIE'], float(change['waterDepth']))


def listen_to_changes():
    """
    Als er een actie wordt uitgevoerd in de mongodb database, dan wordt er gecheckt of dat een insert is.
    Zo ja, dan wordt check_update aangeroepen om te controleren of er een meetpunt moet worden geupdate
    """
    change_stream = mongodb.collection.watch()
    for change in change_stream:
        if change['operationType'] == 'insert':
            check_update(change['fullDocument'])


listen_to_changes()
