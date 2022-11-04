import sys, os
import datetime
from rws_data import RWSData

# Test with mongodb
sys.path.append(os.getcwd() + '/..')
from covadem.data.rws_data import RWSData
rws = RWSData()
start = datetime.datetime(2018, 1, 1)
end = datetime.datetime(2019, 12, 31)
# Meetpunten plus coordinaten
meetpunten = rws.get_coordinates_meetpunten()
print(meetpunten)
# Gebruikte meetpunt
meetpunt = meetpunten[meetpunten['name'] == 'Gennep']
# Waterdiepte van covadem per coordinate
# for naam in meetpunten['name']:
#     try:
waterdiepte = rws.get_covadem_waterdepth(start, end, 'Gennep', 0.1)
print(waterdiepte)