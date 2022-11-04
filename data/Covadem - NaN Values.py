# Import standard modules. 
import sys, os
import os.path
import pandas as pd
import datetime
import numpy as np
from numpy import isnan

# Geef de juiste systeempad op. 
# Dit moet ik doen zodat Visual Studio code de classes en functies uit andere bestanden goed kan ophalen. 
sys.path.append(r"C:\Users\N_ADi\Desktop\covadem\covadem\covadem")

# Import modules from other files.
from database_code.mongodb_loading import MongoCovadem, MongoRWS
from data.rws_data import RWSData

# Import modules for calculating the NaN values. 
from fancyimpute import IterativeImputer
from sklearn.impute import SimpleImputer
import sklearn.impute
from sklearn import impute
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.pipeline import Pipeline

# Import classifiers.
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer

"""Het volgende stuk code geeft weer hoe ik een DataFrame aanmaak en invul, gebaseerd op de data die uit MongoDB opgehaald wordt. 
Ik heb een specifieke start en eindtijd gedefinieerd. De aangegeven start en eind tijd zijn zo gekozen zodat ik over de 'complete' data van 2018 en 2019 beschik.
Ik heb voor het voorbeeld Vuren gekozen als meetpunt. Dit meetpunt heeft namelijk (vergeleken met de rest) redelijk veel data.
De radius is 0.5, dit is dus de 'straal' om het desbetreffende meetpunt heen, 0.5 wordt in kilometers uitgedrukt.
Ik heb de time kolom verwijderd, omdat deze verder geen toegevoegde waarde heeft aan hetgeen ik wil bereiken."""

# Create a variable and call RWSData().
data = RWSData()
# Define a start and end date. 
start = datetime.datetime(2018, 1, 1)
end = datetime.datetime(2019, 12, 31)
# Create a df, define parameters, pick 'Vuren', pick radius. Parameters may differ.  
df = data.get_covadem_waterdepth(start, end,'Vuren', 0.5)
# Drop the time column, as this column isn't necessary for further process of the code.
df = df.drop('time', 1)
# Print out the dtypes of the present columns. 
print(df.dtypes)

"""Het volgende stuk code geeft aan hoeveel missing entries er zijn (ingevulde waarden die ontbreken), dit zijn de NaN-values.
Dit doet het stuk code door het aantal missing entries in een heel getal te weergeven. Vervolgens wordt er een percentage getoond
van het aantal aan data (waterdiepte) die ontbreekt. Dit heb ik gedaan zodat ik een inzage heb in hoeveel data er per meetpunt ontbreekt. 
Op die manier kan ik controleren of er voldoende data aanwezig is om mijn model op te trainen. Een for loop is gebruikt voor bovengenoemde."""
 
for i in range (len(df.columns)):
     # Sum the NaN-values of the columns in the DataFrame.
     missing_data = df[df.columns[i]].isna().sum()
     # Define a calculation for the percentage.
     perc = missing_data / len(df) * 100
     # Print the amount of missing entries and percentage.
     print('>%d, Missing entries, (no values): %d, %.2f procent of the specific measure point data is missing.' % (i, missing_data, perc))

"""Het volgende stuk code verandert de datatypes van de kolommen naar datatypes die gebruikt kunnen worden voor het
imputeren van de NaN-values. De datatypes moeten ofwel integers ofwel floats zijn. Imputeren kan niet gebeuren op datatypes
zoals bijvoorbeeld datetime et cetera. Ik wilde eerst de kolomtype time converteren naar integers, echter heeft deze kolom
geen relevante rol en was het niet nodig."""

# Make the time column numerical. 
# df['time'] = pd.to_numeric(df['time'], errors='coerce')
df['y'] = pd.to_numeric(df['y'], errors='coerce')

"""Het volgende stuk code geeft weer hoe ik de NaN-values opvul. Ik heb gekozen voor de SimpleImputer module, omdat deze meerdere toepasbare manieren heeft van
opvullen. Ook kan je aangeven welk type values je wil opvullen.Ik heb als opvul manier voor de gemiddelde gekozen. 
De SimpleImputer pakt dus alle aanwezige data aan waterdieptes, pakt daar het gemiddelde uit en vult de NaN-values op met gemiddelden. 
Omdat er té veel data ontbreekt, zullen dit geen correcte getallen zijn. Dit heb ik daarom ook alleen gedaan om te testen."""

# Call the SimpleImputer module, pick what values should be replaced and pick the strategy. 
imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
# SimpleImputer module should perform on the y (waterdepth) column. 
df[['y']] = imputer.fit_transform(df[['y']])

"""Ik heb een tweede benadering onderzocht, namelijk het bouwen van een pipeline die de datatypes converteert én gelijk
imputeert. Deze pipeline is in staat om zowel numerieke variabelen als categorische variabelen te transformeren."""

# Not needed. 
# Create a pipeline to transform the datatype. 
# Create one for numerical values and one for categorical values.
# numeric_transformer = Pipeline(steps=[
#     ('imputer', SimpleImputer(strategy='median')),
#     ('scaler', StandardScaler())])
# categorical_transformer = Pipeline(steps=[
#     ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
#     ('onehot', OneHotEncoder(handle_unknown='ignore'))])
# numeric_features = df.select_dtypes(include=['int64', 'float64']).columns
# # categorical_features = df.select_dtypes(include=['object']).drop(['time'], axis=0).columns
# preprocessor = ColumnTransformer(
#     transformers=[
#         ('num', numeric_transformer, numeric_features)])
#         # ('cat', categorical_transformer, categorical_features)])

""" Het volgende stuk code toont aan hoe ik mijn DataFrame in sets opsplits."""

# Further split the data in seperate train and test files. 
x_train, x_test = train_test_split(df['y'], shuffle=True, test_size=0.2)

""" Test. """

# Create a classifier. 
random_forest = RandomForestClassifier()
# ('preprocessor', preprocessor),
rf = Pipeline(steps=[('classifier', random_forest)])

print(x_train.shape)
print(type(x_train))
# x_train = pd.DataFrame(x_train.values.reshape((-1, 1))).rename({'0': 'num'})
print(x_train.shape)
rf.fit(x_train)
# x_pred = rf.predict(x_test)

# # # Impute our data, then train
# # X_train_imp = imp.transform(x_train)
# # clf = RandomForestClassifier(n_estimators=10)
# # clf = clf.fit(X_train_imp, y_train)

# # for X_test in [x_test, y_test]:
# #      # Impute each test item, then predict
# #      X_test_imp = imp.transform(X_test)
# #      print(X_test, '->', clf.predict(X_test_imp))





