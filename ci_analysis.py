###############################################################################
# Code for the paper:
#   "Causal association of environmental variables on the occurrence of excesss of 
#    cutaneous leishmaniasis in Colombia: Are we looking to the wrong side?"
#    Gutiérrez, Avila and Altamiranda 
#   
#
###############################################################################





# -*- coding: utf-8 -*-
"""
Created on Thu Nov 25 19:04:08 2021

@author: juand
"""


# importing required libraries
import os, warnings, random
import dowhy
import econml
from dowhy import CausalModel
import pandas as pd
import numpy as np
import econml
from econml.dml import DML
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LassoCV
from zepid.graphics import EffectMeasurePlot
import matplotlib.pyplot as plt
from lightgbm import LGBMRegressor
import numpy as np, scipy.stats as st
import scipy.stats as stats
from zepid.causal.causalgraph import DirectedAcyclicGraph
from zepid.graphics import EffectMeasurePlot
import numpy as np, scipy.stats as st
from sklearn.linear_model import LassoCV
from econml.dml import CausalForestDML
from itertools import product
from econml.dml import SparseLinearDML
from econml.dml import LinearDML
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import RandomForestClassifier
from econml.score import RScorer
from sklearn.model_selection import train_test_split
from econml.dml import CausalForestDML
from joblib import Parallel, delayed
import warnings
from xgboost import XGBRegressor, XGBClassifier
from plotnine import ggplot, aes, geom_line, geom_ribbon, ggtitle, labs, geom_point, geom_hline, theme_linedraw, theme, element_rect, theme_light, element_line, element_text





# Set seeds to make the results more reproducible
def seed_everything(seed=999):
    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    os.environ['TF_DETERMINISTIC_OPS'] = '1'

seed = 999
seed_everything(seed)
warnings.filterwarnings('ignore')
pd.set_option('display.float_format', lambda x: '%.2f' % x)




#%%#
# Create data frame of ATE results
df_ATE = pd.DataFrame(0, index=range(0, 6), columns=['ATE', '95% CI'])

# Convert the first column to numpy.float64
df_ATE['ATE'] = df_ATE['ATE'].astype(np.float64)

# Convert the second column to tuples of zeros
df_ATE['95% CI'] = [(0.0, 0.0)] * 6

# Display the DataFrame
print(df_ATE)


#%%#

#import data
#data_col = pd.read_csv("https://raw.githubusercontent.com/juandavidgutier/causal_inference_cutaneous_leishmaniasis/main/dataset_leish.csv", encoding='latin-1') 

#import dask.dataframe as dd

data_col = pd.read_csv("D:/clases/UDES/articulo leishmaniasis/causal_inference/nuevo/data_final.csv", encoding='latin-1') 
#data_col = dd.read_csv("D:/clases/UDES/articulo leishmaniasis/causal_inference/nuevo/data_final.csv") 


data_col = data_col[(data_col['Year'] <= 2019)]

#potential confounders as binary
data_col['SST3'] = pd.qcut(data_col['SST3'], 2, labels=False)
data_col['SST4'] = pd.qcut(data_col['SST4'], 2, labels=False)
data_col['SST34'] = pd.qcut(data_col['SST34'], 2, labels=False)
data_col['SST12'] = pd.qcut(data_col['SST12'], 2, labels=False)
data_col['SOI'] = pd.qcut(data_col['SOI'], 2, labels=False)
data_col['NATL'] = pd.qcut(data_col['NATL'], 2, labels=False)
data_col['SATL'] = pd.qcut(data_col['SATL'], 2, labels=False)
data_col['TROP'] = pd.qcut(data_col['TROP'], 2, labels=False)
data_col['vectors'] = pd.qcut(data_col['vectors'], 2, labels=False)
data_col['forest'] = pd.qcut(data_col['forest'], 2, labels=False)



#%%#
reg1 = lambda: XGBClassifier(n_estimators=1000, random_state=999)
reg2 = lambda: XGBRegressor(n_estimators=1000, random_state=999)


#%%#
#Air Temperature
Colombia_airtemp = data_col[['excess', 'lag1_excess', 'lag2_excess', 'lag3_excess', 'Temp', 'SOI', 'SST3', 'SST4', 'SST34', 'SST12', 'NATL', 'SATL', 'TROP', 'forest', 'Year', 'Month', 'vectors']] 
Colombia_airtemp = Colombia_airtemp.dropna()
#Soil Temperature
Colombia_soiltemp = data_col[['excess', 'lag1_excess', 'lag2_excess', 'lag3_excess', 'Soil_temp', 'SOI', 'SST3', 'SST4', 'SST34', 'SST12', 'NATL', 'SATL', 'TROP', 'forest', 'Year', 'Month', 'vectors']] 
Colombia_soiltemp = Colombia_soiltemp.dropna()
#Rainfall
Colombia_rain = data_col[['excess', 'lag1_excess', 'lag2_excess', 'lag3_excess', 'Rain', 'SOI', 'SST3', 'SST4', 'SST34', 'SST12', 'NATL', 'SATL', 'TROP', 'forest', 'Year', 'Month', 'vectors']] 
Colombia_rain = Colombia_rain.dropna()
#Runoff
Colombia_runoff = data_col[['excess', 'lag1_excess', 'lag2_excess', 'lag3_excess', 'Qs', 'SOI', 'SST3', 'SST4', 'SST34', 'SST12', 'NATL', 'SATL', 'TROP', 'forest', 'Year', 'Month', 'vectors']] 
Colombia_runoff = Colombia_runoff.dropna()
#Soil Moisture
Colombia_soilmoist = data_col[['excess', 'lag1_excess', 'lag2_excess', 'lag3_excess', 'Soil_mois', 'SOI', 'SST3', 'SST4', 'SST34', 'SST12', 'NATL', 'SATL', 'TROP', 'forest', 'Year', 'Month', 'vectors']] 
Colombia_soilmoist = Colombia_soilmoist.dropna()
#EVI
Colombia_EVI = data_col[['excess', 'lag1_excess', 'lag2_excess', 'lag3_excess', 'EVI', 'SOI', 'SST3', 'SST4', 'SST34', 'SST12', 'NATL', 'SATL', 'TROP', 'forest', 'Year', 'Month', 'vectors']] 
Colombia_EVI = Colombia_EVI.dropna()

#%%#

##Air Temperature

#Step 1: Modeling the causal mechanism
model_leish=CausalModel(
        data = Colombia_airtemp,
        treatment=['Temp'],
        outcome=['excess'],
        graph= """graph[directed 1 node[id "Temp" label "Temp"]
                    node[id "excess" label "excess"]
                    node[id "SOI" label "SOI"]
                    node[id "SST3" label "SST3"]               
                    node[id "SST4" label "SST4"]
                    node[id "SST34" label "SST34"]
                    node[id "SST12" label "SST12"]
                    node[id "NATL" label "NATL"]
                    node[id "SATL" label "SATL"]
                    node[id "TROP" label "TROP"]                                      
                    node[id "forest" label "forest"]
                    node[id "Year" label "Year"]
                    node[id "Month" label "Month"]
                    node[id "vectors" label "vectors"]
                                       
                    edge[source "SOI" target "Temp"]
                    edge[source "SOI" target "excess"]
                                     
                    edge[source "SST3" target "Temp"]
                    edge[source "SST3" target "excess"]
                    
                    edge[source "SST4" target "Temp"]
                    edge[source "SST4" target "excess"]
                    
                    edge[source "SST34" target "Temp"]
                    edge[source "SST34" target "excess"]
                    
                    edge[source "SST12" target "Temp"]
                    edge[source "SST12" target "excess"]
                    
                    edge[source "NATL" target "Temp"]
                    edge[source "NATL" target "excess"]
                    
                    edge[source "SATL" target "Temp"]
                    edge[source "SATL" target "excess"]
                    
                    edge[source "TROP" target "Temp"]
                    edge[source "TROP" target "excess"]
                                                      
                    edge[source "forest" target "Temp"]
                    edge[source "forest" target "excess"]
                    
                    edge[source "Year" target "Temp"]
                    edge[source "Year" target "excess"]
                    edge[source "Year" target "SST3"]
                    edge[source "Year" target "SST4"]
                    edge[source "Year" target "SST34"]
                    edge[source "Year" target "SST12"]
                    edge[source "Year" target "NATL"]
                    edge[source "Year" target "SATL"]
                    edge[source "Year" target "TROP"]
                    edge[source "Year" target "SOI"]
                    
                    edge[source "Month" target "Temp"]
                    edge[source "Month" target "excess"]
                    edge[source "Month" target "SST3"]
                    edge[source "Month" target "SST4"]
                    edge[source "Month" target "SST34"]
                    edge[source "Month" target "SST12"]
                    edge[source "Month" target "NATL"]
                    edge[source "Month" target "SATL"]
                    edge[source "Month" target "TROP"]
                    edge[source "Month" target "SOI"]
                                        
                    edge[source "SOI" target "SST3"]
                    edge[source "SOI" target "SST4"]
                    edge[source "SOI" target "SST34"]
                    edge[source "SOI" target "SST12"]
                    edge[source "SOI" target "NATL"]
                    edge[source "SOI" target "SATL"]
                    edge[source "SOI" target "TROP"]
                                       
                    edge[source "SST3" target "SST4"]
                    edge[source "SST3" target "SST34"]
                    edge[source "SST3" target "SST12"]
                    edge[source "SST3" target "NATL"]
                    edge[source "SST3" target "SATL"]
                    edge[source "SST3" target "TROP"]
                    
                    edge[source "SST4" target "SST34"]
                    edge[source "SST4" target "SST12"]
                    edge[source "SST4" target "NATL"]
                    edge[source "SST4" target "SATL"]
                    edge[source "SST4" target "TROP"]
                    
                    edge[source "SST34" target "SST12"]
                    edge[source "SST34" target "NATL"]
                    edge[source "SST34" target "SATL"]
                    edge[source "SST34" target "TROP"]
                    
                    edge[source "SST12" target "NATL"]
                    edge[source "SST12" target "SATL"]
                    edge[source "SST12" target "TROP"]
                                                    
                    edge[source "NATL" target "SATL"]
                    edge[source "NATL" target "TROP"]
                    
                    edge[source "SATL" target "TROP"]
                    
                    edge[source "SOI" target "vectors"]
                    edge[source "SST3" target "vectors"]
                    edge[source "SST4" target "vectors"]
                    edge[source "SST34" target "vectors"]
                    edge[source "SST12" target "vectors"]
                    edge[source "NATL" target "vectors"]
                    edge[source "SATL" target "vectors"]
                    edge[source "TROP" target "vectors"]
                    edge[source "forest" target "vectors"]
                    edge[source "Temp" target "vectors"]
                    edge[source "vectors" target "excess"]
                    
                    edge[source "Temp" target "excess"]]"""
                    )
    

#Step 2: Identifying effects
identified_estimand_temp = model_leish.identify_effect(proceed_when_unidentifiable=False)
print(identified_estimand_temp)

#Step 3: Fit the model
Colombia_airtemp['Temp'] = Colombia_airtemp['Temp'].astype(np.uint8)
Colombia_airtemp.Temp = stats.zscore(Colombia_airtemp.Temp, nan_policy='omit') 
Colombia_airtemp['Temp'].std() 
Colombia_airtemp['forest'] = Colombia_airtemp['forest'].astype(np.uint8)
Y = Colombia_airtemp.excess.to_numpy() 
T = Colombia_airtemp.Temp.to_numpy()
W = Colombia_airtemp[['SOI', 'SST3', 'SST4', 'SST34', 'SST12', 'NATL', 'SATL', 
                      'TROP', 'forest', 'Year', 'Month']].to_numpy().reshape(-1, 11)
X = Colombia_airtemp[['forest']].to_numpy()

estimate_airtemp = DML(model_y=reg1(), model_t=reg2(), model_final=LassoCV(), discrete_treatment=False,
                       featurizer=PolynomialFeatures(degree=3),
                       linear_first_stages=False, cv=3, random_state=999)

estimate_airtemp.fit(Y=Y, T=T, X=X, W=W, inference='bootstrap')  

# predict effect for each sample X
estimate_airtemp.effect(X)

# ate
ate_AirTemp = estimate_airtemp.ate(X) 
print(ate_AirTemp)

# confidence interval of ate
ci_AirTemp = estimate_airtemp.ate_interval(X) 
print(ci_AirTemp)

# Set values in the df_ATE
df_ATE.at[0, 'ATE'] = ate_AirTemp  
df_ATE.at[0, '95% CI'] = ci_AirTemp  
print(df_ATE)




#%%#

##Soil_temperature

#Step 1: Modeling the causal mechanism
model_leish=CausalModel(
        data = Colombia_soiltemp,
        treatment=['Soil_temp'],
        outcome=['excess'],
        graph= """graph[directed 1 node[id "Soil_temp" label "Soil_temp"]
                    node[id "excess" label "excess"]
                    node[id "SOI" label "SOI"]
                    node[id "SST3" label "SST3"]               
                    node[id "SST4" label "SST4"]
                    node[id "SST34" label "SST34"]
                    node[id "SST12" label "SST12"]
                    node[id "NATL" label "NATL"]
                    node[id "SATL" label "SATL"]
                    node[id "TROP" label "TROP"]                                      
                    node[id "forest" label "forest"]
                    node[id "Year" label "Year"]
                    node[id "Month" label "Month"]
                    node[id "vectors" label "vectors"]
                                       
                    edge[source "SOI" target "Soil_temp"]
                    edge[source "SOI" target "excess"]
                                     
                    edge[source "SST3" target "Soil_temp"]
                    edge[source "SST3" target "excess"]
                    
                    edge[source "SST4" target "Soil_temp"]
                    edge[source "SST4" target "excess"]
                    
                    edge[source "SST34" target "Soil_temp"]
                    edge[source "SST34" target "excess"]
                    
                    edge[source "SST12" target "Soil_temp"]
                    edge[source "SST12" target "excess"]
                    
                    edge[source "NATL" target "Soil_temp"]
                    edge[source "NATL" target "excess"]
                    
                    edge[source "SATL" target "Soil_temp"]
                    edge[source "SATL" target "excess"]
                    
                    edge[source "TROP" target "Soil_temp"]
                    edge[source "TROP" target "excess"]
                                                      
                    edge[source "forest" target "Soil_temp"]
                    edge[source "forest" target "excess"]
                    
                    edge[source "Year" target "Soil_temp"]
                    edge[source "Year" target "excess"]
                    edge[source "Year" target "SST3"]
                    edge[source "Year" target "SST4"]
                    edge[source "Year" target "SST34"]
                    edge[source "Year" target "SST12"]
                    edge[source "Year" target "NATL"]
                    edge[source "Year" target "SATL"]
                    edge[source "Year" target "TROP"]
                    edge[source "Year" target "SOI"]
                    
                    edge[source "Month" target "Soil_temp"]
                    edge[source "Month" target "excess"]
                    edge[source "Month" target "SST3"]
                    edge[source "Month" target "SST4"]
                    edge[source "Month" target "SST34"]
                    edge[source "Month" target "SST12"]
                    edge[source "Month" target "NATL"]
                    edge[source "Month" target "SATL"]
                    edge[source "Month" target "TROP"]
                    edge[source "Month" target "SOI"]
                                        
                    edge[source "SOI" target "SST3"]
                    edge[source "SOI" target "SST4"]
                    edge[source "SOI" target "SST34"]
                    edge[source "SOI" target "SST12"]
                    edge[source "SOI" target "NATL"]
                    edge[source "SOI" target "SATL"]
                    edge[source "SOI" target "TROP"]
                                       
                    edge[source "SST3" target "SST4"]
                    edge[source "SST3" target "SST34"]
                    edge[source "SST3" target "SST12"]
                    edge[source "SST3" target "NATL"]
                    edge[source "SST3" target "SATL"]
                    edge[source "SST3" target "TROP"]
                    
                    edge[source "SST4" target "SST34"]
                    edge[source "SST4" target "SST12"]
                    edge[source "SST4" target "NATL"]
                    edge[source "SST4" target "SATL"]
                    edge[source "SST4" target "TROP"]
                    
                    edge[source "SST34" target "SST12"]
                    edge[source "SST34" target "NATL"]
                    edge[source "SST34" target "SATL"]
                    edge[source "SST34" target "TROP"]
                    
                    edge[source "SST12" target "NATL"]
                    edge[source "SST12" target "SATL"]
                    edge[source "SST12" target "TROP"]
                                                    
                    edge[source "NATL" target "SATL"]
                    edge[source "NATL" target "TROP"]
                    
                    edge[source "SATL" target "TROP"]
                    
                    edge[source "SOI" target "vectors"]
                    edge[source "SST3" target "vectors"]
                    edge[source "SST4" target "vectors"]
                    edge[source "SST34" target "vectors"]
                    edge[source "SST12" target "vectors"]
                    edge[source "NATL" target "vectors"]
                    edge[source "SATL" target "vectors"]
                    edge[source "TROP" target "vectors"]
                    edge[source "forest" target "vectors"]
                    edge[source "Soil_temp" target "vectors"]
                    edge[source "vectors" target "excess"]
                    
                    edge[source "Soil_temp" target "excess"]]"""
                    )
    

#Step 2: Identifying effects
identified_estimand_Soil_temp = model_leish.identify_effect(proceed_when_unidentifiable=False)
print(identified_estimand_Soil_temp)

#Step 3: Fit the model
Colombia_soiltemp['Soil_temp'] = Colombia_soiltemp['Soil_temp'].astype(np.uint8)
Colombia_soiltemp.Soil_temp = stats.zscore(Colombia_soiltemp.Soil_temp, nan_policy='omit') 
Colombia_soiltemp['Soil_temp'].std() 
Colombia_soiltemp['forest'] = Colombia_soiltemp['forest'].astype(np.uint8)
Y = Colombia_soiltemp.excess.to_numpy() 
T = Colombia_soiltemp.Soil_temp.to_numpy()
W = Colombia_soiltemp[['SOI', 'SST3', 'SST4', 'SST34', 'SST12', 'NATL', 'SATL', 
                      'TROP', 'forest', 'Year', 'Month']].to_numpy().reshape(-1, 11)
X = Colombia_soiltemp[['forest']].to_numpy()

estimate_Soil_temp = DML(model_y=reg1(), model_t=reg2(), model_final=LassoCV(), discrete_treatment=False,
                       featurizer=PolynomialFeatures(degree=3),
                       linear_first_stages=False, cv=3, random_state=999)

estimate_Soil_temp.fit(Y=Y, T=T, X=X, W=W, inference='bootstrap')  

# predict effect for each sample X
estimate_Soil_temp.effect(X)

# ate
ate_Soil_temp = estimate_Soil_temp.ate(X) 
print(ate_Soil_temp)

# confidence interval of ate
ci_Soil_temp = estimate_Soil_temp.ate_interval(X) 
print(ci_Soil_temp)

# Set values in the df_ATE
df_ATE.at[1, 'ATE'] = ate_Soil_temp  
df_ATE.at[1, '95% CI'] = ci_Soil_temp  
print(df_ATE)



#%%#

##Rainfall

#Step 1: Modeling the causal mechanism
model_leish=CausalModel(
        data = Colombia_rain,
        treatment=['Rain'],
        outcome=['excess'],
        graph= """graph[directed 1 node[id "Rain" label "Rain"]
                    node[id "excess" label "excess"]
                    node[id "SOI" label "SOI"]
                    node[id "SST3" label "SST3"]               
                    node[id "SST4" label "SST4"]
                    node[id "SST34" label "SST34"]
                    node[id "SST12" label "SST12"]
                    node[id "NATL" label "NATL"]
                    node[id "SATL" label "SATL"]
                    node[id "TROP" label "TROP"]                                      
                    node[id "forest" label "forest"]
                    node[id "Year" label "Year"]
                    node[id "Month" label "Month"]
                    node[id "vectors" label "vectors"]
                                       
                    edge[source "SOI" target "Rain"]
                    edge[source "SOI" target "excess"]
                                     
                    edge[source "SST3" target "Rain"]
                    edge[source "SST3" target "excess"]
                    
                    edge[source "SST4" target "Rain"]
                    edge[source "SST4" target "excess"]
                    
                    edge[source "SST34" target "Rain"]
                    edge[source "SST34" target "excess"]
                    
                    edge[source "SST12" target "Rain"]
                    edge[source "SST12" target "excess"]
                    
                    edge[source "NATL" target "Rain"]
                    edge[source "NATL" target "excess"]
                    
                    edge[source "SATL" target "Rain"]
                    edge[source "SATL" target "excess"]
                    
                    edge[source "TROP" target "Rain"]
                    edge[source "TROP" target "excess"]
                                                      
                    edge[source "forest" target "Rain"]
                    edge[source "forest" target "excess"]
                    
                    edge[source "Year" target "Rain"]
                    edge[source "Year" target "excess"]
                    edge[source "Year" target "SST3"]
                    edge[source "Year" target "SST4"]
                    edge[source "Year" target "SST34"]
                    edge[source "Year" target "SST12"]
                    edge[source "Year" target "NATL"]
                    edge[source "Year" target "SATL"]
                    edge[source "Year" target "TROP"]
                    edge[source "Year" target "SOI"]
                    
                    edge[source "Month" target "Rain"]
                    edge[source "Month" target "excess"]
                    edge[source "Month" target "SST3"]
                    edge[source "Month" target "SST4"]
                    edge[source "Month" target "SST34"]
                    edge[source "Month" target "SST12"]
                    edge[source "Month" target "NATL"]
                    edge[source "Month" target "SATL"]
                    edge[source "Month" target "TROP"]
                    edge[source "Month" target "SOI"]
                                        
                    edge[source "SOI" target "SST3"]
                    edge[source "SOI" target "SST4"]
                    edge[source "SOI" target "SST34"]
                    edge[source "SOI" target "SST12"]
                    edge[source "SOI" target "NATL"]
                    edge[source "SOI" target "SATL"]
                    edge[source "SOI" target "TROP"]
                                       
                    edge[source "SST3" target "SST4"]
                    edge[source "SST3" target "SST34"]
                    edge[source "SST3" target "SST12"]
                    edge[source "SST3" target "NATL"]
                    edge[source "SST3" target "SATL"]
                    edge[source "SST3" target "TROP"]
                    
                    edge[source "SST4" target "SST34"]
                    edge[source "SST4" target "SST12"]
                    edge[source "SST4" target "NATL"]
                    edge[source "SST4" target "SATL"]
                    edge[source "SST4" target "TROP"]
                    
                    edge[source "SST34" target "SST12"]
                    edge[source "SST34" target "NATL"]
                    edge[source "SST34" target "SATL"]
                    edge[source "SST34" target "TROP"]
                    
                    edge[source "SST12" target "NATL"]
                    edge[source "SST12" target "SATL"]
                    edge[source "SST12" target "TROP"]
                                                    
                    edge[source "NATL" target "SATL"]
                    edge[source "NATL" target "TROP"]
                    
                    edge[source "SATL" target "TROP"]
                    
                    edge[source "SOI" target "vectors"]
                    edge[source "SST3" target "vectors"]
                    edge[source "SST4" target "vectors"]
                    edge[source "SST34" target "vectors"]
                    edge[source "SST12" target "vectors"]
                    edge[source "NATL" target "vectors"]
                    edge[source "SATL" target "vectors"]
                    edge[source "TROP" target "vectors"]
                    edge[source "forest" target "vectors"]
                    edge[source "Rain" target "vectors"]
                    edge[source "vectors" target "excess"]
                    
                    edge[source "Rain" target "excess"]]"""
                    )
    

#Step 2: Identifying effects
identified_estimand_Rain = model_leish.identify_effect(proceed_when_unidentifiable=False)
print(identified_estimand_Rain)

#Step 3: Fit the model
Colombia_rain['Rain'] = Colombia_rain['Rain'].astype(np.uint8)
Colombia_rain.Rain = stats.zscore(Colombia_rain.Rain, nan_policy='omit') 
Colombia_rain['Rain'].std() 
Colombia_rain['forest'] = Colombia_rain['forest'].astype(np.uint8)
Y = Colombia_rain.excess.to_numpy() 
T = Colombia_rain.Rain.to_numpy()
W = Colombia_rain[['SOI', 'SST3', 'SST4', 'SST34', 'SST12', 'NATL', 'SATL', 
                      'TROP', 'forest', 'Year', 'Month']].to_numpy().reshape(-1, 11)
X = Colombia_rain[['forest']].to_numpy()

estimate_Rain = DML(model_y=reg1(), model_t=reg2(), model_final=LassoCV(), discrete_treatment=False,
                       featurizer=PolynomialFeatures(degree=3),
                       linear_first_stages=False, cv=3, random_state=999)

estimate_Rain.fit(Y=Y, T=T, X=X, W=W, inference='bootstrap')  

# predict effect for each sample X
estimate_Rain.effect(X)

# ate
ate_Rain = estimate_Rain.ate(X) 
print(ate_Rain)

# confidence interval of ate
ci_Rain = estimate_Rain.ate_interval(X) 
print(ci_Rain)

# Set values in the df_ATE
df_ATE.at[2, 'ATE'] = ate_Rain  
df_ATE.at[2, '95% CI'] = ci_Rain  
print(df_ATE)



#%%#

##Runoff

#Step 1: Modeling the causal mechanism
model_leish=CausalModel(
        data = Colombia_runoff,
        treatment=['Qs'],
        outcome=['excess'],
        graph= """graph[directed 1 node[id "Qs" label "Qs"]
                    node[id "excess" label "excess"]
                    node[id "SOI" label "SOI"]
                    node[id "SST3" label "SST3"]               
                    node[id "SST4" label "SST4"]
                    node[id "SST34" label "SST34"]
                    node[id "SST12" label "SST12"]
                    node[id "NATL" label "NATL"]
                    node[id "SATL" label "SATL"]
                    node[id "TROP" label "TROP"]                                      
                    node[id "forest" label "forest"]
                    node[id "Year" label "Year"]
                    node[id "Month" label "Month"]
                    node[id "vectors" label "vectors"]
                                       
                    edge[source "SOI" target "Qs"]
                    edge[source "SOI" target "excess"]
                                     
                    edge[source "SST3" target "Qs"]
                    edge[source "SST3" target "excess"]
                    
                    edge[source "SST4" target "Qs"]
                    edge[source "SST4" target "excess"]
                    
                    edge[source "SST34" target "Qs"]
                    edge[source "SST34" target "excess"]
                    
                    edge[source "SST12" target "Qs"]
                    edge[source "SST12" target "excess"]
                    
                    edge[source "NATL" target "Qs"]
                    edge[source "NATL" target "excess"]
                    
                    edge[source "SATL" target "Qs"]
                    edge[source "SATL" target "excess"]
                    
                    edge[source "TROP" target "Qs"]
                    edge[source "TROP" target "excess"]
                                                      
                    edge[source "forest" target "Qs"]
                    edge[source "forest" target "excess"]
                    
                    edge[source "Year" target "Qs"]
                    edge[source "Year" target "excess"]
                    edge[source "Year" target "SST3"]
                    edge[source "Year" target "SST4"]
                    edge[source "Year" target "SST34"]
                    edge[source "Year" target "SST12"]
                    edge[source "Year" target "NATL"]
                    edge[source "Year" target "SATL"]
                    edge[source "Year" target "TROP"]
                    edge[source "Year" target "SOI"]
                    
                    edge[source "Month" target "Qs"]
                    edge[source "Month" target "excess"]
                    edge[source "Month" target "SST3"]
                    edge[source "Month" target "SST4"]
                    edge[source "Month" target "SST34"]
                    edge[source "Month" target "SST12"]
                    edge[source "Month" target "NATL"]
                    edge[source "Month" target "SATL"]
                    edge[source "Month" target "TROP"]
                    edge[source "Month" target "SOI"]
                                        
                    edge[source "SOI" target "SST3"]
                    edge[source "SOI" target "SST4"]
                    edge[source "SOI" target "SST34"]
                    edge[source "SOI" target "SST12"]
                    edge[source "SOI" target "NATL"]
                    edge[source "SOI" target "SATL"]
                    edge[source "SOI" target "TROP"]
                                       
                    edge[source "SST3" target "SST4"]
                    edge[source "SST3" target "SST34"]
                    edge[source "SST3" target "SST12"]
                    edge[source "SST3" target "NATL"]
                    edge[source "SST3" target "SATL"]
                    edge[source "SST3" target "TROP"]
                    
                    edge[source "SST4" target "SST34"]
                    edge[source "SST4" target "SST12"]
                    edge[source "SST4" target "NATL"]
                    edge[source "SST4" target "SATL"]
                    edge[source "SST4" target "TROP"]
                    
                    edge[source "SST34" target "SST12"]
                    edge[source "SST34" target "NATL"]
                    edge[source "SST34" target "SATL"]
                    edge[source "SST34" target "TROP"]
                    
                    edge[source "SST12" target "NATL"]
                    edge[source "SST12" target "SATL"]
                    edge[source "SST12" target "TROP"]
                                                    
                    edge[source "NATL" target "SATL"]
                    edge[source "NATL" target "TROP"]
                    
                    edge[source "SATL" target "TROP"]
                    
                    edge[source "SOI" target "vectors"]
                    edge[source "SST3" target "vectors"]
                    edge[source "SST4" target "vectors"]
                    edge[source "SST34" target "vectors"]
                    edge[source "SST12" target "vectors"]
                    edge[source "NATL" target "vectors"]
                    edge[source "SATL" target "vectors"]
                    edge[source "TROP" target "vectors"]
                    edge[source "forest" target "vectors"]
                    edge[source "Qs" target "vectors"]
                    edge[source "vectors" target "excess"]
                    
                    edge[source "Qs" target "excess"]]"""
                    )
    

#Step 2: Identifying effects
identified_estimand_Qs = model_leish.identify_effect(proceed_when_unidentifiable=False)
print(identified_estimand_Qs)

#Step 3: Fit the model
Colombia_runoff['Qs'] = Colombia_runoff['Qs'].astype(np.uint8)
Colombia_runoff.Qs = stats.zscore(Colombia_runoff.Qs, nan_policy='omit') 
Colombia_runoff['Qs'].std() 
Colombia_runoff['forest'] = Colombia_runoff['forest'].astype(np.uint8)
Y = Colombia_runoff.excess.to_numpy() 
T = Colombia_runoff.Qs.to_numpy()
W = Colombia_runoff[['SOI', 'SST3', 'SST4', 'SST34', 'SST12', 'NATL', 'SATL', 
                      'TROP', 'forest', 'Year', 'Month']].to_numpy().reshape(-1, 11)
X = Colombia_runoff[['forest']].to_numpy()

estimate_Qs = DML(model_y=reg1(), model_t=reg2(), model_final=LassoCV(), discrete_treatment=False,
                       featurizer=PolynomialFeatures(degree=3),
                       linear_first_stages=False, cv=3, random_state=999)

estimate_Qs.fit(Y=Y, T=T, X=X, W=W, inference='bootstrap')  

# predict effect for each sample X
estimate_Qs.effect(X)

# ate
ate_Qs = estimate_Qs.ate(X) 
print(ate_Qs)

# confidence interval of ate
ci_Qs = estimate_Qs.ate_interval(X) 
print(ci_Qs)

# Set values in the df_ATE
df_ATE.at[3, 'ATE'] = ate_Qs  
df_ATE.at[3, '95% CI'] = ci_Qs  
print(df_ATE)



#%%#
##Soil Moisture

#Step 1: Modeling the causal mechanism
model_leish=CausalModel(
        data = Colombia_soilmoist,
        treatment=['Soil_mois'],
        outcome=['excess'],
        graph= """graph[directed 1 node[id "Soil_mois" label "Soil_mois"]
                    node[id "excess" label "excess"]
                    node[id "SOI" label "SOI"]
                    node[id "SST3" label "SST3"]               
                    node[id "SST4" label "SST4"]
                    node[id "SST34" label "SST34"]
                    node[id "SST12" label "SST12"]
                    node[id "NATL" label "NATL"]
                    node[id "SATL" label "SATL"]
                    node[id "TROP" label "TROP"]                                      
                    node[id "forest" label "forest"]
                    node[id "Year" label "Year"]
                    node[id "Month" label "Month"]
                    node[id "vectors" label "vectors"]
                                       
                    edge[source "SOI" target "Soil_mois"]
                    edge[source "SOI" target "excess"]
                                     
                    edge[source "SST3" target "Soil_mois"]
                    edge[source "SST3" target "excess"]
                    
                    edge[source "SST4" target "Soil_mois"]
                    edge[source "SST4" target "excess"]
                    
                    edge[source "SST34" target "Soil_mois"]
                    edge[source "SST34" target "excess"]
                    
                    edge[source "SST12" target "Soil_mois"]
                    edge[source "SST12" target "excess"]
                    
                    edge[source "NATL" target "Soil_mois"]
                    edge[source "NATL" target "excess"]
                    
                    edge[source "SATL" target "Soil_mois"]
                    edge[source "SATL" target "excess"]
                    
                    edge[source "TROP" target "Soil_mois"]
                    edge[source "TROP" target "excess"]
                                                      
                    edge[source "forest" target "Soil_mois"]
                    edge[source "forest" target "excess"]
                    
                    edge[source "Year" target "Soil_mois"]
                    edge[source "Year" target "excess"]
                    edge[source "Year" target "SST3"]
                    edge[source "Year" target "SST4"]
                    edge[source "Year" target "SST34"]
                    edge[source "Year" target "SST12"]
                    edge[source "Year" target "NATL"]
                    edge[source "Year" target "SATL"]
                    edge[source "Year" target "TROP"]
                    edge[source "Year" target "SOI"]
                    
                    edge[source "Month" target "Soil_mois"]
                    edge[source "Month" target "excess"]
                    edge[source "Month" target "SST3"]
                    edge[source "Month" target "SST4"]
                    edge[source "Month" target "SST34"]
                    edge[source "Month" target "SST12"]
                    edge[source "Month" target "NATL"]
                    edge[source "Month" target "SATL"]
                    edge[source "Month" target "TROP"]
                    edge[source "Month" target "SOI"]
                                        
                    edge[source "SOI" target "SST3"]
                    edge[source "SOI" target "SST4"]
                    edge[source "SOI" target "SST34"]
                    edge[source "SOI" target "SST12"]
                    edge[source "SOI" target "NATL"]
                    edge[source "SOI" target "SATL"]
                    edge[source "SOI" target "TROP"]
                                       
                    edge[source "SST3" target "SST4"]
                    edge[source "SST3" target "SST34"]
                    edge[source "SST3" target "SST12"]
                    edge[source "SST3" target "NATL"]
                    edge[source "SST3" target "SATL"]
                    edge[source "SST3" target "TROP"]
                    
                    edge[source "SST4" target "SST34"]
                    edge[source "SST4" target "SST12"]
                    edge[source "SST4" target "NATL"]
                    edge[source "SST4" target "SATL"]
                    edge[source "SST4" target "TROP"]
                    
                    edge[source "SST34" target "SST12"]
                    edge[source "SST34" target "NATL"]
                    edge[source "SST34" target "SATL"]
                    edge[source "SST34" target "TROP"]
                    
                    edge[source "SST12" target "NATL"]
                    edge[source "SST12" target "SATL"]
                    edge[source "SST12" target "TROP"]
                                                    
                    edge[source "NATL" target "SATL"]
                    edge[source "NATL" target "TROP"]
                    
                    edge[source "SATL" target "TROP"]
                    
                    edge[source "SOI" target "vectors"]
                    edge[source "SST3" target "vectors"]
                    edge[source "SST4" target "vectors"]
                    edge[source "SST34" target "vectors"]
                    edge[source "SST12" target "vectors"]
                    edge[source "NATL" target "vectors"]
                    edge[source "SATL" target "vectors"]
                    edge[source "TROP" target "vectors"]
                    edge[source "forest" target "vectors"]
                    edge[source "Soil_mois" target "vectors"]
                    edge[source "vectors" target "excess"]
                    
                    edge[source "Soil_mois" target "excess"]]"""
                    )
    

#Step 2: Identifying effects
identified_estimand_Soil_mois = model_leish.identify_effect(proceed_when_unidentifiable=False)
print(identified_estimand_Soil_mois)

#Step 3: Fit the model
Colombia_soilmoist['Soil_mois'] = Colombia_soilmoist['Soil_mois'].astype(np.uint8)
Colombia_soilmoist.Soil_mois = stats.zscore(Colombia_soilmoist.Soil_mois, nan_policy='omit') 
Colombia_soilmoist['Soil_mois'].std() 
Colombia_soilmoist['forest'] = Colombia_soilmoist['forest'].astype(np.uint8)
Y = Colombia_soilmoist.excess.to_numpy() 
T = Colombia_soilmoist.Soil_mois.to_numpy()
W = Colombia_soilmoist[['SOI', 'SST3', 'SST4', 'SST34', 'SST12', 'NATL', 'SATL', 
                      'TROP', 'forest', 'Year', 'Month']].to_numpy().reshape(-1, 11)
X = Colombia_soilmoist[['forest']].to_numpy()

estimate_Soil_mois = DML(model_y=reg1(), model_t=reg2(), model_final=LassoCV(), discrete_treatment=False,
                       featurizer=PolynomialFeatures(degree=3),
                       linear_first_stages=False, cv=3, random_state=999)

estimate_Soil_mois.fit(Y=Y, T=T, X=X, W=W, inference='bootstrap')  

# predict effect for each sample X
estimate_Soil_mois.effect(X)

# ate
ate_Soil_mois = estimate_Soil_mois.ate(X) 
print(ate_Soil_mois)

# confidence interval of ate
ci_Soil_mois = estimate_Soil_mois.ate_interval(X) 
print(ci_Soil_mois)

# Set values in the df_ATE
df_ATE.at[4, 'ATE'] = ate_Soil_mois  
df_ATE.at[4, '95% CI'] = ci_Soil_mois  
print(df_ATE)



#%%#
##EVI

#Step 1: Modeling the causal mechanism
model_leish=CausalModel(
        data = Colombia_EVI,
        treatment=['EVI'],
        outcome=['excess'],
        graph= """graph[directed 1 node[id "EVI" label "EVI"]
                    node[id "excess" label "excess"]
                    node[id "SOI" label "SOI"]
                    node[id "SST3" label "SST3"]               
                    node[id "SST4" label "SST4"]
                    node[id "SST34" label "SST34"]
                    node[id "SST12" label "SST12"]
                    node[id "NATL" label "NATL"]
                    node[id "SATL" label "SATL"]
                    node[id "TROP" label "TROP"]                                      
                    node[id "forest" label "forest"]
                    node[id "Year" label "Year"]
                    node[id "Month" label "Month"]
                    node[id "vectors" label "vectors"]
                                       
                    edge[source "SOI" target "EVI"]
                    edge[source "SOI" target "excess"]
                                     
                    edge[source "SST3" target "EVI"]
                    edge[source "SST3" target "excess"]
                    
                    edge[source "SST4" target "EVI"]
                    edge[source "SST4" target "excess"]
                    
                    edge[source "SST34" target "EVI"]
                    edge[source "SST34" target "excess"]
                    
                    edge[source "SST12" target "EVI"]
                    edge[source "SST12" target "excess"]
                    
                    edge[source "NATL" target "EVI"]
                    edge[source "NATL" target "excess"]
                    
                    edge[source "SATL" target "EVI"]
                    edge[source "SATL" target "excess"]
                    
                    edge[source "TROP" target "EVI"]
                    edge[source "TROP" target "excess"]
                                                      
                    edge[source "forest" target "EVI"]
                    edge[source "forest" target "excess"]
                    
                    edge[source "Year" target "EVI"]
                    edge[source "Year" target "excess"]
                    edge[source "Year" target "SST3"]
                    edge[source "Year" target "SST4"]
                    edge[source "Year" target "SST34"]
                    edge[source "Year" target "SST12"]
                    edge[source "Year" target "NATL"]
                    edge[source "Year" target "SATL"]
                    edge[source "Year" target "TROP"]
                    edge[source "Year" target "SOI"]
                    
                    edge[source "Month" target "EVI"]
                    edge[source "Month" target "excess"]
                    edge[source "Month" target "SST3"]
                    edge[source "Month" target "SST4"]
                    edge[source "Month" target "SST34"]
                    edge[source "Month" target "SST12"]
                    edge[source "Month" target "NATL"]
                    edge[source "Month" target "SATL"]
                    edge[source "Month" target "TROP"]
                    edge[source "Month" target "SOI"]
                                        
                    edge[source "SOI" target "SST3"]
                    edge[source "SOI" target "SST4"]
                    edge[source "SOI" target "SST34"]
                    edge[source "SOI" target "SST12"]
                    edge[source "SOI" target "NATL"]
                    edge[source "SOI" target "SATL"]
                    edge[source "SOI" target "TROP"]
                                       
                    edge[source "SST3" target "SST4"]
                    edge[source "SST3" target "SST34"]
                    edge[source "SST3" target "SST12"]
                    edge[source "SST3" target "NATL"]
                    edge[source "SST3" target "SATL"]
                    edge[source "SST3" target "TROP"]
                    
                    edge[source "SST4" target "SST34"]
                    edge[source "SST4" target "SST12"]
                    edge[source "SST4" target "NATL"]
                    edge[source "SST4" target "SATL"]
                    edge[source "SST4" target "TROP"]
                    
                    edge[source "SST34" target "SST12"]
                    edge[source "SST34" target "NATL"]
                    edge[source "SST34" target "SATL"]
                    edge[source "SST34" target "TROP"]
                    
                    edge[source "SST12" target "NATL"]
                    edge[source "SST12" target "SATL"]
                    edge[source "SST12" target "TROP"]
                                                    
                    edge[source "NATL" target "SATL"]
                    edge[source "NATL" target "TROP"]
                    
                    edge[source "SATL" target "TROP"]
                    
                    edge[source "SOI" target "vectors"]
                    edge[source "SST3" target "vectors"]
                    edge[source "SST4" target "vectors"]
                    edge[source "SST34" target "vectors"]
                    edge[source "SST12" target "vectors"]
                    edge[source "NATL" target "vectors"]
                    edge[source "SATL" target "vectors"]
                    edge[source "TROP" target "vectors"]
                    edge[source "forest" target "vectors"]
                    edge[source "EVI" target "vectors"]
                    edge[source "vectors" target "excess"]
                    
                    edge[source "EVI" target "excess"]]"""
                    )
    

#Step 2: Identifying effects
identified_estimand_EVI = model_leish.identify_effect(proceed_when_unidentifiable=False)
print(identified_estimand_EVI)

#Step 3: Fit the model
Colombia_EVI['EVI'] = Colombia_EVI['EVI'].astype(np.uint8)
Colombia_EVI.EVI = stats.zscore(Colombia_EVI.EVI, nan_policy='omit') 
Colombia_EVI['EVI'].std() 
Colombia_EVI['forest'] = Colombia_EVI['forest'].astype(np.uint8)
Y = Colombia_EVI.excess.to_numpy() 
T = Colombia_EVI.EVI.to_numpy()
W = Colombia_EVI[['SOI', 'SST3', 'SST4', 'SST34', 'SST12', 'NATL', 'SATL', 
                      'TROP', 'forest', 'Year', 'Month']].to_numpy().reshape(-1, 11)
X = Colombia_EVI[['forest']].to_numpy()

estimate_EVI = DML(model_y=reg1(), model_t=reg2(), model_final=LassoCV(), discrete_treatment=False,
                       featurizer=PolynomialFeatures(degree=3),
                       linear_first_stages=False, cv=3, random_state=999)

estimate_EVI.fit(Y=Y, T=T, X=X, W=W, inference='bootstrap')  

# predict effect for each sample X
estimate_EVI.effect(X)

# ate
ate_EVI = estimate_EVI.ate(X) 
print(ate_EVI)

# confidence interval of ate
ci_EVI = estimate_EVI.ate_interval(X) 
print(ci_EVI)

# Set values in the df_ATE
df_ATE.at[5, 'ATE'] = ate_EVI  
df_ATE.at[5, '95% CI'] = ci_EVI  
print(df_ATE)



#%%#

#Figure 3A
labs = ['Air Temperature',
        'Soil Temperature',
        'Rainfall',
        'Runoff',
        'Soil Moisture',
        'EVI']

measure = [0.068, 0.067, '0.000', '0.000', 0.077, '0.000']
lower =   [0.065, 0.064, -0.003, '-0.010', 0.073, -0.002]
upper =   ['0.070', '0.070', 0.003, '0.010', '0.080', 0.002]

p = EffectMeasurePlot(label=labs, effect_measure=measure, lcl=lower, ucl=upper)
p.labels(center=0)
p.colors(pointcolor='r') 
p.labels(effectmeasure='ATE')  
p.plot(figsize=(10, 5), t_adjuster=0.075, max_value=0.1, min_value=-0.1)
plt.tight_layout()
plt.show()








#%%#
#lag1 
# Create data frame of ATE results
df_ATE_lag1 = pd.DataFrame(0, index=range(0, 6), columns=['ATE_lag1', '95% CI'])

# Convert the first column to numpy.float64
df_ATE_lag1['ATE_lag1'] = df_ATE_lag1['ATE_lag1'].astype(np.float64)

# Convert the second column to tuples of zeros
df_ATE_lag1['95% CI'] = [(0.0, 0.0)] * 6

# Display the DataFrame
print(df_ATE_lag1)



#%%#

##Air Temperature lag1

#Step 3: Fit the model
Colombia_airtemp['Temp'] = Colombia_airtemp['Temp'].astype(np.uint8)
Colombia_airtemp.Temp = stats.zscore(Colombia_airtemp.Temp, nan_policy='omit') 
Colombia_airtemp['Temp'].std() 
Colombia_airtemp['forest'] = Colombia_airtemp['forest'].astype(np.uint8)
Y = Colombia_airtemp.lag1_excess.to_numpy() 
T = Colombia_airtemp.Temp.to_numpy()
W = Colombia_airtemp[['SOI', 'SST3', 'SST4', 'SST34', 'SST12', 'NATL', 'SATL', 
                      'TROP', 'forest', 'Year', 'Month']].to_numpy().reshape(-1, 11)
X = Colombia_airtemp[['forest']].to_numpy()

estimate_airtemp = DML(model_y=reg1(), model_t=reg2(), model_final=LassoCV(), discrete_treatment=False,
                       featurizer=PolynomialFeatures(degree=3),
                       linear_first_stages=False, cv=3, random_state=999)

estimate_airtemp.fit(Y=Y, T=T, X=X, W=W, inference='bootstrap')  

# predict effect for each sample X
estimate_airtemp.effect(X)

# ate
ate_AirTemp = estimate_airtemp.ate(X) 
print(ate_AirTemp)

# confidence interval of ate
ci_AirTemp = estimate_airtemp.ate_interval(X) 
print(ci_AirTemp)

# Set values in the df_ATE_lag1
df_ATE_lag1.at[0, 'ATE_lag1'] = ate_AirTemp  
df_ATE_lag1.at[0, '95% CI'] = ci_AirTemp  
print(df_ATE_lag1)




#%%#

##Soil_temperature lag1

#Step 3: Fit the model
Colombia_soiltemp['Soil_temp'] = Colombia_soiltemp['Soil_temp'].astype(np.uint8)
Colombia_soiltemp.Soil_temp = stats.zscore(Colombia_soiltemp.Soil_temp, nan_policy='omit') 
Colombia_soiltemp['Soil_temp'].std() 
Colombia_soiltemp['forest'] = Colombia_soiltemp['forest'].astype(np.uint8)
Y = Colombia_soiltemp.lag1_excess.to_numpy() 
T = Colombia_soiltemp.Soil_temp.to_numpy()
W = Colombia_soiltemp[['SOI', 'SST3', 'SST4', 'SST34', 'SST12', 'NATL', 'SATL', 
                      'TROP', 'forest', 'Year', 'Month']].to_numpy().reshape(-1, 11)
X = Colombia_soiltemp[['forest']].to_numpy()

estimate_Soil_temp = DML(model_y=reg1(), model_t=reg2(), model_final=LassoCV(), discrete_treatment=False,
                       featurizer=PolynomialFeatures(degree=3),
                       linear_first_stages=False, cv=3, random_state=999)

estimate_Soil_temp.fit(Y=Y, T=T, X=X, W=W, inference='bootstrap')  

# predict effect for each sample X
estimate_Soil_temp.effect(X)

# ate
ate_Soil_temp = estimate_Soil_temp.ate(X) 
print(ate_Soil_temp)

# confidence interval of ate
ci_Soil_temp = estimate_Soil_temp.ate_interval(X) 
print(ci_Soil_temp)

# Set values in the df_ATE_lag1
df_ATE_lag1.at[1, 'ATE_lag1'] = ate_Soil_temp  
df_ATE_lag1.at[1, '95% CI'] = ci_Soil_temp  
print(df_ATE_lag1)



#%%#

##Rainfall lag1

#Step 3: Fit the model
Colombia_rain['Rain'] = Colombia_rain['Rain'].astype(np.uint8)
Colombia_rain.Rain = stats.zscore(Colombia_rain.Rain, nan_policy='omit') 
Colombia_rain['Rain'].std() 
Colombia_rain['forest'] = Colombia_rain['forest'].astype(np.uint8)
Y = Colombia_rain.lag1_excess.to_numpy() 
T = Colombia_rain.Rain.to_numpy()
W = Colombia_rain[['SOI', 'SST3', 'SST4', 'SST34', 'SST12', 'NATL', 'SATL', 
                      'TROP', 'forest', 'Year', 'Month']].to_numpy().reshape(-1, 11)
X = Colombia_rain[['forest']].to_numpy()

estimate_Rain = DML(model_y=reg1(), model_t=reg2(), model_final=LassoCV(), discrete_treatment=False,
                       featurizer=PolynomialFeatures(degree=3),
                       linear_first_stages=False, cv=3, random_state=999)

estimate_Rain.fit(Y=Y, T=T, X=X, W=W, inference='bootstrap')  

# predict effect for each sample X
estimate_Rain.effect(X)

# ate
ate_Rain = estimate_Rain.ate(X) 
print(ate_Rain)

# confidence interval of ate
ci_Rain = estimate_Rain.ate_interval(X) 
print(ci_Rain)

# Set values in the df_ATE_lag1
df_ATE_lag1.at[2, 'ATE_lag1'] = ate_Rain  
df_ATE_lag1.at[2, '95% CI'] = ci_Rain  
print(df_ATE_lag1)



#%%#

##Runoff lag1

#Step 3: Fit the model
Colombia_runoff['Qs'] = Colombia_runoff['Qs'].astype(np.uint8)
Colombia_runoff.Qs = stats.zscore(Colombia_runoff.Qs, nan_policy='omit') 
Colombia_runoff['Qs'].std() 
Colombia_runoff['forest'] = Colombia_runoff['forest'].astype(np.uint8)
Y = Colombia_runoff.lag1_excess.to_numpy() 
T = Colombia_runoff.Qs.to_numpy()
W = Colombia_runoff[['SOI', 'SST3', 'SST4', 'SST34', 'SST12', 'NATL', 'SATL', 
                      'TROP', 'forest', 'Year', 'Month']].to_numpy().reshape(-1, 11)
X = Colombia_runoff[['forest']].to_numpy()

estimate_Qs = DML(model_y=reg1(), model_t=reg2(), model_final=LassoCV(), discrete_treatment=False,
                       featurizer=PolynomialFeatures(degree=3),
                       linear_first_stages=False, cv=3, random_state=999)

estimate_Qs.fit(Y=Y, T=T, X=X, W=W, inference='bootstrap')  

# predict effect for each sample X
estimate_Qs.effect(X)

# ate
ate_Qs = estimate_Qs.ate(X) 
print(ate_Qs)

# confidence interval of ate
ci_Qs = estimate_Qs.ate_interval(X) 
print(ci_Qs)

# Set values in the df_ATE_lag1
df_ATE_lag1.at[3, 'ATE_lag1'] = ate_Qs  
df_ATE_lag1.at[3, '95% CI'] = ci_Qs  
print(df_ATE_lag1)



#%%#
##Soil Moisture lag1

#Step 3: Fit the model
Colombia_soilmoist['Soil_mois'] = Colombia_soilmoist['Soil_mois'].astype(np.uint8)
Colombia_soilmoist.Soil_mois = stats.zscore(Colombia_soilmoist.Soil_mois, nan_policy='omit') 
Colombia_soilmoist['Soil_mois'].std() 
Colombia_soilmoist['forest'] = Colombia_soilmoist['forest'].astype(np.uint8)
Y = Colombia_soilmoist.lag1_excess.to_numpy() 
T = Colombia_soilmoist.Soil_mois.to_numpy()
W = Colombia_soilmoist[['SOI', 'SST3', 'SST4', 'SST34', 'SST12', 'NATL', 'SATL', 
                      'TROP', 'forest', 'Year', 'Month']].to_numpy().reshape(-1, 11)
X = Colombia_soilmoist[['forest']].to_numpy()

estimate_Soil_mois = DML(model_y=reg1(), model_t=reg2(), model_final=LassoCV(), discrete_treatment=False,
                       featurizer=PolynomialFeatures(degree=3),
                       linear_first_stages=False, cv=3, random_state=999)

estimate_Soil_mois.fit(Y=Y, T=T, X=X, W=W, inference='bootstrap')  

# predict effect for each sample X
estimate_Soil_mois.effect(X)

# ate
ate_Soil_mois = estimate_Soil_mois.ate(X) 
print(ate_Soil_mois)

# confidence interval of ate
ci_Soil_mois = estimate_Soil_mois.ate_interval(X) 
print(ci_Soil_mois)

# Set values in the df_ATE_lag1
df_ATE_lag1.at[4, 'ATE_lag1'] = ate_Soil_mois  
df_ATE_lag1.at[4, '95% CI'] = ci_Soil_mois  
print(df_ATE_lag1)



#%%#
##EVI lag1

#Step 3: Fit the model
Colombia_EVI['EVI'] = Colombia_EVI['EVI'].astype(np.uint8)
Colombia_EVI.EVI = stats.zscore(Colombia_EVI.EVI, nan_policy='omit') 
Colombia_EVI['EVI'].std() 
Colombia_EVI['forest'] = Colombia_EVI['forest'].astype(np.uint8)
Y = Colombia_EVI.lag1_excess.to_numpy() 
T = Colombia_EVI.EVI.to_numpy()
W = Colombia_EVI[['SOI', 'SST3', 'SST4', 'SST34', 'SST12', 'NATL', 'SATL', 
                      'TROP', 'forest', 'Year', 'Month']].to_numpy().reshape(-1, 11)
X = Colombia_EVI[['forest']].to_numpy()

estimate_EVI = DML(model_y=reg1(), model_t=reg2(), model_final=LassoCV(), discrete_treatment=False,
                       featurizer=PolynomialFeatures(degree=3),
                       linear_first_stages=False, cv=3, random_state=999)

estimate_EVI.fit(Y=Y, T=T, X=X, W=W, inference='bootstrap')  

# predict effect for each sample X
estimate_EVI.effect(X)

# ate
ate_EVI = estimate_EVI.ate(X) 
print(ate_EVI)

# confidence interval of ate
ci_EVI = estimate_EVI.ate_interval(X) 
print(ci_EVI)

# Set values in the df_ATE_lag1
df_ATE_lag1.at[5, 'ATE_lag1'] = ate_EVI  
df_ATE_lag1.at[5, '95% CI'] = ci_EVI  
print(df_ATE_lag1)



#%%#

#Figure 3B
labs = ['Air Temperature',
        'Soil Temperature',
        'Rainfall',
        'Runoff',
        'Soil Moisture',
        'EVI']

measure = [0.068, 0.067, -0.002, '0.000', 0.077, '0.000']
lower =   [0.066, 0.064, -0.005, -0.014, 0.074, -0.003]
upper =   [0.071, '0.070', 0.001, 0.014, 0.081, 0.003]

p = EffectMeasurePlot(label=labs, effect_measure=measure, lcl=lower, ucl=upper)
p.labels(center=0)
p.colors(pointcolor='r') 
p.labels(effectmeasure='ATE')  
p.plot(figsize=(11, 5.5), t_adjuster=0.075, max_value=0.1, min_value=-0.1)
plt.tight_layout







#%%#
#lag2
# Create data frame of ATE results
df_ATE_lag2 = pd.DataFrame(0, index=range(0, 6), columns=['ATE_lag2', '95% CI'])

# Convert the first column to numpy.float64
df_ATE_lag2['ATE_lag2'] = df_ATE_lag2['ATE_lag2'].astype(np.float64)

# Convert the second column to tuples of zeros
df_ATE_lag2['95% CI'] = [(0.0, 0.0)] * 6

# Display the DataFrame
print(df_ATE_lag2)



#%%#

##Air Temperature lag2

#Step 3: Fit the model
Colombia_airtemp['Temp'] = Colombia_airtemp['Temp'].astype(np.uint8)
Colombia_airtemp.Temp = stats.zscore(Colombia_airtemp.Temp, nan_policy='omit') 
Colombia_airtemp['Temp'].std() 
Colombia_airtemp['forest'] = Colombia_airtemp['forest'].astype(np.uint8)
Y = Colombia_airtemp.lag2_excess.to_numpy() 
T = Colombia_airtemp.Temp.to_numpy()
W = Colombia_airtemp[['SOI', 'SST3', 'SST4', 'SST34', 'SST12', 'NATL', 'SATL', 
                      'TROP', 'forest', 'Year', 'Month']].to_numpy().reshape(-1, 11)
X = Colombia_airtemp[['forest']].to_numpy()

estimate_airtemp = DML(model_y=reg1(), model_t=reg2(), model_final=LassoCV(), discrete_treatment=False,
                       featurizer=PolynomialFeatures(degree=3),
                       linear_first_stages=False, cv=3, random_state=999)

estimate_airtemp.fit(Y=Y, T=T, X=X, W=W, inference='bootstrap')  

# predict effect for each sample X
estimate_airtemp.effect(X)

# ate
ate_AirTemp = estimate_airtemp.ate(X) 
print(ate_AirTemp)

# confidence interval of ate
ci_AirTemp = estimate_airtemp.ate_interval(X) 
print(ci_AirTemp)

# Set values in the df_ATE_lag2
df_ATE_lag2.at[0, 'ATE_lag2'] = ate_AirTemp  
df_ATE_lag2.at[0, '95% CI'] = ci_AirTemp  
print(df_ATE_lag2)




#%%#

##Soil_temperature lag2

#Step 3: Fit the model
Colombia_soiltemp['Soil_temp'] = Colombia_soiltemp['Soil_temp'].astype(np.uint8)
Colombia_soiltemp.Soil_temp = stats.zscore(Colombia_soiltemp.Soil_temp, nan_policy='omit') 
Colombia_soiltemp['Soil_temp'].std() 
Colombia_soiltemp['forest'] = Colombia_soiltemp['forest'].astype(np.uint8)
Y = Colombia_soiltemp.lag2_excess.to_numpy() 
T = Colombia_soiltemp.Soil_temp.to_numpy()
W = Colombia_soiltemp[['SOI', 'SST3', 'SST4', 'SST34', 'SST12', 'NATL', 'SATL', 
                      'TROP', 'forest', 'Year', 'Month']].to_numpy().reshape(-1, 11)
X = Colombia_soiltemp[['forest']].to_numpy()

estimate_Soil_temp = DML(model_y=reg1(), model_t=reg2(), model_final=LassoCV(), discrete_treatment=False,
                       featurizer=PolynomialFeatures(degree=3),
                       linear_first_stages=False, cv=3, random_state=999)

estimate_Soil_temp.fit(Y=Y, T=T, X=X, W=W, inference='bootstrap')  

# predict effect for each sample X
estimate_Soil_temp.effect(X)

# ate
ate_Soil_temp = estimate_Soil_temp.ate(X) 
print(ate_Soil_temp)

# confidence interval of ate
ci_Soil_temp = estimate_Soil_temp.ate_interval(X) 
print(ci_Soil_temp)

# Set values in the df_ATE_lag2
df_ATE_lag2.at[1, 'ATE_lag2'] = ate_Soil_temp  
df_ATE_lag2.at[1, '95% CI'] = ci_Soil_temp  
print(df_ATE_lag2)



#%%#

##Rainfall lag2

#Step 3: Fit the model
Colombia_rain['Rain'] = Colombia_rain['Rain'].astype(np.uint8)
Colombia_rain.Rain = stats.zscore(Colombia_rain.Rain, nan_policy='omit') 
Colombia_rain['Rain'].std() 
Colombia_rain['forest'] = Colombia_rain['forest'].astype(np.uint8)
Y = Colombia_rain.lag2_excess.to_numpy() 
T = Colombia_rain.Rain.to_numpy()
W = Colombia_rain[['SOI', 'SST3', 'SST4', 'SST34', 'SST12', 'NATL', 'SATL', 
                      'TROP', 'forest', 'Year', 'Month']].to_numpy().reshape(-1, 11)
X = Colombia_rain[['forest']].to_numpy()

estimate_Rain = DML(model_y=reg1(), model_t=reg2(), model_final=LassoCV(), discrete_treatment=False,
                       featurizer=PolynomialFeatures(degree=3),
                       linear_first_stages=False, cv=3, random_state=999)

estimate_Rain.fit(Y=Y, T=T, X=X, W=W, inference='bootstrap')  

# predict effect for each sample X
estimate_Rain.effect(X)

# ate
ate_Rain = estimate_Rain.ate(X) 
print(ate_Rain)

# confidence interval of ate
ci_Rain = estimate_Rain.ate_interval(X) 
print(ci_Rain)

# Set values in the df_ATE_lag2
df_ATE_lag2.at[2, 'ATE_lag2'] = ate_Rain  
df_ATE_lag2.at[2, '95% CI'] = ci_Rain  
print(df_ATE_lag2)



#%%#

##Runoff lag2

#Step 3: Fit the model
Colombia_runoff['Qs'] = Colombia_runoff['Qs'].astype(np.uint8)
Colombia_runoff.Qs = stats.zscore(Colombia_runoff.Qs, nan_policy='omit') 
Colombia_runoff['Qs'].std() 
Colombia_runoff['forest'] = Colombia_runoff['forest'].astype(np.uint8)
Y = Colombia_runoff.lag2_excess.to_numpy() 
T = Colombia_runoff.Qs.to_numpy()
W = Colombia_runoff[['SOI', 'SST3', 'SST4', 'SST34', 'SST12', 'NATL', 'SATL', 
                      'TROP', 'forest', 'Year', 'Month']].to_numpy().reshape(-1, 11)
X = Colombia_runoff[['forest']].to_numpy()

estimate_Qs = DML(model_y=reg1(), model_t=reg2(), model_final=LassoCV(), discrete_treatment=False,
                       featurizer=PolynomialFeatures(degree=3),
                       linear_first_stages=False, cv=3, random_state=999)

estimate_Qs.fit(Y=Y, T=T, X=X, W=W, inference='bootstrap')  

# predict effect for each sample X
estimate_Qs.effect(X)

# ate
ate_Qs = estimate_Qs.ate(X) 
print(ate_Qs)

# confidence interval of ate
ci_Qs = estimate_Qs.ate_interval(X) 
print(ci_Qs)

# Set values in the df_ATE_lag2
df_ATE_lag2.at[3, 'ATE_lag2'] = ate_Qs  
df_ATE_lag2.at[3, '95% CI'] = ci_Qs  
print(df_ATE_lag2)



#%%#
##Soil Moisture lag2

#Step 3: Fit the model
Colombia_soilmoist['Soil_mois'] = Colombia_soilmoist['Soil_mois'].astype(np.uint8)
Colombia_soilmoist.Soil_mois = stats.zscore(Colombia_soilmoist.Soil_mois, nan_policy='omit') 
Colombia_soilmoist['Soil_mois'].std() 
Colombia_soilmoist['forest'] = Colombia_soilmoist['forest'].astype(np.uint8)
Y = Colombia_soilmoist.lag2_excess.to_numpy() 
T = Colombia_soilmoist.Soil_mois.to_numpy()
W = Colombia_soilmoist[['SOI', 'SST3', 'SST4', 'SST34', 'SST12', 'NATL', 'SATL', 
                      'TROP', 'forest', 'Year', 'Month']].to_numpy().reshape(-1, 11)
X = Colombia_soilmoist[['forest']].to_numpy()

estimate_Soil_mois = DML(model_y=reg1(), model_t=reg2(), model_final=LassoCV(), discrete_treatment=False,
                       featurizer=PolynomialFeatures(degree=3),
                       linear_first_stages=False, cv=3, random_state=999)

estimate_Soil_mois.fit(Y=Y, T=T, X=X, W=W, inference='bootstrap')  

# predict effect for each sample X
estimate_Soil_mois.effect(X)

# ate
ate_Soil_mois = estimate_Soil_mois.ate(X) 
print(ate_Soil_mois)

# confidence interval of ate
ci_Soil_mois = estimate_Soil_mois.ate_interval(X) 
print(ci_Soil_mois)

# Set values in the df_ATE_lag2
df_ATE_lag2.at[4, 'ATE_lag2'] = ate_Soil_mois  
df_ATE_lag2.at[4, '95% CI'] = ci_Soil_mois  
print(df_ATE_lag2)



#%%#
##EVI lag2

#Step 3: Fit the model
Colombia_EVI['EVI'] = Colombia_EVI['EVI'].astype(np.uint8)
Colombia_EVI.EVI = stats.zscore(Colombia_EVI.EVI, nan_policy='omit') 
Colombia_EVI['EVI'].std() 
Colombia_EVI['forest'] = Colombia_EVI['forest'].astype(np.uint8)
Y = Colombia_EVI.lag2_excess.to_numpy() 
T = Colombia_EVI.EVI.to_numpy()
W = Colombia_EVI[['SOI', 'SST3', 'SST4', 'SST34', 'SST12', 'NATL', 'SATL', 
                      'TROP', 'forest', 'Year', 'Month']].to_numpy().reshape(-1, 11)
X = Colombia_EVI[['forest']].to_numpy()

estimate_EVI = DML(model_y=reg1(), model_t=reg2(), model_final=LassoCV(), discrete_treatment=False,
                       featurizer=PolynomialFeatures(degree=3),
                       linear_first_stages=False, cv=3, random_state=999)

estimate_EVI.fit(Y=Y, T=T, X=X, W=W, inference='bootstrap')  

# predict effect for each sample X
estimate_EVI.effect(X)

# ate
ate_EVI = estimate_EVI.ate(X) 
print(ate_EVI)

# confidence interval of ate
ci_EVI = estimate_EVI.ate_interval(X) 
print(ci_EVI)

# Set values in the df_ATE_lag2
df_ATE_lag2.at[5, 'ATE_lag2'] = ate_EVI  
df_ATE_lag2.at[5, '95% CI'] = ci_EVI  
print(df_ATE_lag2)



#%%#

#Figure 3C
labs = ['Air Temperature',
        'Soil Temperature',
        'Rainfall',
        'Runoff',
        'Soil Moisture',
        'EVI']

measure = [0.069, 0.068, -0.002, -0.008, 0.078, -0.001]
lower =   [0.066, 0.066, -0.005, -0.022, 0.074, -0.004]
upper =   [0.072, '0.070', 0.001, 0.006, 0.082, 0.002]

p = EffectMeasurePlot(label=labs, effect_measure=measure, lcl=lower, ucl=upper)
p.labels(center=0)
p.colors(pointcolor='r') 
p.labels(effectmeasure='ATE')  
p.plot(figsize=(11, 5.5), t_adjuster=0.075, max_value=0.1, min_value=-0.1)
plt.tight_layout








#%%#
#lag3
# Create data frame of ATE results
df_ATE_lag3 = pd.DataFrame(0, index=range(0, 6), columns=['ATE_lag3', '95% CI'])

# Convert the first column to numpy.float64
df_ATE_lag3['ATE_lag3'] = df_ATE_lag3['ATE_lag3'].astype(np.float64)

# Convert the second column to tuples of zeros
df_ATE_lag3['95% CI'] = [(0.0, 0.0)] * 6

# Display the DataFrame
print(df_ATE_lag3)



#%%#

##Air Temperature lag3

#Step 3: Fit the model
Colombia_airtemp['Temp'] = Colombia_airtemp['Temp'].astype(np.uint8)
Colombia_airtemp.Temp = stats.zscore(Colombia_airtemp.Temp, nan_policy='omit') 
Colombia_airtemp['Temp'].std() 
Colombia_airtemp['forest'] = Colombia_airtemp['forest'].astype(np.uint8)
Y = Colombia_airtemp.lag3_excess.to_numpy() 
T = Colombia_airtemp.Temp.to_numpy()
W = Colombia_airtemp[['SOI', 'SST3', 'SST4', 'SST34', 'SST12', 'NATL', 'SATL', 
                      'TROP', 'forest', 'Year', 'Month']].to_numpy().reshape(-1, 11)
X = Colombia_airtemp[['forest']].to_numpy()

estimate_airtemp = DML(model_y=reg1(), model_t=reg2(), model_final=LassoCV(), discrete_treatment=False,
                       featurizer=PolynomialFeatures(degree=3),
                       linear_first_stages=False, cv=3, random_state=999)

estimate_airtemp.fit(Y=Y, T=T, X=X, W=W, inference='bootstrap')  

# predict effect for each sample X
estimate_airtemp.effect(X)

# ate
ate_AirTemp = estimate_airtemp.ate(X) 
print(ate_AirTemp)

# confidence interval of ate
ci_AirTemp = estimate_airtemp.ate_interval(X) 
print(ci_AirTemp)

# Set values in the df_ATE_lag3
df_ATE_lag3.at[0, 'ATE_lag3'] = ate_AirTemp  
df_ATE_lag3.at[0, '95% CI'] = ci_AirTemp  
print(df_ATE_lag3)




#%%#

##Soil_temperature lag3

#Step 3: Fit the model
Colombia_soiltemp['Soil_temp'] = Colombia_soiltemp['Soil_temp'].astype(np.uint8)
Colombia_soiltemp.Soil_temp = stats.zscore(Colombia_soiltemp.Soil_temp, nan_policy='omit') 
Colombia_soiltemp['Soil_temp'].std() 
Colombia_soiltemp['forest'] = Colombia_soiltemp['forest'].astype(np.uint8)
Y = Colombia_soiltemp.lag3_excess.to_numpy() 
T = Colombia_soiltemp.Soil_temp.to_numpy()
W = Colombia_soiltemp[['SOI', 'SST3', 'SST4', 'SST34', 'SST12', 'NATL', 'SATL', 
                      'TROP', 'forest', 'Year', 'Month']].to_numpy().reshape(-1, 11)
X = Colombia_soiltemp[['forest']].to_numpy()

estimate_Soil_temp = DML(model_y=reg1(), model_t=reg2(), model_final=LassoCV(), discrete_treatment=False,
                       featurizer=PolynomialFeatures(degree=3),
                       linear_first_stages=False, cv=3, random_state=999)

estimate_Soil_temp.fit(Y=Y, T=T, X=X, W=W, inference='bootstrap')  

# predict effect for each sample X
estimate_Soil_temp.effect(X)

# ate
ate_Soil_temp = estimate_Soil_temp.ate(X) 
print(ate_Soil_temp)

# confidence interval of ate
ci_Soil_temp = estimate_Soil_temp.ate_interval(X) 
print(ci_Soil_temp)

# Set values in the df_ATE_lag3
df_ATE_lag3.at[1, 'ATE_lag3'] = ate_Soil_temp  
df_ATE_lag3.at[1, '95% CI'] = ci_Soil_temp  
print(df_ATE_lag3)



#%%#

##Rainfall lag3

#Step 3: Fit the model
Colombia_rain['Rain'] = Colombia_rain['Rain'].astype(np.uint8)
Colombia_rain.Rain = stats.zscore(Colombia_rain.Rain, nan_policy='omit') 
Colombia_rain['Rain'].std() 
Colombia_rain['forest'] = Colombia_rain['forest'].astype(np.uint8)
Y = Colombia_rain.lag3_excess.to_numpy() 
T = Colombia_rain.Rain.to_numpy()
W = Colombia_rain[['SOI', 'SST3', 'SST4', 'SST34', 'SST12', 'NATL', 'SATL', 
                      'TROP', 'forest', 'Year', 'Month']].to_numpy().reshape(-1, 11)
X = Colombia_rain[['forest']].to_numpy()

estimate_Rain = DML(model_y=reg1(), model_t=reg2(), model_final=LassoCV(), discrete_treatment=False,
                       featurizer=PolynomialFeatures(degree=3),
                       linear_first_stages=False, cv=3, random_state=999)

estimate_Rain.fit(Y=Y, T=T, X=X, W=W, inference='bootstrap')  

# predict effect for each sample X
estimate_Rain.effect(X)

# ate
ate_Rain = estimate_Rain.ate(X) 
print(ate_Rain)

# confidence interval of ate
ci_Rain = estimate_Rain.ate_interval(X) 
print(ci_Rain)

# Set values in the df_ATE_lag3
df_ATE_lag3.at[2, 'ATE_lag3'] = ate_Rain  
df_ATE_lag3.at[2, '95% CI'] = ci_Rain  
print(df_ATE_lag3)



#%%#

##Runoff lag3

#Step 3: Fit the model
Colombia_runoff['Qs'] = Colombia_runoff['Qs'].astype(np.uint8)
Colombia_runoff.Qs = stats.zscore(Colombia_runoff.Qs, nan_policy='omit') 
Colombia_runoff['Qs'].std() 
Colombia_runoff['forest'] = Colombia_runoff['forest'].astype(np.uint8)
Y = Colombia_runoff.lag3_excess.to_numpy() 
T = Colombia_runoff.Qs.to_numpy()
W = Colombia_runoff[['SOI', 'SST3', 'SST4', 'SST34', 'SST12', 'NATL', 'SATL', 
                      'TROP', 'forest', 'Year', 'Month']].to_numpy().reshape(-1, 11)
X = Colombia_runoff[['forest']].to_numpy()

estimate_Qs = DML(model_y=reg1(), model_t=reg2(), model_final=LassoCV(), discrete_treatment=False,
                       featurizer=PolynomialFeatures(degree=3),
                       linear_first_stages=False, cv=3, random_state=999)

estimate_Qs.fit(Y=Y, T=T, X=X, W=W, inference='bootstrap')  

# predict effect for each sample X
estimate_Qs.effect(X)

# ate
ate_Qs = estimate_Qs.ate(X) 
print(ate_Qs)

# confidence interval of ate
ci_Qs = estimate_Qs.ate_interval(X) 
print(ci_Qs)

# Set values in the df_ATE_lag3
df_ATE_lag3.at[3, 'ATE_lag3'] = ate_Qs  
df_ATE_lag3.at[3, '95% CI'] = ci_Qs  
print(df_ATE_lag3)



#%%#
##Soil Moisture lag3

#Step 3: Fit the model
Colombia_soilmoist['Soil_mois'] = Colombia_soilmoist['Soil_mois'].astype(np.uint8)
Colombia_soilmoist.Soil_mois = stats.zscore(Colombia_soilmoist.Soil_mois, nan_policy='omit') 
Colombia_soilmoist['Soil_mois'].std() 
Colombia_soilmoist['forest'] = Colombia_soilmoist['forest'].astype(np.uint8)
Y = Colombia_soilmoist.lag3_excess.to_numpy() 
T = Colombia_soilmoist.Soil_mois.to_numpy()
W = Colombia_soilmoist[['SOI', 'SST3', 'SST4', 'SST34', 'SST12', 'NATL', 'SATL', 
                      'TROP', 'forest', 'Year', 'Month']].to_numpy().reshape(-1, 11)
X = Colombia_soilmoist[['forest']].to_numpy()

estimate_Soil_mois = DML(model_y=reg1(), model_t=reg2(), model_final=LassoCV(), discrete_treatment=False,
                       featurizer=PolynomialFeatures(degree=3),
                       linear_first_stages=False, cv=3, random_state=999)

estimate_Soil_mois.fit(Y=Y, T=T, X=X, W=W, inference='bootstrap')  

# predict effect for each sample X
estimate_Soil_mois.effect(X)

# ate
ate_Soil_mois = estimate_Soil_mois.ate(X) 
print(ate_Soil_mois)

# confidence interval of ate
ci_Soil_mois = estimate_Soil_mois.ate_interval(X) 
print(ci_Soil_mois)

# Set values in the df_ATE_lag3
df_ATE_lag3.at[4, 'ATE_lag3'] = ate_Soil_mois  
df_ATE_lag3.at[4, '95% CI'] = ci_Soil_mois  
print(df_ATE_lag3)



#%%#
##EVI lag3

#Step 3: Fit the model
Colombia_EVI['EVI'] = Colombia_EVI['EVI'].astype(np.uint8)
Colombia_EVI.EVI = stats.zscore(Colombia_EVI.EVI, nan_policy='omit') 
Colombia_EVI['EVI'].std() 
Colombia_EVI['forest'] = Colombia_EVI['forest'].astype(np.uint8)
Y = Colombia_EVI.lag3_excess.to_numpy() 
T = Colombia_EVI.EVI.to_numpy()
W = Colombia_EVI[['SOI', 'SST3', 'SST4', 'SST34', 'SST12', 'NATL', 'SATL', 
                      'TROP', 'forest', 'Year', 'Month']].to_numpy().reshape(-1, 11)
X = Colombia_EVI[['forest']].to_numpy()

estimate_EVI = DML(model_y=reg1(), model_t=reg2(), model_final=LassoCV(), discrete_treatment=False,
                       featurizer=PolynomialFeatures(degree=3),
                       linear_first_stages=False, cv=3, random_state=999)

estimate_EVI.fit(Y=Y, T=T, X=X, W=W, inference='bootstrap')  

# predict effect for each sample X
estimate_EVI.effect(X)

# ate
ate_EVI = estimate_EVI.ate(X) 
print(ate_EVI)

# confidence interval of ate
ci_EVI = estimate_EVI.ate_interval(X) 
print(ci_EVI)

# Set values in the df_ATE_lag3
df_ATE_lag3.at[5, 'ATE_lag3'] = ate_EVI  
df_ATE_lag3.at[5, '95% CI'] = ci_EVI  
print(df_ATE_lag3)



#%%#

#Figure 3D
labs = ['Air Temperature',
        'Soil Temperature',
        'Rainfall',
        'Runoff',
        'Soil Moisture',
        'EVI']

measure = [0.069, 0.068, '0.000', '0.000', '0.080', '0.000']
lower =   [0.066, 0.066, -0.003, -0.013, 0.077, -0.002]
upper =   [0.071, 0.071, 0.003, 0.013, 0.083, 0.002]

p = EffectMeasurePlot(label=labs, effect_measure=measure, lcl=lower, ucl=upper)
p.labels(center=0)
p.colors(pointcolor='r') 
p.labels(effectmeasure='ATE')  
p.plot(figsize=(11, 5.5), t_adjuster=0.075, max_value=0.1, min_value=-0.1)
plt.tight_layout