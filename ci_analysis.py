###############################################################################
# Code for the paper:
#   "Causal association of environmental variables on the occurrence of excess_casess of 
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
from sklearn.ensemble import GradientBoostingRegressor, GradientBoostingClassifier
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
from econml.dml import KernelDML



# Set seeds to make the results more reproducible
def seed_everything(seed=123):
    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    os.environ['TF_DETERMINISTIC_OPS'] = '1'

seed = 123
seed_everything(seed)
warnings.filterwarnings('ignore')
pd.set_option('display.float_format', lambda x: '%.2f' % x)




#import data
data_col = pd.read_csv("https://raw.githubusercontent.com/juandavidgutier/causal_inference_cutaneous_leishmaniasis/main/dataset_leish.csv", encoding='latin-1') 

#z-score
data_col.SST3 = stats.zscore(data_col.SST3, nan_policy='omit') 
data_col.SST4 = stats.zscore(data_col.SST4, nan_policy='omit')
data_col.SST34 = stats.zscore(data_col.SST34, nan_policy='omit') 
data_col.SST12 = stats.zscore(data_col.SST12, nan_policy='omit') 
data_col.Equatorial_SOI = stats.zscore(data_col.Equatorial_SOI, nan_policy='omit')
data_col.SOI = stats.zscore(data_col.SOI, nan_policy='omit') 
data_col.NATL = stats.zscore(data_col.NATL, nan_policy='omit')
data_col.SATL = stats.zscore(data_col.SATL, nan_policy='omit')  
data_col.TROP = stats.zscore(data_col.TROP, nan_policy='omit')
data_col.forest_percent = stats.zscore(data_col.forest_percent, nan_policy='omit')


data_col = data_col.dropna()


#temperature
Colombia_temp = data_col[['excess_cases', 'Temperature', 'SOI', 'Equatorial_SOI', 'SST3', 'SST4', 'SST34', 'SST12', 'NATL', 'SATL', 'TROP', 'forest_percent']] 
#soil temperature
Colombia_soiltemp = data_col[['excess_cases', 'SoilTMP0_10cm', 'SOI', 'Equatorial_SOI', 'SST3', 'SST4', 'SST34', 'SST12', 'NATL', 'SATL', 'TROP', 'forest_percent']] 
#rainfall
Colombia_rainfall = data_col[['excess_cases', 'Rainfall', 'SOI', 'Equatorial_SOI', 'SST3', 'SST4', 'SST34', 'SST12', 'NATL', 'SATL', 'TROP', 'forest_percent']] 
#runoff
Colombia_runoff = data_col[['excess_cases', 'Qs', 'SOI', 'Equatorial_SOI', 'SST3', 'SST4', 'SST34', 'SST12', 'NATL', 'SATL', 'TROP', 'forest_percent']] 
#soil moisture
Colombia_soilmoisture = data_col[['excess_cases', 'SoilMoi0_10cm', 'SOI', 'Equatorial_SOI', 'SST3', 'SST4', 'SST34', 'SST12', 'NATL', 'SATL', 'TROP', 'forest_percent']] 
#EVI
Colombia_EVI = data_col[['excess_cases', 'EVI', 'SOI', 'Equatorial_SOI', 'SST3', 'SST4', 'SST34', 'SST12', 'NATL', 'SATL', 'TROP', 'forest_percent']] 


###############################################################################
#####Temperature

Y = Colombia_temp.excess_cases.to_numpy() #Y = data_card['incidencia100k_cardiovasculares'].values
T = Colombia_temp.Temperature.to_numpy()
W = Colombia_temp[['SOI', 'Equatorial_SOI', 'SST3', 'SST4', 'SST34', 'SST12', 'NATL', 'SATL', 'TROP', 'forest_percent']].to_numpy().reshape(-1, 10)
X = Colombia_temp[['forest_percent']].to_numpy()


#Model Selection for Causal Effect Model with the RScorer
# A multitude of possible approaches for CATE estimation under conditional exogeneity


X_train, X_val, T_train, T_val, Y_train, Y_val, W_train, W_val = train_test_split(X, T, Y, W, test_size=.4)

#%load_ext bootstrapreload
#%bootstrapreload 2

## Ignore warnings
warnings.filterwarnings('ignore') 

reg1 = lambda: GradientBoostingClassifier(n_estimators=2000, random_state=123)
reg2 = lambda: GradientBoostingRegressor(n_estimators=2000, random_state=123)

models = [
        ('ldml', LinearDML(model_y=reg1(), model_t=reg2(), discrete_treatment=False,
                             linear_first_stages=False, cv=3, random_state=123)),
        ('sldml', SparseLinearDML(model_y=reg1(), model_t=reg2(), discrete_treatment=False,
                                    featurizer=PolynomialFeatures(degree=2, include_bias=False),
                                    linear_first_stages=False, cv=3, random_state=123)),
        ('dml', DML(model_y=reg1(), model_t=reg2(), model_final=LassoCV(), discrete_treatment=False,
                             featurizer=PolynomialFeatures(degree=3),
                             linear_first_stages=False, cv=3, random_state=123)),
        ('forest', CausalForestDML(model_y=reg1(), model_t=reg2(), featurizer=PolynomialFeatures(degree=3), 
                             discrete_treatment=False, cv=3, random_state=123)),
        ('kernel', KernelDML(model_y=reg1(), model_t=reg2(),
                             cv=3, random_state=123)),
                 
         ]


def fit_model(name, model):
    return name, model.fit(Y_train, T_train, X=X_train, W=W_train)

models = Parallel(n_jobs=-1, verbose=1, backend="threading")(delayed(fit_model)(name, mdl) for name, mdl in models)

#Choose model with highest RScore
scorer = RScorer(model_y=reg1(), model_t=reg2(),
                 discrete_treatment=False, cv=3,
                 mc_iters=3, mc_agg='median')

scorer.fit(Y_val, T_val, X=X_val, W=W_val)


rscore = [scorer.score(mdl) for _, mdl in models]
print(rscore)#best model CausalForestDML


#Step 1: Modeling the causal mechanism
model_leish=CausalModel(
        data = Colombia_temp,
        treatment=['Temperature'],
        outcome=['excess_cases'],
        graph= """graph[directed 1 node[id "Temperature" label "Temperature"]
                    node[id "excess_cases" label "excess_cases"]
                    node[id "SOI" label "SOI"]
                    node[id "Equatorial_SOI" label "Equatorial_SOI"]
                    node[id "SST3" label "SST3"]               
                    node[id "SST4" label "SST4"]
                    node[id "SST34" label "SST34"]
                    node[id "SST12" label "SST12"]
                    node[id "NATL" label "NATL"]
                    node[id "SATL" label "SATL"]
                    node[id "TROP" label "TROP"]                                      
                    node[id "forest_percent" label "forest_percent"]
                                       
                    edge[source "SOI" target "Temperature"]
                    edge[source "SOI" target "excess_cases"]
                    
                    edge[source "Equatorial_SOI" target "Temperature"]
                    edge[source "Equatorial_SOI" target "excess_cases"]
                    
                    edge[source "SST3" target "Temperature"]
                    edge[source "SST3" target "excess_cases"]
                    
                    edge[source "SST4" target "Temperature"]
                    edge[source "SST4" target "excess_cases"]
                    
                    edge[source "SST34" target "Temperature"]
                    edge[source "SST34" target "excess_cases"]
                    
                    edge[source "SST12" target "Temperature"]
                    edge[source "SST12" target "excess_cases"]
                    
                    edge[source "NATL" target "Temperature"]
                    edge[source "NATL" target "excess_cases"]
                    
                    edge[source "SATL" target "Temperature"]
                    edge[source "SATL" target "excess_cases"]
                    
                    edge[source "TROP" target "Temperature"]
                    edge[source "TROP" target "excess_cases"]
                                                      
                    edge[source "forest_percent" target "Temperature"]
                    edge[source "forest_percent" target "excess_cases"]
                    
                    edge[source "SOI" target "Equatorial_SOI"]
                    edge[source "SOI" target "SST3"]
                    edge[source "SOI" target "SST4"]
                    edge[source "SOI" target "SST34"]
                    edge[source "SOI" target "SST12"]
                    edge[source "SOI" target "NATL"]
                    edge[source "SOI" target "SATL"]
                    edge[source "SOI" target "TROP"]
                    
                    edge[source "Equatorial_SOI" target "SST3"]
                    edge[source "Equatorial_SOI" target "SST4"]
                    edge[source "Equatorial_SOI" target "SST34"]
                    edge[source "Equatorial_SOI" target "SST12"]
                    edge[source "Equatorial_SOI" target "NATL"]
                    edge[source "Equatorial_SOI" target "SATL"]
                    edge[source "Equatorial_SOI" target "TROP"]
                    
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
                    
                    edge[source "Temperature" target "excess_cases"]]"""
                    )
    
#view model 
#model_leish.view_model()

#Step 2: Identifying effects
identified_estimand_temp = model_leish.identify_effect(proceed_when_unidentifiable=False)
print(identified_estimand_temp)

estimate_temp = CausalForestDML(model_y=reg1(), model_t=reg2(), featurizer=PolynomialFeatures(degree=3), 
                             discrete_treatment=False, cv=3, random_state=123)

estimate_temp = estimate_temp.dowhy

# fit the model
estimate_temp.fit(Y=Y, T=T, X=X, W=W, inference='blb')  

# predict effect for each sample X
estimate_temp.effect(X)

# ate
ate_Temperature = estimate_temp.ate(X) 
print(ate_Temperature)

# confidence interval of ate
ci_Temperature = estimate_temp.ate_interval(X) 
print(ci_Temperature)


#Step 4: Refute the effect
#with random common cause
random_Temperature = estimate_temp.refute_estimate(method_name="random_common_cause", random_state=123)
print(random_Temperature)

#with replace a random subset of the data
subset_Temperature = estimate_temp.refute_estimate(method_name="data_subset_refuter", subset_fraction=0.9, num_simulations=3)
print(subset_Temperature)

#with placebo 
placebo_Temperature = estimate_temp.refute_estimate(method_name="placebo_treatment_refuter", placebo_type="permute", num_simulations=3)
print(placebo_Temperature)





#####soil Temperature

Y = Colombia_soiltemp.excess_cases.to_numpy() #Y = data_card['incidencia100k_cardiovasculares'].values
T = Colombia_soiltemp.SoilTMP0_10cm.to_numpy()
W = Colombia_soiltemp[['SOI', 'Equatorial_SOI', 'SST3', 'SST4', 'SST34', 'SST12', 'NATL', 'SATL', 'TROP', 'forest_percent']].to_numpy().reshape(-1, 10)
X = Colombia_soiltemp[['forest_percent']].to_numpy()


#Model Selection for Causal Effect Model with the RScorer
# A multitude of possible approaches for CATE estimation under conditional exogeneity
X_train, X_val, T_train, T_val, Y_train, Y_val, W_train, W_val = train_test_split(X, T, Y, W, test_size=.4)


models = [
        ('ldml', LinearDML(model_y=reg1(), model_t=reg2(), discrete_treatment=False,
                             linear_first_stages=False, cv=3, random_state=123)),
        ('sldml', SparseLinearDML(model_y=reg1(), model_t=reg2(), discrete_treatment=False,
                                    featurizer=PolynomialFeatures(degree=2, include_bias=False),
                                    linear_first_stages=False, cv=3, random_state=123)),
        ('dml', DML(model_y=reg1(), model_t=reg2(), model_final=LassoCV(), discrete_treatment=False,
                             featurizer=PolynomialFeatures(degree=3),
                             linear_first_stages=False, cv=3, random_state=123)),
        ('forest', CausalForestDML(model_y=reg1(), model_t=reg2(), featurizer=PolynomialFeatures(degree=3), 
                             discrete_treatment=False, cv=3, random_state=123)),
        ('kernel', KernelDML(model_y=reg1(), model_t=reg2(),
                             cv=3, random_state=123)),
                
         ]

def fit_model(name, model):
    return name, model.fit(Y_train, T_train, X=X_train, W=W_train)

models = Parallel(n_jobs=-1, verbose=1, backend="threading")(delayed(fit_model)(name, mdl) for name, mdl in models)


#Choose model with highest RScore
scorer = RScorer(model_y=reg1(), model_t=reg2(),
                 discrete_treatment=False, cv=3,
                 mc_iters=3, mc_agg='median')

scorer.fit(Y_val, T_val, X=X_val, W=W_val)


rscore = [scorer.score(mdl) for _, mdl in models]
print(rscore)#best model DML


#Step 1: Modeling the causal mechanism
model_leish=CausalModel(
        data = Colombia_soiltemp,
        treatment=['SoilTMP0_10cm'],
        outcome=['excess_cases'],
        graph= """graph[directed 1 node[id "SoilTMP0_10cm" label "SoilTMP0_10cm"]
                    node[id "excess_cases" label "excess_cases"]
                    node[id "SOI" label "SOI"]
                    node[id "Equatorial_SOI" label "Equatorial_SOI"]
                    node[id "SST3" label "SST3"]               
                    node[id "SST4" label "SST4"]
                    node[id "SST34" label "SST34"]
                    node[id "SST12" label "SST12"]
                    node[id "NATL" label "NATL"]
                    node[id "SATL" label "SATL"]
                    node[id "TROP" label "TROP"]                                      
                    node[id "forest_percent" label "forest_percent"]
                                       
                    edge[source "SOI" target "Temperature"]
                    edge[source "SOI" target "excess_cases"]
                    
                    edge[source "Equatorial_SOI" target "Temperature"]
                    edge[source "Equatorial_SOI" target "excess_cases"]
                    
                    edge[source "SST3" target "Temperature"]
                    edge[source "SST3" target "excess_cases"]
                    
                    edge[source "SST4" target "Temperature"]
                    edge[source "SST4" target "excess_cases"]
                    
                    edge[source "SST34" target "Temperature"]
                    edge[source "SST34" target "excess_cases"]
                    
                    edge[source "SST12" target "Temperature"]
                    edge[source "SST12" target "excess_cases"]
                    
                    edge[source "NATL" target "Temperature"]
                    edge[source "NATL" target "excess_cases"]
                    
                    edge[source "SATL" target "Temperature"]
                    edge[source "SATL" target "excess_cases"]
                    
                    edge[source "TROP" target "Temperature"]
                    edge[source "TROP" target "excess_cases"]
                                                      
                    edge[source "forest_percent" target "Temperature"]
                    edge[source "forest_percent" target "excess_cases"]
                    
                    edge[source "SOI" target "Equatorial_SOI"]
                    edge[source "SOI" target "SST3"]
                    edge[source "SOI" target "SST4"]
                    edge[source "SOI" target "SST34"]
                    edge[source "SOI" target "SST12"]
                    edge[source "SOI" target "NATL"]
                    edge[source "SOI" target "SATL"]
                    edge[source "SOI" target "TROP"]
                    
                    edge[source "Equatorial_SOI" target "SST3"]
                    edge[source "Equatorial_SOI" target "SST4"]
                    edge[source "Equatorial_SOI" target "SST34"]
                    edge[source "Equatorial_SOI" target "SST12"]
                    edge[source "Equatorial_SOI" target "NATL"]
                    edge[source "Equatorial_SOI" target "SATL"]
                    edge[source "Equatorial_SOI" target "TROP"]
                    
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
                    
                    edge[source "SoilTMP0_10cm" target "excess_cases"]]"""
                    )
        
#view model 
#model_leish.view_model()    


#Step 2: Identifying effects
identified_estimand_soiltemp = model_leish.identify_effect(proceed_when_unidentifiable=False)
print(identified_estimand_soiltemp)


#Step 3: Estimation of the effect
estimate_soiltemp = DML(model_y=reg1(), model_t=reg2(), model_final=LassoCV(), discrete_treatment=False,
                        featurizer=PolynomialFeatures(degree=3),
                        linear_first_stages=False, cv=3, random_state=123)

    
    

estimate_soiltemp = estimate_soiltemp.dowhy

# fit the model
estimate_soiltemp.fit(Y=Y, T=T, X=X, W=W, inference='bootstrap')  

# predict effect for each sample X
estimate_soiltemp.effect(X)

# ate
ate_SoilTemperature = estimate_soiltemp.ate(X) 
print(ate_SoilTemperature)

# confidence interval of ate
ci_SoilTemperature = estimate_soiltemp.ate_interval(X) 
print(ci_SoilTemperature)

#Step 4: Refute the effect
#with random common cause
random_SoilTemperature = estimate_soiltemp.refute_estimate(method_name="random_common_cause", random_state=123)
print(random_SoilTemperature)

#with replace a random subset of the data
subset_SoilTemperature = estimate_soiltemp.refute_estimate(method_name="data_subset_refuter", subset_fraction=0.9, num_simulations=3)
print(subset_SoilTemperature)

#with placebo 
placebo_SoilTemperature = estimate_soiltemp.refute_estimate(method_name="placebo_treatment_refuter", placebo_type="permute", num_simulations=3)
print(placebo_SoilTemperature)



#####Rainfall

Y = Colombia_rainfall.excess_cases.to_numpy() #Y = data_card['incidencia100k_cardiovasculares'].values
T = Colombia_rainfall.Rainfall.to_numpy()
W = Colombia_rainfall[['SOI', 'Equatorial_SOI', 'SST3', 'SST4', 'SST34', 'SST12', 'NATL', 'SATL', 'TROP', 'forest_percent']].to_numpy().reshape(-1, 10)
X = Colombia_rainfall[['forest_percent']].to_numpy()


#Model Selection for Causal Effect Model with the RScorer
# A multitude of possible approaches for CATE estimation under conditional exogeneity
X_train, X_val, T_train, T_val, Y_train, Y_val, W_train, W_val = train_test_split(X, T, Y, W, test_size=.4)


models = [
        ('ldml', LinearDML(model_y=reg1(), model_t=reg2(), discrete_treatment=False,
                             linear_first_stages=False, cv=3, random_state=123)),
        ('sldml', SparseLinearDML(model_y=reg1(), model_t=reg2(), discrete_treatment=False,
                                    featurizer=PolynomialFeatures(degree=2, include_bias=False),
                                    linear_first_stages=False, cv=3, random_state=123)),
        ('dml', DML(model_y=reg1(), model_t=reg2(), model_final=LassoCV(), discrete_treatment=False,
                             featurizer=PolynomialFeatures(degree=3),
                             linear_first_stages=False, cv=3, random_state=123)),
        ('forest', CausalForestDML(model_y=reg1(), model_t=reg2(), featurizer=PolynomialFeatures(degree=3), 
                             discrete_treatment=False, cv=3, random_state=123)),
        ('kernel', KernelDML(model_y=reg1(), model_t=reg2(),
                             cv=3, random_state=123)),
        
          
         ]


def fit_model(name, model):
    return name, model.fit(Y_train, T_train, X=X_train, W=W_train)

models = Parallel(n_jobs=-1, verbose=1, backend="threading")(delayed(fit_model)(name, mdl) for name, mdl in models)

#Choose model with highest RScore
scorer = RScorer(model_y=reg1(), model_t=reg2(),
                 discrete_treatment=False, cv=3,
                 mc_iters=3, mc_agg='median')

scorer.fit(Y_val, T_val, X=X_val, W=W_val)


rscore = [scorer.score(mdl) for _, mdl in models]
print(rscore)#best model SparseLinearDML


#Step 1: Modeling the causal mechanism
model_leish=CausalModel(
        data = Colombia_rainfall,
        treatment=['Rainfall'],
        outcome=['excess_cases'],
        graph= """graph[directed 1 node[id "Rainfall" label "Rainfall"]
                    node[id "excess_cases" label "excess_cases"]
                    node[id "SOI" label "SOI"]
                    node[id "Equatorial_SOI" label "Equatorial_SOI"]
                    node[id "SST3" label "SST3"]               
                    node[id "SST4" label "SST4"]
                    node[id "SST34" label "SST34"]
                    node[id "SST12" label "SST12"]
                    node[id "NATL" label "NATL"]
                    node[id "SATL" label "SATL"]
                    node[id "TROP" label "TROP"]                                      
                    node[id "forest_percent" label "forest_percent"]
                                       
                    edge[source "SOI" target "Temperature"]
                    edge[source "SOI" target "excess_cases"]
                    
                    edge[source "Equatorial_SOI" target "Temperature"]
                    edge[source "Equatorial_SOI" target "excess_cases"]
                    
                    edge[source "SST3" target "Temperature"]
                    edge[source "SST3" target "excess_cases"]
                    
                    edge[source "SST4" target "Temperature"]
                    edge[source "SST4" target "excess_cases"]
                    
                    edge[source "SST34" target "Temperature"]
                    edge[source "SST34" target "excess_cases"]
                    
                    edge[source "SST12" target "Temperature"]
                    edge[source "SST12" target "excess_cases"]
                    
                    edge[source "NATL" target "Temperature"]
                    edge[source "NATL" target "excess_cases"]
                    
                    edge[source "SATL" target "Temperature"]
                    edge[source "SATL" target "excess_cases"]
                    
                    edge[source "TROP" target "Temperature"]
                    edge[source "TROP" target "excess_cases"]
                                                      
                    edge[source "forest_percent" target "Temperature"]
                    edge[source "forest_percent" target "excess_cases"]
                    
                    edge[source "SOI" target "Equatorial_SOI"]
                    edge[source "SOI" target "SST3"]
                    edge[source "SOI" target "SST4"]
                    edge[source "SOI" target "SST34"]
                    edge[source "SOI" target "SST12"]
                    edge[source "SOI" target "NATL"]
                    edge[source "SOI" target "SATL"]
                    edge[source "SOI" target "TROP"]
                    
                    edge[source "Equatorial_SOI" target "SST3"]
                    edge[source "Equatorial_SOI" target "SST4"]
                    edge[source "Equatorial_SOI" target "SST34"]
                    edge[source "Equatorial_SOI" target "SST12"]
                    edge[source "Equatorial_SOI" target "NATL"]
                    edge[source "Equatorial_SOI" target "SATL"]
                    edge[source "Equatorial_SOI" target "TROP"]
                    
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
                    
                    edge[source "Rainfall" target "excess_cases"]]"""
                    )

#view model 
#model_leish.view_model()        

#Step 2: Identifying effects
identified_estimand_rain = model_leish.identify_effect(proceed_when_unidentifiable=False)
print(identified_estimand_rain)

#Step 3: Estimation of the effect 
estimate_rain = SparseLinearDML(model_y=reg1(), model_t=reg2(), discrete_treatment=False,
                                    featurizer=PolynomialFeatures(degree=2, include_bias=False),
                                    linear_first_stages=False, cv=3, random_state=123)

estimate_rain = estimate_rain.dowhy

# fit the model
estimate_rain.fit(Y=Y, T=T, X=X, W=W, inference='bootstrap')  

# predict effect for each sample X
estimate_rain.effect(X)

# ate
ate_Rain = estimate_rain.ate(X) 
print(ate_Rain)

# confidence interval of ate
ci_Rain = estimate_rain.ate_interval(X) 
print(ci_Rain)

#Step 4: Refute the effect
#with random common cause
random_Rain = estimate_rain.refute_estimate(method_name="random_common_cause", random_state=123)
print(random_Rain)

#with replace a random subset of the data
subset_Rain = estimate_rain.refute_estimate(method_name="data_subset_refuter", subset_fraction=0.9, num_simulations=3)
print(subset_Rain)

#with placebo 
placebo_Rain = estimate_rain.refute_estimate(method_name="placebo_treatment_refuter", placebo_type="permute", num_simulations=3)
print(placebo_Rain)




#####Runoff

Y = Colombia_runoff.excess_cases.to_numpy() #Y = data_card['incidencia100k_cardiovasculares'].values
T = Colombia_runoff.Qs.to_numpy()
W = Colombia_runoff[['SOI', 'Equatorial_SOI', 'SST3', 'SST4', 'SST34', 'SST12', 'NATL', 'SATL', 'TROP', 'forest_percent']].to_numpy().reshape(-1, 10)
X = Colombia_runoff[['forest_percent']].to_numpy()



#Model Selection for Causal Effect Model with the RScorer
# A multitude of possible approaches for CATE estimation under conditional exogeneity
X_train, X_val, T_train, T_val, Y_train, Y_val, W_train, W_val = train_test_split(X, T, Y, W, test_size=.4)

models = [
        ('ldml', LinearDML(model_y=reg1(), model_t=reg2(), discrete_treatment=False,
                             linear_first_stages=False, cv=3, random_state=123)),
        ('sldml', SparseLinearDML(model_y=reg1(), model_t=reg2(), discrete_treatment=False,
                                    featurizer=PolynomialFeatures(degree=2, include_bias=False),
                                    linear_first_stages=False, cv=3, random_state=123)),
        ('dml', DML(model_y=reg1(), model_t=reg2(), model_final=LassoCV(), discrete_treatment=False,
                             featurizer=PolynomialFeatures(degree=3),
                             linear_first_stages=False, cv=3, random_state=123)),
        ('forest', CausalForestDML(model_y=reg1(), model_t=reg2(), featurizer=PolynomialFeatures(degree=3), 
                             discrete_treatment=False, cv=3, random_state=123)),
        ('kernel', KernelDML(model_y=reg1(), model_t=reg2(),
                             cv=3, random_state=123)),
                 
         ]

def fit_model(name, model):
    return name, model.fit(Y_train, T_train, X=X_train, W=W_train)

models = Parallel(n_jobs=-1, verbose=1, backend="threading")(delayed(fit_model)(name, mdl) for name, mdl in models)


#Choose model with highest RScore
scorer = RScorer(model_y=reg1(), model_t=reg2(),
                 discrete_treatment=False, cv=3,
                 mc_iters=3, mc_agg='median')

scorer.fit(Y_val, T_val, X=X_val, W=W_val)


rscore = [scorer.score(mdl) for _, mdl in models]
print(rscore) #best model DML



#Step 1: Modeling the causal mechanism
model_leish=CausalModel(
        data = Colombia_runoff,
        treatment=['Qs'],
        outcome=['excess_cases'],
        graph= """graph[directed 1 node[id "Qs" label "Qs"]
                     node[id "excess_cases" label "excess_cases"]
                    node[id "SOI" label "SOI"]
                    node[id "Equatorial_SOI" label "Equatorial_SOI"]
                    node[id "SST3" label "SST3"]               
                    node[id "SST4" label "SST4"]
                    node[id "SST34" label "SST34"]
                    node[id "SST12" label "SST12"]
                    node[id "NATL" label "NATL"]
                    node[id "SATL" label "SATL"]
                    node[id "TROP" label "TROP"]                                      
                    node[id "forest_percent" label "forest_percent"]
                                       
                    edge[source "SOI" target "Temperature"]
                    edge[source "SOI" target "excess_cases"]
                    
                    edge[source "Equatorial_SOI" target "Temperature"]
                    edge[source "Equatorial_SOI" target "excess_cases"]
                    
                    edge[source "SST3" target "Temperature"]
                    edge[source "SST3" target "excess_cases"]
                    
                    edge[source "SST4" target "Temperature"]
                    edge[source "SST4" target "excess_cases"]
                    
                    edge[source "SST34" target "Temperature"]
                    edge[source "SST34" target "excess_cases"]
                    
                    edge[source "SST12" target "Temperature"]
                    edge[source "SST12" target "excess_cases"]
                    
                    edge[source "NATL" target "Temperature"]
                    edge[source "NATL" target "excess_cases"]
                    
                    edge[source "SATL" target "Temperature"]
                    edge[source "SATL" target "excess_cases"]
                    
                    edge[source "TROP" target "Temperature"]
                    edge[source "TROP" target "excess_cases"]
                                                      
                    edge[source "forest_percent" target "Temperature"]
                    edge[source "forest_percent" target "excess_cases"]
                    
                    edge[source "SOI" target "Equatorial_SOI"]
                    edge[source "SOI" target "SST3"]
                    edge[source "SOI" target "SST4"]
                    edge[source "SOI" target "SST34"]
                    edge[source "SOI" target "SST12"]
                    edge[source "SOI" target "NATL"]
                    edge[source "SOI" target "SATL"]
                    edge[source "SOI" target "TROP"]
                    
                    edge[source "Equatorial_SOI" target "SST3"]
                    edge[source "Equatorial_SOI" target "SST4"]
                    edge[source "Equatorial_SOI" target "SST34"]
                    edge[source "Equatorial_SOI" target "SST12"]
                    edge[source "Equatorial_SOI" target "NATL"]
                    edge[source "Equatorial_SOI" target "SATL"]
                    edge[source "Equatorial_SOI" target "TROP"]
                    
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
                    
                    edge[source "Qs" target "excess_cases"]]"""
                    )
              
        
#view model 
#model_leish.view_model()

#Step 2: Identifying effects
identified_estimand_runoff = model_leish.identify_effect(proceed_when_unidentifiable=False)
print(identified_estimand_runoff)

#Step 3: Estimation of the effect 
estimate_runoff = DML(model_y=reg1(), model_t=reg2(), model_final=LassoCV(), discrete_treatment=False,
                        featurizer=PolynomialFeatures(degree=3),
                        linear_first_stages=False, cv=3, random_state=123)

estimate_runoff = estimate_runoff.dowhy

# fit the model
estimate_runoff.fit(Y=Y, T=T, X=X, W=W, inference='bootstrap')  

# predict effect for each sample X
estimate_runoff.effect(X)

# ate
ate_Runoff = estimate_runoff.ate(X) 
print(ate_Runoff)

# confidence interval of ate
ci_Runoff = estimate_runoff.ate_interval(X) 
print(ci_Runoff)

#Step 4: Refute the effect
#with random common cause
random_Runoff = estimate_runoff.refute_estimate(method_name="random_common_cause", random_state=123)
print(random_Runoff)

#with replace a random subset of the data
subset_Runoff = estimate_runoff.refute_estimate(method_name="data_subset_refuter", subset_fraction=0.9, num_simulations=3)
print(subset_Runoff)

#with placebo 
placebo_Runoff = estimate_runoff.refute_estimate(method_name="placebo_treatment_refuter", placebo_type="permute", num_simulations=3)
print(placebo_Runoff)




#####soil moisture

Y = Colombia_soilmoisture.excess_cases.to_numpy() #Y = data_card['incidencia100k_cardiovasculares'].values
T = Colombia_soilmoisture.SoilMoi0_10cm.to_numpy()
W = Colombia_soilmoisture[['SOI', 'Equatorial_SOI', 'SST3', 'SST4', 'SST34', 'SST12', 'NATL', 'SATL', 'TROP', 'forest_percent']].to_numpy().reshape(-1, 10)
X = Colombia_soilmoisture[['forest_percent']].to_numpy()

#Model Selection for Causal Effect Model with the RScorer
# A multitude of possible approaches for CATE estimation under conditional exogeneity
X_train, X_val, T_train, T_val, Y_train, Y_val, W_train, W_val = train_test_split(X, T, Y, W, test_size=.4)

models = [
        ('ldml', LinearDML(model_y=reg1(), model_t=reg2(), discrete_treatment=False,
                             linear_first_stages=False, cv=3, random_state=123)),
        ('sldml', SparseLinearDML(model_y=reg1(), model_t=reg2(), discrete_treatment=False,
                                    featurizer=PolynomialFeatures(degree=2, include_bias=False),
                                    linear_first_stages=False, cv=3, random_state=123)),
        ('dml', DML(model_y=reg1(), model_t=reg2(), model_final=LassoCV(), discrete_treatment=False,
                             featurizer=PolynomialFeatures(degree=3),
                             linear_first_stages=False, cv=3, random_state=123)),
        ('forest', CausalForestDML(model_y=reg1(), model_t=reg2(), featurizer=PolynomialFeatures(degree=3), 
                             discrete_treatment=False, cv=3, random_state=123)),
        ('kernel', KernelDML(model_y=reg1(), model_t=reg2(),
                             cv=3, random_state=123)),
                 
         ]



def fit_model(name, model):
    return name, model.fit(Y_train, T_train, X=X_train, W=W_train)

models = Parallel(n_jobs=-1, verbose=1, backend="threading")(delayed(fit_model)(name, mdl) for name, mdl in models)


#Choose model with highest RScore
scorer = RScorer(model_y=reg1(), model_t=reg2(),
                 discrete_treatment=False, cv=3,
                 mc_iters=3, mc_agg='median')

scorer.fit(Y_val, T_val, X=X_val, W=W_val)


rscore = [scorer.score(mdl) for _, mdl in models]
print(rscore) #best model SparseLinearDML



#Step 1: Modeling the causal mechanism
model_leish=CausalModel(
        data = Colombia_soilmoisture,
        treatment=['SoilMoi0_10cm'],
        outcome=['excess_cases'],
        graph= """graph[directed 1 node[id "SoilMoi0_10cm" label "SoilMoi0_10cm"]
                     node[id "excess_cases" label "excess_cases"]
                    node[id "SOI" label "SOI"]
                    node[id "Equatorial_SOI" label "Equatorial_SOI"]
                    node[id "SST3" label "SST3"]               
                    node[id "SST4" label "SST4"]
                    node[id "SST34" label "SST34"]
                    node[id "SST12" label "SST12"]
                    node[id "NATL" label "NATL"]
                    node[id "SATL" label "SATL"]
                    node[id "TROP" label "TROP"]                                      
                    node[id "forest_percent" label "forest_percent"]
                                       
                    edge[source "SOI" target "Temperature"]
                    edge[source "SOI" target "excess_cases"]
                    
                    edge[source "Equatorial_SOI" target "Temperature"]
                    edge[source "Equatorial_SOI" target "excess_cases"]
                    
                    edge[source "SST3" target "Temperature"]
                    edge[source "SST3" target "excess_cases"]
                    
                    edge[source "SST4" target "Temperature"]
                    edge[source "SST4" target "excess_cases"]
                    
                    edge[source "SST34" target "Temperature"]
                    edge[source "SST34" target "excess_cases"]
                    
                    edge[source "SST12" target "Temperature"]
                    edge[source "SST12" target "excess_cases"]
                    
                    edge[source "NATL" target "Temperature"]
                    edge[source "NATL" target "excess_cases"]
                    
                    edge[source "SATL" target "Temperature"]
                    edge[source "SATL" target "excess_cases"]
                    
                    edge[source "TROP" target "Temperature"]
                    edge[source "TROP" target "excess_cases"]
                                                      
                    edge[source "forest_percent" target "Temperature"]
                    edge[source "forest_percent" target "excess_cases"]
                    
                    edge[source "SOI" target "Equatorial_SOI"]
                    edge[source "SOI" target "SST3"]
                    edge[source "SOI" target "SST4"]
                    edge[source "SOI" target "SST34"]
                    edge[source "SOI" target "SST12"]
                    edge[source "SOI" target "NATL"]
                    edge[source "SOI" target "SATL"]
                    edge[source "SOI" target "TROP"]
                    
                    edge[source "Equatorial_SOI" target "SST3"]
                    edge[source "Equatorial_SOI" target "SST4"]
                    edge[source "Equatorial_SOI" target "SST34"]
                    edge[source "Equatorial_SOI" target "SST12"]
                    edge[source "Equatorial_SOI" target "NATL"]
                    edge[source "Equatorial_SOI" target "SATL"]
                    edge[source "Equatorial_SOI" target "TROP"]
                    
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
                    
                    edge[source "SoilMoi0_10cm" target "excess_cases"]]"""
                    )
        
        
        
#view model 
#model_leish.view_model()

#Step 2: Identifying effects
identified_estimand_soilmoist = model_leish.identify_effect(proceed_when_unidentifiable=False)
print(identified_estimand_soilmoist)

#Step 3: Estimation of the effect 
estimate_soilmoist = SparseLinearDML(model_y=reg1(), model_t=reg2(), discrete_treatment=False,
                                    featurizer=PolynomialFeatures(degree=2, include_bias=False),
                                    linear_first_stages=False, cv=3, random_state=123)

estimate_soilmoist = estimate_soilmoist.dowhy

# fit the model
estimate_soilmoist.fit(Y=Y, T=T, X=X, W=W, inference='bootstrap')  

# predict effect for each sample X
estimate_soilmoist.effect(X)

# ate
ate_Soilmoist = estimate_soilmoist.ate(X) 
print(ate_Soilmoist)

# confidence interval of ate
ci_Soilmoist = estimate_soilmoist.ate_interval(X) 
print(ci_Soilmoist)

#Step 4: Refute the effect
#with random common cause
random_Soilmoist = estimate_soilmoist.refute_estimate(method_name="random_common_cause", random_state=123)
print(random_Soilmoist)

#with replace a random subset of the data
subset_Soilmoist = estimate_soilmoist.refute_estimate(method_name="data_subset_refuter", subset_fraction=0.9, num_simulations=3)
print(subset_Soilmoist)

#with placebo 
placebo_Soilmoist = estimate_soilmoist.refute_estimate(method_name="placebo_treatment_refuter", placebo_type="permute", num_simulations=3)
print(placebo_Soilmoist)




#####EVI

Y = Colombia_EVI.excess_cases.to_numpy() #Y = data_card['incidencia100k_cardiovasculares'].values
T = Colombia_EVI.EVI.to_numpy()
W = Colombia_EVI[['SOI', 'Equatorial_SOI', 'SST3', 'SST4', 'SST34', 'SST12', 'NATL', 'SATL', 'TROP', 'forest_percent']].to_numpy().reshape(-1, 10)
X = Colombia_EVI[['forest_percent']].to_numpy()


#Model Selection for Causal Effect Model with the RScorer
# A multitude of possible approaches for CATE estimation under conditional exogeneity
X_train, X_val, T_train, T_val, Y_train, Y_val, W_train, W_val = train_test_split(X, T, Y, W, test_size=.4)


models = [
        ('ldml', LinearDML(model_y=reg1(), model_t=reg2(), discrete_treatment=False,
                             linear_first_stages=False, cv=3, random_state=123)),
        ('sldml', SparseLinearDML(model_y=reg1(), model_t=reg2(), discrete_treatment=False,
                                    featurizer=PolynomialFeatures(degree=2, include_bias=False),
                                    linear_first_stages=False, cv=3, random_state=123)),
        ('dml', DML(model_y=reg1(), model_t=reg2(), model_final=LassoCV(), discrete_treatment=False,
                             featurizer=PolynomialFeatures(degree=3),
                             linear_first_stages=False, cv=3, random_state=123)),
        ('forest', CausalForestDML(model_y=reg1(), model_t=reg2(), featurizer=PolynomialFeatures(degree=3), 
                             discrete_treatment=False, cv=3, random_state=123)),
        ('kernel', KernelDML(model_y=reg1(), model_t=reg2(),
                             cv=3, random_state=123)),
                  
         ]



def fit_model(name, model):
    return name, model.fit(Y_train, T_train, X=X_train, W=W_train)

models = Parallel(n_jobs=-1, verbose=1, backend="threading")(delayed(fit_model)(name, mdl) for name, mdl in models)


#Choose model with highest RScore
scorer = RScorer(model_y=reg1(), model_t=reg2(),
                 discrete_treatment=False, cv=3,
                 mc_iters=3, mc_agg='median')

scorer.fit(Y_val, T_val, X=X_val, W=W_val)


rscore = [scorer.score(mdl) for _, mdl in models]
print(rscore) #best model SparseLinearDML


#Step 1: Modeling the causal mechanism
model_leish=CausalModel(
        data = Colombia_EVI,
        treatment=['EVI'],
        outcome=['excess_cases'],
        graph= """graph[directed 1 node[id "EVI" label "EVI"]
                     node[id "excess_cases" label "excess_cases"]
                    node[id "SOI" label "SOI"]
                    node[id "Equatorial_SOI" label "Equatorial_SOI"]
                    node[id "SST3" label "SST3"]               
                    node[id "SST4" label "SST4"]
                    node[id "SST34" label "SST34"]
                    node[id "SST12" label "SST12"]
                    node[id "NATL" label "NATL"]
                    node[id "SATL" label "SATL"]
                    node[id "TROP" label "TROP"]                                      
                    node[id "forest_percent" label "forest_percent"]
                                       
                    edge[source "SOI" target "Temperature"]
                    edge[source "SOI" target "excess_cases"]
                    
                    edge[source "Equatorial_SOI" target "Temperature"]
                    edge[source "Equatorial_SOI" target "excess_cases"]
                    
                    edge[source "SST3" target "Temperature"]
                    edge[source "SST3" target "excess_cases"]
                    
                    edge[source "SST4" target "Temperature"]
                    edge[source "SST4" target "excess_cases"]
                    
                    edge[source "SST34" target "Temperature"]
                    edge[source "SST34" target "excess_cases"]
                    
                    edge[source "SST12" target "Temperature"]
                    edge[source "SST12" target "excess_cases"]
                    
                    edge[source "NATL" target "Temperature"]
                    edge[source "NATL" target "excess_cases"]
                    
                    edge[source "SATL" target "Temperature"]
                    edge[source "SATL" target "excess_cases"]
                    
                    edge[source "TROP" target "Temperature"]
                    edge[source "TROP" target "excess_cases"]
                                                      
                    edge[source "forest_percent" target "Temperature"]
                    edge[source "forest_percent" target "excess_cases"]
                    
                    edge[source "SOI" target "Equatorial_SOI"]
                    edge[source "SOI" target "SST3"]
                    edge[source "SOI" target "SST4"]
                    edge[source "SOI" target "SST34"]
                    edge[source "SOI" target "SST12"]
                    edge[source "SOI" target "NATL"]
                    edge[source "SOI" target "SATL"]
                    edge[source "SOI" target "TROP"]
                    
                    edge[source "Equatorial_SOI" target "SST3"]
                    edge[source "Equatorial_SOI" target "SST4"]
                    edge[source "Equatorial_SOI" target "SST34"]
                    edge[source "Equatorial_SOI" target "SST12"]
                    edge[source "Equatorial_SOI" target "NATL"]
                    edge[source "Equatorial_SOI" target "SATL"]
                    edge[source "Equatorial_SOI" target "TROP"]
                    
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
                    
                    edge[source "EVI" target "excess_cases"]]"""
                    )
        
        
        
#view model 
#model_leish.view_model()

#Step 2: Identifying effects
identified_estimand_EVI = model_leish.identify_effect(proceed_when_unidentifiable=False)
print(identified_estimand_EVI)

#Step 3: Estimation of the effect 
estimate_EVI = SparseLinearDML(model_y=reg1(), model_t=reg2(), discrete_treatment=False,
                                    featurizer=PolynomialFeatures(degree=2, include_bias=False),
                                    linear_first_stages=False, cv=3, random_state=123)

estimate_EVI = estimate_EVI.dowhy

# fit the model
estimate_EVI.fit(Y=Y, T=T, X=X, W=W, inference='bootstrap')  

# predict effect for each sample X
estimate_EVI.effect(X)

# ate
ate_EVI = estimate_EVI.ate(X) 
print(ate_EVI)

# confidence interval of ate
ci_EVI = estimate_EVI.ate_interval(X) 
print(ci_EVI)

#Step 4: Refute the effect
#with random common cause
random_EVI = estimate_EVI.refute_estimate(method_name="random_common_cause", random_state=123)
print(random_EVI)

#with replace a random subset of the data
subset_EVI = estimate_EVI.refute_estimate(method_name="data_subset_refuter", subset_fraction=0.9, num_simulations=3)
print(subset_EVI)

#with placebo 
placebo_EVI = estimate_EVI.refute_estimate(method_name="placebo_treatment_refuter", placebo_type="permute", num_simulations=3)
print(placebo_EVI)




####################
#Figure 3
labs = ['Air_temperature',
        'Soil_temperature',
        'Rainfall',
        'Runoff',
        'Soil_moisture',
        'EVI']

measure = [0.013, '0.040', 0.019, 0.036, 0.048, 0.057]
lower =   [-0.022, '0.030', 0.015, 0.029, 0.038, '0.050']
upper =   [0.048, '0.050', 0.023, 0.043, 0.058, 0.064]

p = EffectMeasurePlot(label=labs, effect_measure=measure, lcl=lower, ucl=upper)
p.labels(center=0)
p.colors(pointcolor='r') 
p.labels(effectmeasure='ATE')  
p.plot(figsize=(10, 5), t_adjuster=0.075, max_value=0.25, min_value=-0.25)
plt.tight_layout()
plt.show()


