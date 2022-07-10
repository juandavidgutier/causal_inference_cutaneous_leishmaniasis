###############################################################################
# Code for the paper:
#   "Causal association of environmental variables on the occurrence of outbreaks of 
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
from econml.orf import DMLOrthoForest
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
data_col = pd.read_csv("https://raw.githubusercontent.com/juandavidgutier/causal_inference_cutaneous_leishmaniasis/main/dataset.csv", encoding='latin-1') 
data_col = data_col.dropna()
data_col['SoilTMP0_10cm'] = data_col['SoilTMP0_10cm'].astype(float)

#temperature
Colombia_temp = data_col[['outbreak', 'Temperature', 'qbo', 'wpac', 'epac', 'Forest']] 
#soil temperature
Colombia_soiltemp = data_col[['outbreak', 'SoilTMP0_10cm', 'qbo', 'wpac', 'epac', 'Forest']] 
Colombia_soiltemp['SoilTMP0_10cm'] = Colombia_soiltemp['SoilTMP0_10cm'].astype(float)
#rainfall
Colombia_rainfall = data_col[['outbreak', 'Rainfall', 'qbo', 'wpac', 'epac', 'Forest']] 
#runoff
Colombia_runoff = data_col[['outbreak', 'Qs', 'qbo', 'wpac', 'epac', 'Forest']] 
#soil moisture
Colombia_soilmoisture = data_col[['outbreak', 'SoilMoi0_10cm', 'qbo', 'wpac', 'epac', 'Forest']] 
#EVI
Colombia_EVI = data_col[['outbreak', 'EVI', 'qbo', 'wpac', 'epac', 'Forest']] 


###############################################################################
#####Temperature

Y = Colombia_temp.outbreak.to_numpy() #Y = data_card['incidencia100k_cardiovasculares'].values
T = Colombia_temp.Temperature.to_numpy()
W = Colombia_temp[['epac', 'wpac', 'qbo', 'Forest']].to_numpy().reshape(-1, 4)
X = Colombia_temp[['Forest']].to_numpy()


#Model Selection for Causal Effect Model with the RScorer
# A multitude of possible approaches for CATE estimation under conditional exogeneity


X_train, X_val, T_train, T_val, Y_train, Y_val, W_train, W_val = train_test_split(X, T, Y, W, test_size=.4)

#%load_ext bootstrapreload
#%bootstrapreload 2

## Ignore warnings
warnings.filterwarnings('ignore') 

reg1 = lambda: GradientBoostingClassifier()
reg2 = lambda: GradientBoostingRegressor()

models = [
        ('ldml', LinearDML(model_y=reg1(), model_t=reg2(), discrete_treatment=False,
                             linear_first_stages=False, cv=3, random_state=123)),
        ('sldml', SparseLinearDML(model_y=reg1(), model_t=reg2(), discrete_treatment=False,
                                    featurizer=PolynomialFeatures(degree=2, include_bias=False),
                                    linear_first_stages=False, cv=3, random_state=123)),
        ('dml', DML(model_y=reg1(), model_t=reg2(), model_final=LassoCV(), discrete_treatment=False,
                             featurizer=PolynomialFeatures(degree=3),
                             linear_first_stages=False, cv=3, random_state=123)),
        #('ortho', DMLOrthoForest(model_Y=reg1(), model_T=reg2(), model_T_final=LassoCV(), model_Y_final=LassoCV(), 
        #                     discrete_treatment=False, global_res_cv=3, random_state=123)),
         ('forest', CausalForestDML(model_y=reg1(), model_t=reg2(), featurizer=PolynomialFeatures(degree=3), 
                             discrete_treatment=False, cv=3, random_state=123)),
          
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


#Step 1: Modeforestg the causal mechanism
model_leish=CausalModel(
        data = Colombia_temp,
        treatment=['Temperature'],
        outcome=['outbreak'],
        graph= """graph[directed 1 node[id "Temperature" label "Temperature"]
                    node[id "outbreak" label "outbreak"]
                    node[id "wpac" label "wpac"]
                    node[id "qbo" label "qbo"]
                    node[id "epac" label "epac"]
                    node[id "Forest" label "Forest"]
                                       
                    edge[source "wpac" target "Temperature"]
                    edge[source "wpac" target "outbreak"]
                    
                    edge[source "qbo" target "Temperature"]
                    edge[source "qbo" target "outbreak"]
                    
                    edge[source "epac" target "Temperature"]
                    edge[source "epac" target "outbreak"]
                    
                    edge[source "Forest" target "Temperature"]
                    edge[source "Forest" target "outbreak"]
                    
                    edge[source "wpac" target "qbo"]
                    edge[source "wpac" target "epac"]
                    edge[source "epac" target "qbo"]
                    edge[source "epac" target "wpac"]
                    edge[source "qbo" target "wpac"]
                    edge[source "qbo" target "epac"]
                    
                    edge[source "Temperature" target "outbreak"]]"""
                    )
    
#view model 
#model_leish.view_model()

#Step 2: Identifying effects
identified_estimand_temp = model_leish.identify_effect(proceed_when_unidentifiable=False)
print(identified_estimand_temp)

estimate_temp = DML(model_y=reg1(), model_t=reg2(), model_final=LassoCV(), discrete_treatment=False,
                    featurizer=PolynomialFeatures(degree=3),
                    linear_first_stages=False, cv=3, random_state=123)

estimate_temp = estimate_temp.dowhy

# fit the CATE model
estimate_temp.fit(Y=Y, T=T, X=X, W=W, inference='bootstrap')  

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

##with add unobserved common cause
unobserved_Temperature = estimate_temp.refute_estimate(method_name="add_unobserved_common_cause", 
                                                       effect_strength_on_treatment=0.005, effect_strength_on_outcome=0.005,
                                                       confounders_effect_on_outcome="binary_flip")
print(unobserved_Temperature)

#with replace a random subset of the data
subset_Temperature = estimate_temp.refute_estimate(method_name="data_subset_refuter", subset_fraction=0.8, num_simulations=3)
print(subset_Temperature)

#with placebo 
placebo_Temperature = estimate_temp.refute_estimate(method_name="placebo_treatment_refuter", placebo_type="permute", num_simulations=3)
print(placebo_Temperature)





#####soil Temperature

Y = Colombia_soiltemp.outbreak.to_numpy() #Y = data_card['incidencia100k_cardiovasculares'].values
T = Colombia_soiltemp.SoilTMP0_10cm.to_numpy()
W = Colombia_soiltemp[['epac', 'wpac', 'qbo', 'Forest']].to_numpy().reshape(-1, 4)
X = Colombia_soiltemp[['Forest']].to_numpy()


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
        #('ortho', DMLOrthoForest(model_Y=reg1(), model_T=reg2(), model_T_final=LassoCV(), model_Y_final=LassoCV(), 
        #                     discrete_treatment=False, global_res_cv=3, random_state=123)),
         ('forest', CausalForestDML(model_y=reg1(), model_t=reg2(), featurizer=PolynomialFeatures(degree=3), 
                             discrete_treatment=False, cv=3, random_state=123)),
          
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


#Step 1: Modeforestg the causal mechanism
model_leish=CausalModel(
        data = Colombia_soiltemp,
        treatment=['SoilTMP0_10cm'],
        outcome=['outbreak'],
        graph= """graph[directed 1 node[id "SoilTMP0_10cm" label "SoilTMP0_10cm"]
                    node[id "outbreak" label "outbreak"]
                    node[id "wpac" label "wpac"]
                    node[id "qbo" label "qbo"]
                    node[id "epac" label "epac"]
                    node[id "Forest" label "Forest"]
                                       
                    edge[source "wpac" target "SoilTMP0_10cm"]
                    edge[source "wpac" target "outbreak"]
                    
                    edge[source "qbo" target "SoilTMP0_10cm"]
                    edge[source "qbo" target "outbreak"]
                    
                    edge[source "epac" target "SoilTMP0_10cm"]
                    edge[source "epac" target "outbreak"]
                    
                    edge[source "Forest" target "SoilTMP0_10cm"]
                    edge[source "Forest" target "outbreak"]
                    
                    edge[source "wpac" target "qbo"]
                    edge[source "wpac" target "epac"]
                    edge[source "epac" target "qbo"]
                    edge[source "epac" target "wpac"]
                    edge[source "qbo" target "wpac"]
                    edge[source "qbo" target "epac"]
                    
                    edge[source "SoilTMP0_10cm" target "outbreak"]]"""
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

# fit the CATE model
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

##with add unobserved common cause
unobserved_SoilTemperature = estimate_soiltemp.refute_estimate(method_name="add_unobserved_common_cause", 
                                                               effect_strength_on_treatment=0.005, effect_strength_on_outcome=0.005,
                                                               confounders_effect_on_outcome="binary_flip")
print(unobserved_SoilTemperature)

#with replace a random subset of the data
subset_SoilTemperature = estimate_soiltemp.refute_estimate(method_name="data_subset_refuter", subset_fraction=0.8, num_simulations=3)
print(subset_SoilTemperature)

#with placebo 
placebo_SoilTemperature = estimate_soiltemp.refute_estimate(method_name="placebo_treatment_refuter", placebo_type="permute", num_simulations=3)
print(placebo_SoilTemperature)



#####Rainfall

Y = Colombia_rainfall.outbreak.to_numpy() #Y = data_card['incidencia100k_cardiovasculares'].values
T = Colombia_rainfall.Rainfall.to_numpy()
W = Colombia_rainfall[['epac', 'wpac', 'qbo', 'Forest']].to_numpy().reshape(-1, 4)
X = Colombia_rainfall[['Forest']].to_numpy()


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
        #('ortho', DMLOrthoForest(model_Y=reg1(), model_T=reg2(), model_T_final=LassoCV(), model_Y_final=LassoCV(), 
        #                     discrete_treatment=False, global_res_cv=3, random_state=123)),
         ('forest', CausalForestDML(model_y=reg1(), model_t=reg2(), featurizer=PolynomialFeatures(degree=3), 
                             discrete_treatment=False, cv=3, random_state=123)),
          
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


#Step 1: Modeforestg the causal mechanism
model_leish=CausalModel(
        data = Colombia_rainfall,
        treatment=['Rainfall'],
        outcome=['outbreak'],
        graph= """graph[directed 1 node[id "Rainfall" label "Rainfall"]
                    node[id "outbreak" label "outbreak"]
                    node[id "wpac" label "wpac"]
                    node[id "qbo" label "qbo"]
                    node[id "epac" label "epac"]
                    node[id "Forest" label "Forest"]
                    
                    edge[source "wpac" target "Rainfall"]
                    edge[source "wpac" target "outbreak"]
                    
                    edge[source "qbo" target "Rainfall"]
                    edge[source "qbo" target "outbreak"]
                    
                    edge[source "epac" target "Rainfall"]
                    edge[source "epac" target "outbreak"]
                    
                    edge[source "Forest" target "Rainfall"]
                    edge[source "Forest" target "outbreak"]
                    
                    edge[source "wpac" target "qbo"]
                    edge[source "wpac" target "epac"]
                    edge[source "epac" target "qbo"]
                    edge[source "epac" target "wpac"]
                    edge[source "qbo" target "wpac"]
                    edge[source "qbo" target "epac"]
                    
                    edge[source "Rainfall" target "outbreak"]]"""
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

# fit the CATE model
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

##with add unobserved common cause
unobserved_Rain = estimate_rain.refute_estimate(method_name="add_unobserved_common_cause", effect_strength_on_treatment=0.005, 
                                                effect_strength_on_outcome=0.005, confounders_effect_on_outcome="binary_flip")
print(unobserved_Rain)

#with replace a random subset of the data
subset_Rain = estimate_rain.refute_estimate(method_name="data_subset_refuter", subset_fraction=0.8, num_simulations=3)
print(subset_Rain)

#with placebo 
placebo_Rain = estimate_rain.refute_estimate(method_name="placebo_treatment_refuter", placebo_type="permute", num_simulations=3)
print(placebo_Rain)




#####Runoff

Y = Colombia_runoff.outbreak.to_numpy() #Y = data_card['incidencia100k_cardiovasculares'].values
T = Colombia_runoff.Qs.to_numpy()
W = Colombia_runoff[['epac', 'wpac', 'qbo', 'Forest']].to_numpy().reshape(-1, 4)
X = Colombia_runoff[['Forest']].to_numpy()



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
        #('ortho', DMLOrthoForest(model_Y=reg1(), model_T=reg2(), model_T_final=LassoCV(), model_Y_final=LassoCV(), 
        #                     discrete_treatment=False, global_res_cv=3, random_state=123)),
         ('forest', CausalForestDML(model_y=reg1(), model_t=reg2(), featurizer=PolynomialFeatures(degree=3), 
                             discrete_treatment=False, cv=3, random_state=123)),
          
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



#Step 1: Modeforestg the causal mechanism
model_leish=CausalModel(
        data = Colombia_runoff,
        treatment=['Qs'],
        outcome=['outbreak'],
        graph= """graph[directed 1 node[id "Qs" label "Qs"]
                    node[id "outbreak" label "outbreak"]
                    node[id "wpac" label "wpac"]
                    node[id "qbo" label "qbo"]
                    node[id "epac" label "epac"]
                    node[id "Forest" label "Forest"]
                    
                    edge[source "wpac" target "Qs"]
                    edge[source "wpac" target "outbreak"]
                    
                    edge[source "qbo" target "Qs"]
                    edge[source "qbo" target "outbreak"]
                    
                    edge[source "epac" target "Qs"]
                    edge[source "epac" target "outbreak"]
                    
                    edge[source "Forest" target "Qs"]
                    edge[source "Forest" target "outbreak"]
                    
                    edge[source "wpac" target "qbo"]
                    edge[source "wpac" target "epac"]
                    edge[source "epac" target "qbo"]
                    edge[source "epac" target "wpac"]
                    edge[source "qbo" target "wpac"]
                    edge[source "qbo" target "epac"]
                    
                    edge[source "Qs" target "outbreak"]]"""
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

# fit the CATE model
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

##with add unobserved common cause
unobserved_Runoff = estimate_runoff.refute_estimate(method_name="add_unobserved_common_cause", effect_strength_on_treatment=0.005, 
                                                    effect_strength_on_outcome=0.005, confounders_effect_on_outcome="binary_flip")
print(unobserved_Runoff)

#with replace a random subset of the data
subset_Runoff = estimate_runoff.refute_estimate(method_name="data_subset_refuter", subset_fraction=0.8, num_simulations=3)
print(subset_Runoff)

#with placebo 
placebo_Runoff = estimate_runoff.refute_estimate(method_name="placebo_treatment_refuter", placebo_type="permute", num_simulations=3)
print(placebo_Runoff)




#####soil moisture

Y = Colombia_soilmoisture.outbreak.to_numpy() #Y = data_card['incidencia100k_cardiovasculares'].values
T = Colombia_soilmoisture.SoilMoi0_10cm.to_numpy()
W = Colombia_soilmoisture[['epac', 'wpac', 'qbo', 'Forest']].to_numpy().reshape(-1, 4)
X = Colombia_soilmoisture[['Forest']].to_numpy()

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
        #('ortho', DMLOrthoForest(model_Y=reg1(), model_T=reg2(), model_T_final=LassoCV(), model_Y_final=LassoCV(), 
        #                     discrete_treatment=False, global_res_cv=3, random_state=123)),
         ('forest', CausalForestDML(model_y=reg1(), model_t=reg2(), featurizer=PolynomialFeatures(degree=3), 
                             discrete_treatment=False, cv=3, random_state=123)),
          
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



#Step 1: Modeforestg the causal mechanism
model_leish=CausalModel(
        data = Colombia_soilmoisture,
        treatment=['SoilMoi0_10cm'],
        outcome=['outbreak'],
        graph= """graph[directed 1 node[id "SoilMoi0_10cm" label "SoilMoi0_10cm"]
                    node[id "outbreak" label "outbreak"]
                    node[id "wpac" label "wpac"]
                    node[id "qbo" label "qbo"]
                    node[id "epac" label "epac"]
                    node[id "Forest" label "Forest"]
                    
                    edge[source "wpac" target "SoilMoi0_10cm"]
                    edge[source "wpac" target "outbreak"]
                    
                    edge[source "qbo" target "SoilMoi0_10cm"]
                    edge[source "qbo" target "outbreak"]
                    
                    edge[source "epac" target "SoilMoi0_10cm"]
                    edge[source "epac" target "outbreak"]
                    
                    edge[source "Forest" target "SoilMoi0_10cm"]
                    edge[source "Forest" target "outbreak"]
                    
                    edge[source "wpac" target "qbo"]
                    edge[source "wpac" target "epac"]
                    edge[source "epac" target "qbo"]
                    edge[source "epac" target "wpac"]
                    edge[source "qbo" target "wpac"]
                    edge[source "qbo" target "epac"]
                    
                    edge[source "SoilMoi0_10cm" target "outbreak"]]"""
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

# fit the CATE model
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

##with add unobserved common cause
unobserved_Soilmoist = estimate_soilmoist.refute_estimate(method_name="add_unobserved_common_cause", effect_strength_on_treatment=0.005, 
                                                          effect_strength_on_outcome=0.005, confounders_effect_on_outcome="binary_flip")
print(unobserved_Soilmoist)

#with replace a random subset of the data
subset_Soilmoist = estimate_soilmoist.refute_estimate(method_name="data_subset_refuter", subset_fraction=0.8, num_simulations=3)
print(subset_Soilmoist)

#with placebo 
placebo_Soilmoist = estimate_soilmoist.refute_estimate(method_name="placebo_treatment_refuter", placebo_type="permute", num_simulations=3)
print(placebo_Soilmoist)




#####EVI

Y = Colombia_EVI.outbreak.to_numpy() #Y = data_card['incidencia100k_cardiovasculares'].values
T = Colombia_EVI.EVI.to_numpy()
W = Colombia_EVI[['epac', 'wpac', 'qbo', 'Forest']].to_numpy().reshape(-1, 4)
X = Colombia_EVI[['Forest']].to_numpy()


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
        #('ortho', DMLOrthoForest(model_Y=reg1(), model_T=reg2(), model_T_final=LassoCV(), model_Y_final=LassoCV(), 
        #                     discrete_treatment=False, global_res_cv=3, random_state=123)),
         ('forest', CausalForestDML(model_y=reg1(), model_t=reg2(), featurizer=PolynomialFeatures(degree=3), 
                             discrete_treatment=False, cv=3, random_state=123)),
          
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


#Step 1: Modeforestg the causal mechanism
model_leish=CausalModel(
        data = Colombia_EVI,
        treatment=['EVI'],
        outcome=['outbreak'],
        graph= """graph[directed 1 node[id "EVI" label "EVI"]
                    node[id "outbreak" label "outbreak"]
                    node[id "wpac" label "wpac"]
                    node[id "qbo" label "qbo"]
                    node[id "epac" label "epac"]
                    node[id "Forest" label "Forest"]
                    
                    edge[source "wpac" target "EVI"]
                    edge[source "wpac" target "outbreak"]
                    
                    edge[source "qbo" target "EVI"]
                    edge[source "qbo" target "outbreak"]
                    
                    edge[source "epac" target "EVI"]
                    edge[source "epac" target "outbreak"]
                    
                    edge[source "Forest" target "EVI"]
                    edge[source "Forest" target "outbreak"]
                    
                    edge[source "wpac" target "qbo"]
                    edge[source "wpac" target "epac"]
                    edge[source "epac" target "qbo"]
                    edge[source "epac" target "wpac"]
                    edge[source "qbo" target "wpac"]
                    edge[source "qbo" target "epac"]
                    
                    edge[source "EVI" target "outbreak"]]"""
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

# fit the CATE model
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

##with add unobserved common cause
unobserved_EVI = estimate_EVI.refute_estimate(method_name="add_unobserved_common_cause", effect_strength_on_treatment=0.005,
                                              effect_strength_on_outcome=0.005, confounders_effect_on_outcome="binary_flip")
print(unobserved_EVI)

#with replace a random subset of the data
subset_EVI = estimate_EVI.refute_estimate(method_name="data_subset_refuter", subset_fraction=0.8, num_simulations=3)
print(subset_EVI)

#with placebo 
placebo_EVI = estimate_EVI.refute_estimate(method_name="placebo_treatment_refuter", placebo_type="permute", num_simulations=3)
print(placebo_EVI)




####################
#Figure XXXX
labs = ['Air_temperature',
        'Soil_temperature',
        'Rainfall',
        'Runoff',
        'Soil_moisture',
        'EVI']

measure = [0.046, '0.040', 0.019, 0.036, 0.048, 0.057]
lower =   [0.037, '0.030', 0.015, 0.029, 0.038, '0.050']
upper =   [0.055, '0.050', 0.023, 0.043, 0.058, 0.064]

p = EffectMeasurePlot(label=labs, effect_measure=measure, lcl=lower, ucl=upper)
p.labels(center=0)
p.colors(pointcolor='r') #, pointshape="|")
p.labels(effectmeasure='ATE')  
p.plot(figsize=(10, 5), t_adjuster=0.075, max_value=0.25, min_value=-0.25)
plt.tight_layout()
plt.show()


