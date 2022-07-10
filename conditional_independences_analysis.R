library(dagitty)
library(lavaan)
library(CondIndTests)
library(dplyr)
library(GGally)

url_path = "https://raw.githubusercontent.com/juandavidgutier/causal_inference_cutaneous_leishmaniasis/master/dataset.csv"
data <- read.csv(url_path)
data_temp <- dplyr::select(data, outbreak, Temperature, 
                          wpac, epac, qbo, Forest) 

#DAG
dag_temp <- dagitty('dag {

Temperature [pos="0,0"]
Forest [pos="0.99, 1"]
epac [pos="0.66, 1"]
qbo [pos="0, 1"]
wpac [pos="0.33, 1"]
outbreak [pos="1, 0"]
wpac -> Temperature 
wpac -> outbreak 
qbo -> Temperature 
qbo -> outbreak 
epac -> Temperature 
epac -> outbreak 

qbo -> wpac
qbo -> epac
wpac -> qbo
wpac -> epac
epac -> qbo
epac -> wpac

Forest -> outbreak
Forest -> Temperature
Temperature -> outbreak}')  

plot(dag_temp)

## Independencias condicionales
impliedConditionalIndependencies(dag_temp)
corr <- lavCor(data_temp)

#plot con ci convencinales (método analitico)
localTests(dag_temp, sample.cov=corr, sample.nobs=nrow(data_temp))
plotLocalTestResults(localTests(dag_temp, sample.cov=corr, sample.nobs=nrow(data_temp)), xlim=c(-1,1))



