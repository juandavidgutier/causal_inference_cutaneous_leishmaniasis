library(ggdag)
library(dagitty)
library(lavaan)
library(CondIndTests)
library(dplyr)
library(GGally)
library(tidyr)
library(MKdescr)


#################################################################################
#episodios
#implied Conditional Independencies
dataset <- read.csv("D:/clases/UDES/articulo leishmaniasis/causal_inference/nuevo/data_final.csv")
dataset <- select(dataset, Year, Month, SOI, SST3, SST4, SST34, SST12, NATL, SATL, TROP, forest, vectors, Temp, excess)

#sd units
dataset$SST12 <- zscore(dataset$SST12, na.rm = TRUE)  
dataset$SST3 <- zscore(dataset$SST3, na.rm = TRUE) 
dataset$SST34 <- zscore(dataset$SST34, na.rm = TRUE) 
dataset$SST4 <- zscore(dataset$SST4, na.rm = TRUE) 
dataset$SOI <- zscore(dataset$SOI, na.rm = TRUE) 
dataset$NATL <- zscore(dataset$NATL, na.rm = TRUE) 
dataset$SATL <- zscore(dataset$SATL, na.rm = TRUE) 
dataset$TROP <- zscore(dataset$TROP, na.rm = TRUE)
dataset$forest <- zscore(dataset$forest, na.rm = TRUE) 
dataset$vectors <- zscore(dataset$vectors, na.rm = TRUE) 
dataset$Temp <- zscore(dataset$Temp, na.rm = TRUE) 


#OJO OJO PARA SOLO CASOS COMPLETOS
dataset <- dataset[complete.cases(dataset), ] 
str(dataset)

#descriptive analysis
#ggpairs(dataset)

#DAG 
dag <- dagitty('dag {
excess [pos="0, 0.5"]
Temp  [pos="-1, 0.5"]

SOI [pos="-1.7, 1.2"]
SST3 [pos="-1.8, 1.3"]
SST4 [pos="-1.9, 1.4"]
SST34 [pos="-2, 1.5"]
SST12 [pos="-2.1, 1.6"]
NATL [pos="-2.2, 1.7"]
SATL [pos="-2.3, 1.8"]
TROP [pos="-2.4, 1.9"]
forest [pos="-2.0, -2.1"] 
vectors [pos="-0.5, -1.8"] 
Year [pos="-2.6, -2.1"] 
Month [pos="-2.7, -2.3"] 

SST12 -> SST3
SST12 -> SST34
SST12 -> SST4
SST12 -> SOI
SST12 -> NATL
SST12 -> SATL
SST12 -> TROP


SST3 -> SST34
SST3 -> SST4
SST3 -> SOI
SST3 -> NATL
SST3 -> SATL
SST3 -> TROP

SST34 -> SST4
SST34 -> SOI
SST34 -> NATL
SST34 -> SATL
SST34 -> TROP


SST4 -> SOI
SST4 -> NATL
SST4 -> SATL
SST4 -> TROP

SOI -> NATL
SOI -> SATL
SOI -> TROP


NATL -> SATL
NATL -> TROP

SATL -> TROP


SST12 -> Temp
SST3 -> Temp
SST34 -> Temp
SST4 -> Temp
SOI -> Temp
NATL -> Temp
SATL -> Temp
TROP -> Temp


SST12 -> excess
SST3 -> excess
SST34 -> excess
SST4 -> excess
SOI -> excess
NATL -> excess
SATL -> excess
TROP -> excess


SST12 -> vectors
SST3 -> vectors
SST34 -> vectors
SST4 -> vectors
SOI -> vectors
NATL -> vectors
SATL -> vectors
TROP -> vectors

forest -> vectors
Temp -> vectors
forest -> Temp
forest -> excess

Year -> SST12
Year -> SST3
Year -> SST34
Year -> SST4
Year -> SOI
Year -> NATL
Year -> SATL
Year -> TROP
Year -> Temp
Year -> excess

Month -> SST12
Month -> SST3
Month -> SST34
Month -> SST4
Month -> SOI
Month -> NATL
Month -> SATL
Month -> TROP
Month -> Temp
Month -> excess

vectors -> excess

Temp -> excess



}')  


plot(dag)


## check whether any correlations are perfect (i.e., collinearity)
myCov <- cov(dataset)
round(myCov, 2)

myCor <- cov2cor(myCov)
noDiag <- myCor
diag(noDiag) <- 0
any(noDiag == 1)

## if not, check for multicollinearity (i.e., is one variable a linear combination of 2+ variables?)
det(myCov) < 0
## or
any(eigen(myCov)$values < 0)


## Independencias condicionales
impliedConditionalIndependencies(dag)
corr <- lavCor(dataset)

#plot con ci convencinales (método analitico)
localTests(dag, sample.cov=corr, sample.nobs=nrow(dataset))
plotLocalTestResults(localTests(dag, sample.cov=corr, sample.nobs=nrow(dataset)), xlim=c(-1,1))





#identification
simple_dag <- dagify(
  excess ~  Temp + SST12 + SST3 + SST34 + SST4 + SOI + E_SOI + NATL + SATL +  TROP + wpac + cpac + epac + Rain + Qs + IPM,
  Temp ~ SST12 + SST3 + SST34 + SST4 + SOI + E_SOI + NATL + SATL +  TROP + wpac + cpac + epac, 
  Rain ~ Temp + SST12 + SST3 + SST34 + SST4 + SOI + E_SOI + NATL + SATL +  TROP + wpac + cpac + epac, 
  Qs ~ Temp + SST12 + SST3 + SST34 + SST4 + SOI + E_SOI + NATL + SATL +  TROP + wpac + cpac + epac, 
  IPM ~  Rain + Qs,
  
  SST12 ~ SST3 + SST34 + SST4 + SOI + E_SOI + NATL + SATL +  TROP + wpac + cpac + epac,
  SST3 ~ SST34 + SST4 + SOI + E_SOI + NATL + SATL +  TROP + wpac + cpac + epac,
  SST34 ~ SST4 + SOI + E_SOI + NATL + SATL +  TROP + wpac + cpac + epac,
  SST4 ~ SOI + E_SOI + NATL + SATL +  TROP + wpac + cpac + epac,
  SOI ~ E_SOI + NATL + SATL +  TROP + wpac + cpac + epac,
  E_SOI ~ NATL + SATL +  TROP + wpac + cpac + epac,
  NATL ~ SATL +  TROP + wpac + cpac + epac,
  SATL ~  TROP + wpac + cpac + epac,
  TROP ~ wpac + cpac + epac,
  wpac ~ cpac + epac,
  cpac ~  epac,
  
  Qs ~ Rain,
  
  exposure = "Temp",
  outcome = "excess",
  coords = list(x = c(Temp=2, IPM=1, excess=2, SST12=3, SST3=3.1, SST34=3.2, SST4=3.3, SOI=3.4, E_SOI=3.5, NATL=3.6, SATL=3.7, TROP=3.8, wpac=3.9, cpac=4.0, epac=4.1,
                      Qs=3.5, Rain=3),
                y = c(Temp=2, IPM=1.5, excess=1, SST12=3, SST3=3.1, SST34=3.2, SST4=3.3, SOI=3.4, E_SOI=3.5, NATL=3.6, SATL=3.7, TROP=3.8, wpac=3.9, cpac=4.0, epac=4.1,
                      Qs=1.8, Rain=1.4))
)


# theme_dag() coloca la trama en un fondo blanco sin etiquetas en los ejes
ggdag(simple_dag) + 
  theme_dag()

ggdag_status(simple_dag) +
  theme_dag()


#paths
#paths(simple_dag)

#ggdag_paths(simple_dag) +
#  theme_dag()

#adjusting
adjustmentSets(simple_dag,  type = "minimal")
## {z_miseria, indixes}


ggdag_adjustment_set(simple_dag, shadow = TRUE) +
  theme_dag()



#################################################################################
#rainfall
#implied Conditional Independencies
#dataset <- read.csv("D:/clases/UDES/articulo hepatitis/causal inference/NiñaNiño/obtencion dataset/victor_wendy/ci/dag/Temp_data_final.csv")
#dataset <- filter(dataset, top<=100)
dataset <- select(dataset, excess, SOI, E_SOI, SST3, SST4, SST34, SST12, NATL, SATL, TROP, wpac, cpac, epac, IPM, Qs, Rain)

#OJO OJO PARA SOLO CASOS COMPLETOS
dataset <- dataset[complete.cases(dataset), ] 
str(dataset)

#descriptive analysis
#ggpairs(dataset)

#DAG 
dag <- dagitty('dag {
excess [pos="0, 0.5"]
Rain  [pos="-1, 0.5"]

SOI [pos="-1.6, 1.1"]
E_SOI [pos="-1.7, 1.2"]
SST3 [pos="-1.8, 1.3"]
SST4 [pos="-1.9, 1.4"]
SST34 [pos="-2, 1.5"]
SST12 [pos="-2.1, 1.6"]
NATL [pos="-2.2, 1.7"]
SATL [pos="-2.3, 1.8"]
TROP [pos="-2.4, 1.9"]
wpac [pos="-2.5, 2.0"]
cpac [pos="-2.6, 2.1"]
epac [pos="-2.7, 2.2"]

IPM [pos="-0.2, -0.2"]
Qs [pos="-1.1, -0.5"]


SST12 -> SST3
SST12 -> SST34
SST12 -> SST4
SST12 -> SOI
SST12 -> E_SOI
SST12 -> NATL
SST12 -> SATL
SST12 -> TROP
SST12 -> wpac
SST12 -> cpac
SST12 -> epac


SST3 -> SST34
SST3 -> SST4
SST3 -> SOI
SST3 -> E_SOI
SST3 -> NATL
SST3 -> SATL
SST3 -> TROP
SST3 -> wpac
SST3 -> cpac
SST3 -> epac

SST34 -> SST4
SST34 -> SOI
SST34 -> E_SOI
SST34 -> NATL
SST34 -> SATL
SST34 -> TROP
SST34 -> wpac
SST34 -> cpac
SST34 -> epac

SST4 -> SOI
SST4 -> E_SOI
SST4 -> NATL
SST4 -> SATL
SST4 -> TROP
SST4 -> wpac
SST4 -> cpac
SST4 -> epac

SOI -> E_SOI
SOI -> NATL
SOI -> SATL
SOI -> TROP
SOI -> wpac
SOI -> cpac
SOI -> epac

E_SOI -> NATL
E_SOI -> SATL
E_SOI -> TROP
E_SOI -> wpac
E_SOI -> cpac
E_SOI -> epac

NATL -> SATL
NATL -> TROP
NATL -> wpac
NATL -> cpac
NATL -> epac

SATL -> TROP
SATL -> wpac
SATL -> cpac
SATL -> epac
TROP -> wpac
TROP -> cpac
TROP -> epac

wpac -> cpac
wpac -> epac

cpac -> epac

SST12 -> Rain
SST3 -> Rain
SST34 -> Rain
SST4 -> Rain
SOI -> Rain
E_SOI -> Rain
NATL -> Rain
SATL -> Rain
TROP -> Rain
wpac -> Rain
cpac -> Rain
epac -> Rain

SST12 -> excess
SST3 -> excess
SST34 -> excess
SST4 -> excess
SOI -> excess
E_SOI -> excess
NATL -> excess
SATL -> excess
TROP -> excess
wpac -> excess
cpac -> excess
epac -> excess

SST12 -> Qs
SST3 -> Qs
SST34 -> Qs
SST4 -> Qs
SOI -> Qs
E_SOI -> Qs
NATL -> Qs
SATL -> Qs
TROP -> Qs
wpac -> Qs
cpac -> Qs
epac -> Qs


Rain -> Qs

Rain -> excess
Qs -> excess

Rain -> IPM
Qs -> IPM

Rain -> Qs

IPM -> excess


}')  


plot(dag)


## check whether any correlations are perfect (i.e., collinearity)
myCov <- cov(dataset)
round(myCov, 2)

myCor <- cov2cor(myCov)
noDiag <- myCor
diag(noDiag) <- 0
any(noDiag == 1)

## if not, check for multicollinearity (i.e., is one variable a linear combination of 2+ variables?)
det(myCov) < 0
## or
any(eigen(myCov)$values < 0)


## Independencias condicionales
impliedConditionalIndependencies(dag)
corr <- lavCor(dataset)

#plot con ci convencinales (método analitico)
localTests(dag, sample.cov=corr, sample.nobs=nrow(dataset))
plotLocalTestResults(localTests(dag, sample.cov=corr, sample.nobs=nrow(dataset)), xlim=c(-1,1))





#identification
simple_dag <- dagify(
  excess ~  Rain + SST12 + SST3 + SST34 + SST4 + SOI + E_SOI + NATL + SATL +  TROP + wpac + cpac + epac + Qs + IPM,
  Rain ~ SST12 + SST3 + SST34 + SST4 + SOI + E_SOI + NATL + SATL +  TROP + wpac + cpac + epac, 
  Qs ~ Rain + SST12 + SST3 + SST34 + SST4 + SOI + E_SOI + NATL + SATL +  TROP + wpac + cpac + epac, 
  IPM ~  Rain + Qs,
  
  SST12 ~ SST3 + SST34 + SST4 + SOI + E_SOI + NATL + SATL +  TROP + wpac + cpac + epac,
  SST3 ~ SST34 + SST4 + SOI + E_SOI + NATL + SATL +  TROP + wpac + cpac + epac,
  SST34 ~ SST4 + SOI + E_SOI + NATL + SATL +  TROP + wpac + cpac + epac,
  SST4 ~ SOI + E_SOI + NATL + SATL +  TROP + wpac + cpac + epac,
  SOI ~ E_SOI + NATL + SATL +  TROP + wpac + cpac + epac,
  E_SOI ~ NATL + SATL +  TROP + wpac + cpac + epac,
  NATL ~ SATL +  TROP + wpac + cpac + epac,
  SATL ~  TROP + wpac + cpac + epac,
  TROP ~ wpac + cpac + epac,
  wpac ~ cpac + epac,
  cpac ~  epac,
  
  Qs ~ Rain,
  
  exposure = "Rain",
  outcome = "excess",
  coords = list(x = c(Rain=2, IPM=1, excess=2, SST12=3, SST3=3.1, SST34=3.2, SST4=3.3, SOI=3.4, E_SOI=3.5, NATL=3.6, SATL=3.7, TROP=3.8, wpac=3.9, cpac=4.0, epac=4.1,
                      Qs=3.5),
                y = c(Rain=2, IPM=1.5, excess=1, SST12=3, SST3=3.1, SST34=3.2, SST4=3.3, SOI=3.4, E_SOI=3.5, NATL=3.6, SATL=3.7, TROP=3.8, wpac=3.9, cpac=4.0, epac=4.1,
                      Qs=1.8))
)


# theme_dag() coloca la trama en un fondo blanco sin etiquetas en los ejes
ggdag(simple_dag) + 
  theme_dag()

ggdag_status(simple_dag) +
  theme_dag()


#paths
#paths(simple_dag)

#ggdag_paths(simple_dag) +
#  theme_dag()

#adjusting
adjustmentSets(simple_dag,  type = "minimal")
## {z_miseria, indixes}


ggdag_adjustment_set(simple_dag, shadow = TRUE) +
  theme_dag()



