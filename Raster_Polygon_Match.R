###########################################################################################################
#                                                                                                         #  
#     PLEASE NOTE THAT YOU HAVE TO DOWNLOAD PREVIOUSLY THE .NC4 FILES FROM NASA GIOVANNI                  #
#                                                                                                         #
#                                                                                                         #          
###########################################################################################################      



install.packages("rasterVis")
install.packages("maptools")
install.packages("maps")
install.packages("hash")


library(raster)
library(ncdf4)
library(maptools)
library(hash)



directory_list = "D:/subset_GLDAS_NOAH025_M_2.1_20220828_135941.csv"

#change next lines to work with EVI index#
initial_position_of_name = 144
final_position_of_name = 157
initial_position_of_year_in_name = 1
final_position_of_year_in_name = 4
initial_position_of_month_in_name = 5
final_position_of_month_in_name = 6
variables <- c("Qs_acc", "SoilTMP0_10cm_inst", "SoilMoi0_10cm_inst", "Rainf_tavg", "Tair_f_inst")

#Directory with .nc4 files
dir = "D:/"
#Paht for the shape file of Colombia
munShapePath <- "D:/Municipios wgs84_Disolv.shp"
#Path to save data
destination_file = "D:/Data_Nasa.csv"

#Number of municipalities in shape file
qtyMun <- 1122
#------------------------------------------------------------------------------------------------------------------------------------------

#list of namonth
list = read.csv(directory_list, header=FALSE, stringsAsFactors=FALSE)
x <- length(list[,1])
namonth <- c(1:x)
for(i in c(1:x)){
  namonth[i] <- substr(list[i,1], initial_position_of_name, final_position_of_name)
}

#read shape file
munShape <- readShapePoly(munShapePath)
#Dataframe with table of sahpe file
munShape.df <- as(munShape, "data.frame")
#Table to save values for municipality
data_table <- 0

#pixel match
for (period in 1:length(namonth)){
  print(period)
  file_name = namonth[period]
  full_path = paste0(dir, file_name)
  data = array(0, c(qtyMun,6+length(variables)))
  columns <- c("ID", "Code_DANE", "year", "month","period", "Code_DANE-period")
  columns <- c(columns, variables)
  colnamonth(data) <- columns
  for(var in 1:length(variables)){
    
    var_raster <- raster(full_path, varname=variables[var])
    data_var <- extract(var_raster, munShape, fun=NULL, na.rm=FALSE, weights=TRUE, normalizeWeights=TRUE, cellnumbers=TRUE, small=FALSE, df=FALSE,factors=TRUE, sp=FALSE)
      for(i in 1:qtyMun){
      values <- data_var[[i]][, 2]
      weights <- data_var[[i]][, 3]
      weighted_value_mun <- 0
      
      if (is.null(values)){
        weighted_value_mun=NA
      }else{
        
        for(j in 1:length(values)){
          
          if(is.na(values[j])){
            weighted_value_mun = weighted_value_mun + (mean(values, na.rm=TRUE) * weights[j])
  
          }else{      
            weighted_value_mun = weighted_value_mun + (values[j]*weights[j])
  
          }
        }
      }  
      DANE_code = munShape.df[i,"Codigo_DAN"]
      year = substr(namonth[period],initial_position_of_year_in_name, final_position_of_year_in_name )
      month = substr(namonth[period], initial_position_of_month_in_name, final_position_of_month_in_name)
      stringPeriod = ""
      if(period < 10){
        stringPeriod = paste0("00", period)
      }else{
        if(period < 100){
          stringPeriod = paste0("0", period)
        }else{
          if(period < 1000){
            stringPeriod = paste0("", period)
          }else{
            stringPeriod = period  
          }
        }
      }
      data[i,1] = i
      data[i,2] = DANE_code
      data[i,3] = year
      data[i,4] = month
      data[i,5] = period
      data[i,6] = paste0(DANE_code, stringPeriod)
      data[i,6+var] = weighted_value_mun
    }
    
  } 
  if(period == 1){
    data_table = data
  }else{
    data_table = rbind(data_table, data)
  }
  
}

View(data_table)

#save data in a .csv
write.csv(data_table, file = destination_file)



