###########################################################################################################
#                                                                                                         #  
#     PLEASE NOTE THAT YOU SHOULD HAVE AN USER AND PASSWORD TO DOWNLOAD DATA FROM NASA GIOVANNI           #
#     AND PREVIOUSLY YOU HAVE TO SET UP WGET IN YOUR PC                                                   #
#                                                                                                         #          
###########################################################################################################                                                                                                       #

NASA_user <- 'juandavidgutier'
NASA_pw <- 'Gatolochis2210'

#-----------------------------VARIABLES TO MODIFY--------------------------------------------------------------------------------------------

directory_save <- "D:/clases/UDES/scripts/nasa/script mensual/descargas/"

#position of name in the .csv file with the links list 
initial_position_of_name = 144
final_position_of_name = 157

#----------------------------------------------------------------------------------------------------------------------------------------------


list = read.csv("D:/clases/UDES/articulo leishmaniasis/causal_inference/manuscrito/script y data/descargas/subset_GLDAS_NOAH025_M_2.1_20220828_135941.csv", sep=",", header=FALSE, stringsAsFactors=FALSE)
str(list)

save_directory_path <-directory_save

x <- length(list[,1])
name <- c(1:x)

for(i in (1:x)){
  url = list[i,1]
  name[i] <- substr(list[i,1], initial_position_of_name, final_position_of_name)

  
  download.file(url = url,
                destfile = paste(save_directory_path, name[i] , sep = ""),
                method = 'wget',
                extra = paste('--load-cookies ~/.urs_cookies --save-cookies ~/.urs_cookies --keep-session-cookies --user=', NASA_user, ' --password=', NASA_pw, ' --content-disposition', sep = ""))
  }





