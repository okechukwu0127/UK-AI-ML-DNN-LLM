library(tidyverse); 
#library(broom); 
#library(lmtest); 
#library(sandwich)
library(here)
library(ggplot2)

melanoma <- read_csv("/Users/oeze/Documents/wlv/7CS039/melanoma.csv")
View(melanoma)

summary(melanoma)
