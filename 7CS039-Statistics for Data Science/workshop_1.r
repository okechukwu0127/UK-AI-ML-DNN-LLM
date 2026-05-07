library(tidyr)

View(airquality)
AirTemp <- airquality
mean(AirTemp$Temp)
View(AirTemp[1:5,])


#clean the data by removing NA's columng
clean_airquality <- drop_na(AirTemp)
head(clean_airquality)


 View(clean_airquality)
 
 # ============================================
 #Find the mean reading for all of the appropriate
 #variables by frstly removing the NA values.
 # ============================================
 
mean(AirTemp$Ozone, na.rm = TRUE) #Remove NAs in the mean calculation
mean(AirTemp$Solar.R, na.rm = TRUE) #Remove NAs in the mean calculation

mean(AirTemp$Wind, na.rm = TRUE) #Remove NAs in the mean calculation
mean(AirTemp$Month, na.rm = TRUE) #Remove NAs in the mean calculation
mean(AirTemp$Day, na.rm = TRUE) #Remove NAs in the mean calculation

# ============================================
#Variance & Standard Deviation
# ============================================
#compute the variance 
var(airquality$Temp)

#standard deviation 
sd(airquality$Temp)


#The Median
median(airquality$Temp)


#compute the five number summary for the variable Temp
summary(airquality$Temp)

summary(airquality)

#Notice that Mont andDay are coded as numeric variables even though they are
#clearly nominal/categorical. This can be mended as follows

AirTemp$Month = factor(AirTemp$Month)
AirTemp$Day = factor(AirTemp$Day)
summary(AirTemp)
hist(AirTemp$Temp, main="Temperature History")
help(hist)


par(mfrow=c(2,2))

hist(airquality$Temp)
hist(airquality$Ozone)
hist(airquality$Wind)
hist(airquality$Solar.R)


par(mfrow=c(1,1))
boxplot(airquality$Temp)

# comparing the fluctuations in temperature across months.
boxplot(airquality$Temp ~ airquality$Month)

par(mfrow=c(1,1))
#Enhanced Boxplot with Better Formatting:
boxplot(airquality$Temp ~ airquality$Month,
        main = "Temperature Distribution by Month",
        xlab = "Month", 
        ylab = "Temperature (°F)",
        col = "lightblue",
        names = c("May", "June", "July", "August", "September"))


# ============================================
#Interpretation of Your Boxplot:
# ============================================
# Each box shows temperature distribution for that month:
# Middle line = median temperature
# Box = interquartile range (middle 50% of temperatures)
# Whiskers = range of typical values
# Dots (○) = potential outliers (Unusual values)


#Shows how ozone concentrations vary across months, likely higher in summer months.
boxplot(airquality$Ozone ~ airquality$Month, 
        main = "Ozone Levels by Month",
        xlab = "Month", ylab = "Ozone")

#Reveals seasonal wind patterns - some months may be windier than others.
boxplot(airquality$Wind ~ airquality$Month,
        main = "Wind Speed by Month",
        xlab = "Month", ylab = "Wind Speed")


par(mfrow = c(2, 2))  # 2x2 grid of plots
boxplot(airquality$Ozone ~ airquality$Month, main = "Ozone by Month", xlab = "Month", ylab = "Ozone")
boxplot(airquality$Solar.R ~ airquality$Month, main = "Solar.R by Month", xlab = "Month", ylab = "Solar R")
boxplot(airquality$Wind ~ airquality$Month, main = "Wind by Month", xlab = "Month", ylab = "Wind Speed")
boxplot(airquality$Temp ~ airquality$Month, main = "Temp by Month", xlab = "Month", ylab = "Temperature")
