library(tidyverse); 
#library(broom); 
#library(lmtest); 
#library(sandwich)
library(here)

#This code performs a correlation and simple linear 
#regression analysis to investigate the relationship 
#between a person's Age and their bodyfat percentage
Bodyfat <- read_csv("/Users/oeze/Documents/wlv/7CS039/Bodyfat.csv")


#This command adds the Bodyfat data frame to R's search path
#you can type Age instead of Bodyfat$Age)
attach(Bodyfat)

#This calculates the Pearson correlation coefficient (r) 
#between the Age and bodyfat variables.
cor(Age, bodyfat, method="pearson") 

#r = 0.291
#This indicates a weak positive linear relationship
#The value of 0.291 is closer to 0 (no relationship) 
#than to 1 (perfect relationship)

plot(Age, bodyfat)

#Now let's build our regression model (note that it's y∼x).
my_model=lm(formula = bodyfat~Age)

my_model

#Coefficients:
#(Intercept)          Age  
#10.4633       0.1936 
#y = my_model= 0.19x + 10.5
#bodyfat = 10.4633 + 0.1936 * Age


summary(my_model)
#Residuals:
#  Min      1Q      Median   3Q      Max 
#-18.2053  -6.1513   0.2979   5.3665  27.1656 

#Coefficients:
#  Estimate Std. Error t value Pr(>|t|)    
#(Intercept) 10.46326    1.87278   5.587 6.02e-08 ***
#  Age          0.19355    0.04018   4.818 2.52e-06 ***


#Residual standard error: 8.021 on 250 degrees of freedom
#Multiple R-squared:  0.08495,	Adjusted R-squared:  0.08129 
#F-statistic: 23.21 on 1 and 250 DF,  p-value: 2.522e-06

#and we see that r = √0.084985 ≃ 0.291 as before. How do we interpret this value for r?


#Repeat the above analysis for any choice of two variables from the dataset.

# Clear any existing plot devices and reset parameters
if(!is.null(dev.list())) dev.off()
par(mar = c(3, 3, 2, 1))  # Much smaller margins


## 1. Check the correlation between Weight and Abdomen
cor_weight_abdomen <- cor(Weight, Abdomen, method="pearson")

plot(Weight, Abdomen, 
     main = "Scatter Plot: Weight vs Abdomen Circumference",
     xlab = "Weight (lbs)", 
     ylab = "Abdomen Circumference (cm)",
     pch = 16, col = "blue")
grid()


# 3. Build regression model
abdomen_model <- lm(Abdomen ~ Weight)


# 4. Add regression line
abline(abdomen_model, col = "red", lwd = 2)


# 5. Display results
print(summary(abdomen_model))



# Using ggplot2 which handles margins more automatically
ggplot(Bodyfat, aes(x = Weight, y = Abdomen)) +
  geom_point(color = "blue", alpha = 0.7) +
  geom_smooth(method = "lm", color = "red", se = TRUE) +
  labs(title = "Weight vs Abdomen Circumference",
       x = "Weight (lbs)",
       y = "Abdomen Circumference (cm)") +
  theme_minimal()


age <- Bodyfat$Age
abdomen <- Bodyfat$Abdomen
data <- data.frame(age, abdomen)


cor(data)
LinReg <- lm(abdomen ~ age)
abline(LinReg, col = "green", lwd = 2)

predict(LinReg, data.frame(age = 40))
pred_LinReg <- predict(LinReg, data.frame(age = seq(from = 0, to = 80, by = 5)))
names(pred_LinReg) <- seq(0, 80, 5)
View(pred_LinReg)

# Plot the real data
plot(data$age, data$abdomen, pch = 16,
     xlab = "Age", ylab = "Abdomen Circumference (cm)",
     main = "Real Bodyfat Data: Age vs Abdomen")

