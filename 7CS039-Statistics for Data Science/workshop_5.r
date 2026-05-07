library(tidyverse); 
#library(broom); 
#library(lmtest); 
#library(sandwich)
library(here)
library(ggplot2)

Bodyfat <- read_csv("/Users/oeze/Documents/wlv/7CS039/Bodyfat-1.csv")
View(Bodyfat)

summary(Bodyfat)

# Histogram (shows frequency distribution)
# Resize plot window before running hist()
par(mar = c(6, 5, 4, 2))  # Bottom, Left, Top, Right margins
#par(mar = c(5, 4, 4, 2) + 0.1)  # Default margins
par(mfrow = c(2, 2))  # 2 rows, 2 columns

hist(Bodyfat$Weight, 
     main = "Frequency Distribution of Weight",
     xlab = "Weight (lbs)",
     ylab = "Frequency",
     col = "lightblue")

hist(Bodyfat$Age, 
     main = "Frequency Distribution of Age",
     xlab = "Age",
     ylab = "Frequency",
     col = "lightblue")

hist(Bodyfat$Biceps, 
     main = "Frequency Distribution of Biceps",
     xlab = "Biceps",
     ylab = "Frequency",
     col = "lightblue")


hist(Bodyfat$Wrist, 
     main = "Frequency Distribution of Wrists",
     xlab = "Wrists",
     ylab = "Frequency",
     col = "lightblue")


par(mfrow = c(1, 1))  # 2 rows, 2 columns
plot(Age, bodyfat)

my_model=lm(formula = bodyfat~Age)


# Select only numeric variables for correlation
numeric_vars <- Bodyfat %>% 
  select_if(is.numeric)

# Calculate correlation matrix
cor_matrix <- cor(numeric_vars, use = "complete.obs")
round(cor_matrix, 3)  # Rounded to 3 decimals

# Or view as a nice table
library(knitr)
kable(round(cor_matrix, 3))



# Find strong correlations
strong_cor <- which(abs(cor_matrix) > 0.7 & cor_matrix < 1, arr.ind = TRUE)
strong_cor_pairs <- data.frame(
  Var1 = rownames(cor_matrix)[strong_cor[,1]],
  Var2 = colnames(cor_matrix)[strong_cor[,2]],
  Correlation = cor_matrix[strong_cor]
)
strong_cor_pairs


