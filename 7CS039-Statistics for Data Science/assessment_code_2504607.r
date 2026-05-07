#Load required libraries
library(tidyverse)

# Load the dataset
melanoma <- read.csv("/Users/oeze/Documents/wlv/7CS039/melanoma.csv")


############################################################
# DATA PREPARATION
############################################################

melanoma$sex <- factor(melanoma$sex, levels = c(0,1),
                       labels = c("Female","Male"))

melanoma$ulcer <- factor(melanoma$ulcer, levels = c(0,1),
                         labels = c("Absent","Present"))

melanoma$status <- factor(melanoma$status,
                          levels = c(1,2,3),
                          labels = c("Died of melanoma",
                                     "Alive",
                                     "Died of other causes"))

str(melanoma)
summary(melanoma)
View(melanoma)


############################################################
# (i) SUMMARY STATISTICS
############################################################

summary(melanoma$time)
summary(melanoma$age)
summary(melanoma$thickness)

table(melanoma$sex)
prop.table(table(melanoma$sex))

table(melanoma$ulcer)
table(melanoma$status)


############################################################
# (ii) GRAPHICAL SUMMARIES
############################################################
par(mfrow = c(3, 3))
# Histograms
hist(melanoma$time,
     main="Distribution of Survival Time",
     xlab="Survival Time (days)")

hist(melanoma$age,
     main="Distribution of Age",
     xlab="Age (years)")

hist(melanoma$thickness,
     main="Distribution of Tumour Thickness",
     xlab="Thickness (mm)")

# Boxplots
boxplot(time ~ sex, data=melanoma,
        main="Survival Time by Gender",
        xlab="Gender",
        ylab="Survival Time (days)")

boxplot(thickness ~ ulcer, data=melanoma,
        main="Tumour Thickness by Ulceration Status",
        xlab="Ulceration",
        ylab="Thickness (mm)")

# Bar plot
barplot(table(melanoma$status),
        main="Patient Status at End of Study",
        ylab="Frequency")


############################################################
# (iii) REGRESSION AND CORRELATION ANALYSIS
############################################################

# Scatterplots with regression lines
plot(time ~ thickness, data=melanoma,
     main="Survival Time vs Tumour Thickness",
     xlab="Thickness (mm)",
     ylab="Survival Time (days)")
abline(lm(time ~ thickness, data=melanoma), col="red")

plot(time ~ age, data=melanoma,
     main="Survival Time vs Age",
     xlab="Age (years)",
     ylab="Survival Time (days)")
abline(lm(time ~ age, data=melanoma), col="red")

plot(thickness ~ age, data=melanoma,
     main="Tumour Thickness vs Age",
     xlab="Age (years)",
     ylab="Thickness (mm)")
abline(lm(thickness ~ age, data=melanoma), col="red")

# Linear regression models
model_time_thickness <- lm(time ~ thickness, data=melanoma)
summary(model_time_thickness)

model_time_age <- lm(time ~ age, data=melanoma)
summary(model_time_age)

model_thickness_age <- lm(thickness ~ age, data=melanoma)
summary(model_thickness_age)

# Correlations
cor(melanoma$time, melanoma$thickness)
cor(melanoma$time, melanoma$age)
cor(melanoma$thickness, melanoma$age)


############################################################
# (v) TWO-SAMPLE TESTS BY GENDER
############################################################

t.test(time ~ sex, data=melanoma)
t.test(age ~ sex, data=melanoma)
wilcox.test(thickness ~ sex, data=melanoma)



############################################################
# (vi) QQ-PLOTS
############################################################

par(mfrow = c(3, 3))

qqnorm(melanoma$time,
       main = "QQ-Plot of Survival Time")
qqline(melanoma$time)

qqnorm(melanoma$age,
       main = "QQ-Plot of Age")
qqline(melanoma$age)



qqnorm(melanoma$thickness,
       main = "QQ-Plot of Tumour Thickness")
qqline(melanoma$thickness)

qqnorm(melanoma$age[melanoma$sex == "Male"],
       main = "QQ-Plot of Age (Male)")
qqline(melanoma$age[melanoma$sex == "Male"])



qqnorm(melanoma$age[melanoma$sex == "Female"],
       main = "QQ-Plot of Age (Female)")
qqline(melanoma$age[melanoma$sex == "Female"])

qqnorm(melanoma$thickness[melanoma$sex == "Male"],
       main = "QQ-Plot of Tumour Thickness (Male)")
qqline(melanoma$thickness[melanoma$sex == "Male"])


qqnorm(melanoma$thickness[melanoma$sex == "Female"],
       main = "QQ-Plot of Tumour Thickness (Female)")
qqline(melanoma$thickness[melanoma$sex == "Female"])

