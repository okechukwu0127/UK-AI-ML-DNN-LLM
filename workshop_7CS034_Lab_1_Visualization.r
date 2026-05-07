library(tidyverse)
#View(penguins)
# Load the packages
library(palmerpenguins)
library(ggplot2)
library(dplyr)
library(ggthemes)

# View the dataset
head(penguins)


# View Penguins Table
View(penguins)

# Check the structure
str(penguins)


# Get help on the dataset
?penguins

ggplot(data = penguins,
      # aes(x = flipper_length_mm, y = body_mass_g)
       mapping = aes(x = flipper_length_mm, y = body_mass_g)
       )+
        geom_point(size = 2, alpha = 0.7,mapping = aes(color = species, shape=species))+
        geom_smooth(method = "lm")+
  scale_color_colorblind()
  #scale_color_discrete()



# Q1. How many rows are in penguins? How many columns?
#Answer: The penguins data frame contains 344 rows (observations) and 8 columns (variables).
#This is stated in the text: “penguins contains 344

# Q2. What does the bill_depth_mm variable in the penguins data frame describe? Read
#the help for ?penguins to find out.
#Answer : The bill_depth_mm variable describes the depth (thickness) of a penguin’s bill (beak) in millimeters.

# Q3. Make a scatterplot of bill_depth_mm vs. bill_length_mm. That is, make a scat-
#  terplot with bill_depth_mm on the y-axis and bill_length_mm on the x-axis. De-
#  scribe the relationship between these two variables.


#A
ggplot(data = penguins,
       aes(x=bill_length_mm, y =bill_depth_mm, color =species)
       
) +
  geom_point(size =3, alpha = 0.8)+
  geom_smooth(method = "lm", se=TRUE) +
  labs(title= "Bill Dept vs Bill Length",
        x="Bill Length (mm)",
       y="Bill Dept (mm)")

#The RelationShip between x and y variable
#Adelie have more Bill Length
#Chinstrap has more or longer bill compared to the other species
#Gentoo has slightly more build Deps than Chintrap

#B
ggplot(data = penguins,
       aes(x=bill_depth_mm, y =bill_length_mm)
       
) +
  geom_point(size =3, alpha = 0.8)+
  geom_smooth(method = "lm", se=TRUE) +
  labs(title= "Bill Dept vs Bill Length",
       x="Bill Length (mm)",
       y="Bill Dept (mm)")
    #this shows how the plan moves across the points based on the dept and bills clustering.
    #NOT BEST TO SHOW THE RELAITONSHIP BETWN THE VARIABLES





#Q4. What happens if you make a scatterplot of species vs. bill_depth_mm? What
#might be a better choice of geom?


ggplot(data = penguins,
        mapping = aes(x=species, y=bill_depth_mm)
       )+
  geom_boxplot(size =1.5, mapping = aes(color = species))+theme_minimal()

    #BEST GEOM would be boxplot or geom_violin or even geom_jitter

# Q5. Why does the following give an error and how would you fix it?
    ggplot(data = penguins)+geom_point()
    #The code above would return and error because x and y aesthetics  (x and y coordinates)  was not defined. 
    #Only the penguins data was loaded

