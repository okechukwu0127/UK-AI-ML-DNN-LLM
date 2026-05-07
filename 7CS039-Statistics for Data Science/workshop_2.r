data <-sample(1:6,1)
View(data)


library(titanic)
library(tidyverse)

data <- titanic_train
View(data)
head(data)

data <- data |> select(-1)
View(data)


data <- data |>
  
  mutate(Survived = factor(Survived, levels = c(0, 1),
                           labels = c("Died", "Survived")),
                            Sex = factor(Sex),
                            Pclass = factor(Pclass))

table(data$Survived)


#We'll start with Survived
#Creates a bar chart showing passenger counts for each survival status
data |>
    ggplot(aes(x = Survived)) +
    geom_bar(width = 0.4, fill = "yellow", color = "red") +
    theme_classic() +
      theme(
       plot.title = element_text(family = "Times New Roman", hjust = 0.5),
        axis.text = element_text(family = "Times New Roman",face = "bold"),
        axis.title = element_text(family = "Times New Roman", face = "bold")
       ) +
    labs(title = "Overall Survival Rates", x = NULL, y = "Passenger count")


#Now we investigate survival rate by Sex
data |>
  ggplot(aes(x = Sex, fill = Survived)) +
  geom_bar(width = 0.4) +
  theme_classic() +
  theme(
    plot.title = element_text(family = "Times New Roman", hjust = 0.5),
    axis.text = element_text(family = "Times New Roman",face = "bold"),
    axis.title = element_text(family = "Times New Roman", face = "bold"),
    legend.title = element_blank(),
    legend.text = element_text(family = "Times New Roman")
  ) +
  labs(title = "Survival rates by Sex", x = NULL, y = "Passenger count")


#Now we investigate survival rate by Pclass
data |>
  ggplot(aes(x = Pclass, fill = Survived)) +
  geom_bar(width = 0.4) +
  theme_classic() +
  theme(
    plot.title = element_text(family = "Times New Roman", hjust = 0.5),
    axis.text = element_text(family = "Times New Roman",face = "bold"),
    axis.title = element_text(family = "Times New Roman", face = "bold"),
    legend.title = element_blank(),
    legend.text = element_text(family = "Times New Roman")
  ) +
  labs(title = "Survival rates by Passenger Class", x = NULL, y = "Passenger count")


#To see how survival rates differ by age we can use a histogram and a boxplot.
data %>%
  ggplot(aes(x = Age, fill = Survived)) +
  geom_histogram( color = "grey") +
  theme_classic() +
  scale_fill_manual(values = c("Died" ="blue", "Survived" = "green")) +
  theme(
    plot.title = element_text(family = "Times New Roman", hjust = 0.5),
    axis.text = element_text(family = "Times New Roman",face = "bold"),
    axis.title = element_text(family = "Times New Roman", face = "bold"),
    legend.title = element_blank(),
    legend.text = element_text(family = "Times New Roman")
  ) +
  labs(title = "Survival rates by Age")






data %>%
  ggplot(aes(x = Survived, y = Age)) +
  geom_boxplot() +
  theme_classic() +
  theme(
    plot.title = element_text(family = "Times New Roman", hjust = 0.5),
    axis.text = element_text(family = "Times New Roman", face = "bold"),
    axis.title = element_text(family = "Times New Roman", face = "bold"),
    legend.title = element_blank(),
    legend.text = element_text(family = "Times New Roman")
  ) +
  labs(title = "Survival rates by Age", x = NULL)


#Survived and Pclass
data %>%
  ggplot(aes(x = Sex, fill = Survived)) +
  geom_bar(width = 0.4) +
  facet_wrap(~ Pclass) +
  theme_test() +
  theme(
    plot.title = element_text(family = "Times New Roman", hjust = 0.5),
    axis.text = element_text(family = "Times New Roman", face = "bold"),
    axis.title = element_text(family = "Times New Roman", face = "bold"),
    legend.title = element_blank(),
    legend.text = element_text(family = "Times New Roman")
  ) +
  labs(title = "Survival rates Sex and Passenger class",
       x = NULL, y = "Passenger Count")



# Survived, Age, Sex, Pclass
data %>%
  ggplot(aes(x = Age, fill = Survived)) +
  geom_histogram(width = 0.4, color = "red") +
  facet_wrap(~Sex + Pclass) +
  scale_fill_manual(values = c("Died" ="blue", "Survived" = "green")) +
  theme_test() +
  theme(
    plot.title = element_text(family = "Times New Roman", hjust = 0.5),
    axis.text = element_text(family = "Times New Roman", face = "bold"),
    axis.title = element_text(family = "Times New Roman", face = "bold"),
    legend.title = element_blank(),
    legend.text = element_text(family = "Times New Roman")
  ) +
  labs(title = "Survival rates Age, Sex and Passenger class")

