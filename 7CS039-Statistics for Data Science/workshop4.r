library(MASS)
library(tidyverse)
head(birthwt)


birthwt <- as_tibble(MASS::birthwt) #tibble instead of a traditional data frame.


head(birthwt)

#we relabel the 0 as no and the 1 as yes for the columns smoke, ht, ui, low
birthwt <- birthwt %>%
  mutate(race = recode_factor(race, '1' = "white",'2' = "black", '3' = "other")) %>%
  mutate_at(c("smoke", "ht", "ui", "low"),
            ~ recode_factor(.x, '0' = "no", '1' = "yes"))

head(birthwt)


#Let's start by looking at the relationship between birth weights and whether or not the mother smokes.

qplot(x = smoke, y = bwt,
      geom = "boxplot", data = birthwt,
      xlab = "Mother smokes",
      ylab = "Birthweight (grams)",
      fill = I("pink"))



#The first thing we want to do is extract the data for mothers who smoke
mothers_who_smokes <- filter(birthwt, smoke=='yes')
mean(mothers_who_smokes$bwt)


#We can use the following code to calculate for the no and yes group
birthwt %>%
   group_by(smoke) %>%
   summarize(mean_bwt = mean(bwt),
               sd_bwt = sd(bwt))





#Let's also calculate the standard error which takes into account the sizes of the samples.

birthwt %>%
  group_by(smoke) %>%
  summarize(num.obs = n(),
            mean_bwt = round(mean(bwt), 0),
            sd_bwt = round(sd(bwt), 0),
            se_bwt = round(sd(bwt) / sqrt(num.obs), 0))


#Two Sample Test for the Difference between two Population Means


bwt_t_test <- t.test(bwt ~ smoke, data = birthwt)
bwt_t_test


#QQ-plot (quantile-quantile plot
#We use the stat_qq() function

p_bwt <- ggplot(data = birthwt, aes(sample = bwt))
p_bwt + stat_qq() + stat_qq_line() +
  labs(
    x = "Your X-axis label here",
    y = "Your Y-axis label here",
    title = "Optional title"
  )

#We can produce separate [plots for smokers and non-smokers
p_bwt + stat_qq() + stat_qq_line() + facet_grid(. ~ smoke) +labs(
  x = "Your X-axis label here",
  y = "Your Y-axis label here",
  title = "Optional title"
)

#Repeat the above analysis and investigate the relationship between the moth-
#ers weight at the time of the last menstrual period and the mothers status as
#a smoker/non-smoker. What about the other variables?

qplot(x = smoke, y = lwt,
      geom = "boxplot", data = birthwt,
      xlab = "Mother smokes",
      ylab = "Mothers Weight at the time of the last menstrual",
      fill = I("green"))


birthwt %>%
  group_by(smoke) %>%
  summarize(num.obs = n(), #size of sample
            mean_lwt = mean(lwt), #mean of lwt in the relaionship
            sd_lwt = sd(lwt), #standard deviation of lwt
            se_bwt = round(sd(bwt) / sqrt(num.obs), 0)) #standard error


lwt_t_test <- t.test(lwt ~ smoke, data = birthwt)
lwt_t_test

