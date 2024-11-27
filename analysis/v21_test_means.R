# this script copies data processing and mean calculation 
# for V21 Maze test items
# cf. https://github.com/wilcoxeg/maze_src_orc/blob/main/scripts/imaze_src-orc_analysis.Rmd


library(tidyverse)
library(urltools)
library(dplyr)
library(plotrix)

d = read.csv('src-orc-results.csv', comment = "#") %>%
  mutate(rt = as.numeric(as.character(rt)),
         correct = if_else(correct=="no", F, T))

#Remove data that is after mistake
data_no_na<- d %>% filter(!(is.na(rt)))
message("Removed ", format(100-100*nrow(data_no_na)/nrow(d), digits=4), "% of the data for being na (after a mistake).")

#Find standard deviation and mean of reading time
stdev_rt = sd(data_no_na$rt)
mean_rt = mean(data_no_na$rt)

#Changed data that is more than 2 standard deviations from mean to become NA this means that in the next cell when we sum by reading time, regions that 
# have some of data that is an outlier will become an NA
data_cleaned <- d %>% mutate(rt = replace(rt, rt > mean_rt + 2*stdev_rt, NA)) %>% mutate(rt = replace(rt, rt < mean_rt - 2*stdev_rt, NA))

message("Filtered away all reading times off by 2 standard deviations. This constitutes ", format(nrow(filter(d, rt > mean_rt + 2*stdev_rt)) + nrow(filter(d, rt < mean_rt - 2*stdev_rt))), " words or ", format(100*(nrow(filter(d, rt > mean_rt + 2*stdev_rt)) + nrow(filter(d, rt < mean_rt - 2*stdev_rt))) / nrow(data_no_na), digits=4), "% words across the participants.")

# Get by-region sums
mean_df = data_cleaned %>%
  filter(type == "obj_rc" | type == "subj_rc") %>%
  group_by(MD5, group, type, word_number, region_number, correct) %>% 
  summarise(total_rt=mean(rt), 
            all_correct=all(correct)) %>%
  ungroup() %>%
  filter(!(is.na(total_rt)))


mean_df %>%
  mutate(rc_word = case_when(
    type == "subj_rc" & region_number == 5 ~ "verb",
    type == "subj_rc" & region_number == 6 ~ "the",
    type == "subj_rc" & region_number == 7 ~ "noun",
    type == "obj_rc" & region_number == 3 ~ "the",
    type == "obj_rc" & region_number == 4 ~ "noun",
    type == "obj_rc" & region_number == 5 ~ "verb"
  )) %>%
  mutate(rc_word = factor(rc_word, levels = c("the", "noun", "verb"))) %>%
  filter(!is.na(rc_word), correct == T) %>%
  group_by(type, rc_word) %>%
  summarise(m=mean(total_rt),
            s = std.error(total_rt),
            upper = m + s * 1.96,
            lower = m - s * 1.96) %>%
  ungroup()

