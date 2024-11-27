library(data.table)
library(magrittr)
library(stringi)
library(purrr)
library(ggplot2)
library(patchwork)
library(lme4)

# Load LCS model predictions for RC stimuli and fillers

rc_dir = "LCS_output/"

lcs_rc_preds = list.files(rc_dir, pattern=".txt") %>%
  lapply(function(f) fread(paste0(rc_dir,f)) %>% .[ , fname := f]) %>%
  rbindlist %>%
  .[ , c("language", "_1", "_2", "_3", "delrate", 
         "_4", "model_id",  "_5", "plm", "_6", "run_id") := tstrsplit(fname, "_") ] %>%
  .[ , c("Expt", "ItemID", "Condition") := SentenceID %>%
       stri_replace_all_fixed("Vani_etal_2021.E", "Vani_etal_2021-E") %>%
       stri_replace_all_fixed("Staub_2010.E", "Staub_2010-E") %>%
       tstrsplit("\\.")] %>%
  .[Condition %in% c("ORC", "SRC"), Condition := c("ORC"="obj_rc", "SRC"="subj_rc")[Condition]]


# Get Vani et al filler sentence predictions

studyID = "Vani_etal_2021-E1"

lcs.v21.fillers = lcs_rc_preds %>%
  .[stri_startswith_fixed(Expt, studyID)] %>%
  .[Condition == "filler"] %>%
  .[ , .(Surprisal = mean(Surprisal), # average over repetitions etc
         SurprisalReweighted = mean(SurprisalReweighted)), 
     .(Sentence, SentenceID, Expt, Condition, ItemID, Region, Word,
       mID=model_id, delrate)] %>%
  .[ ,  ItemID:=as.numeric(ItemID)] 

# Read Maze RT data 

rt.v21.fillers =  "https://raw.githubusercontent.com/wilcoxeg/maze_src_orc/refs/heads/main/data/src-orc-results.csv" %>%
  #"src-orc-results.csv" %>%
  fread %>%
  .[type  == "filler" & correct == "yes"] %>%
  .[ , `:=`(Expt = studyID,
            rt = as.numeric(rt))] %>%
  .[rt > (mean(rt) - 2*sd(rt)) & rt < (mean(rt) + 2*sd(rt))] %>% # filter outliers as in paper
  .[ , `:=`(LogRT = log(rt),
            Word = stri_trans_tolower(word),
            ItemWordIndex = as.numeric(word_number),
            PrevWordIndex = as.numeric(word_number)-1,
            SubjectID=MD5,
            ItemID=item)] %>%
  .[PrevWordIndex == -1, PrevWordIndex := NA] %>%
  .[ , WordLength := nchar(word)] %>%
  .[is.na(PrevWordIndex) | PrevWordIndex < ItemWordIndex]

# Optional: put code here to add word frequency from CELEX,
#  which we use as a predictor in the paper.
#  uncomment relevant lines to include

# Merge LCS and RT

rt.v21.fillers = rt.v21.fillers %>% 
  .[lcs.v21.fillers, 
    on=.(ItemID, Word, ItemWordIndex = Region),
    allow.cartesian=TRUE] 

rt.v21.fillers = rt.v21.fillers%>%
  .[rt.v21.fillers[ , 
                    .(PrevWordIndex = ItemWordIndex,
                      PrevSurprisal = Surprisal,
                      PrevSurprisalReweighted = SurprisalReweighted,
                      PrevLogRT = LogRT,
                      #PrevLogWordFreq = LogWordFreq,
                      PrevWordLength = WordLength,
                      SubjectID, ItemID, mID, delrate)],
    on=.(SubjectID, ItemID, PrevWordIndex, mID, delrate)] %>%
  #.[!is.na(LogWordFreq) & !is.na(PrevLogWordFreq) & is.finite(PrevLogWordFreq)] %>%
  .[!is.na(PrevSurprisal) & !is.na(time)]

# Fit LME and extract AIC for each LCS model instance

rt.filler.fits = rt.v21.fillers[ ,
                                 .(AIC = AIC(lmer(LogRT ~ # depvar - log RT on target word. for filler, this is all words
                                                    SurprisalReweighted + # LCS surprisal for target word
                                                    ItemWordIndex + # position in sentence
                                                    #LogWordFreq + # log word frequency from CELEX
                                                    WordLength + # target word length
                                                    PrevSurprisal + # same predictors but for immediately previous word
                                                    #PrevLogWordFreq + 
                                                    PrevWordLength + 
                                                    PrevLogRT + 
                                                    (1|ItemID) + # random intercept for item
                                                    (1|SubjectID), # random intercept for subject
                                                  data=.SD, REML=F))),
                                 .(mID, delrate, Expt)]

# Generate plot

ggsave("Fig3_v21_filler_AIC.pdf",
       ggplot(rt.filler.fits[Expt == "Vani_etal_2021-E1"]) + 
         aes(paste0((1-as.numeric(delrate)) * 100, "%"), AIC) +
         geom_smooth(aes(group=1)) +
         geom_point() +
         theme_minimal() +
         labs(x="Retention rate")
       , h=3, w=4)
