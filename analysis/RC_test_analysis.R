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

#### Predictions for Vani et al (V21) and Staub (S10): Fig 3

# Extract V21 predictions & label critical regions

v21_CR = list(subj_rc = c("_", "_", "that", "verb", "det", "noun", rep("_", 10)),
              obj_rc = c("_", "_", "that", "det", "noun", "verb", rep("_", 10)),
              subj_rcNN = c("_", "_", "that", "verb", "det", "noun", "noun", rep("_", 10)),
              obj_rcNN = c("_", "_", "that", "det", "noun", "noun", "verb", rep("_", 10)))

CR_lk = function(cond, reg, CR=v21_CR) {
  labels = CR[[cond]]
  return(labels[reg])
}

v21.lcs = lcs_rc_preds %>%
  .[stri_startswith_fixed(SentenceID, "Vani_etal_2021")] %>%
  .[ , c("sID", "expt") := tstrsplit(Expt, "-")] %>%
  .[ , ItemID := as.numeric(ItemID)] %>%
  .[expt == "E1" & Condition != "filler"] %>%
  .[ , CR := mapply(CR_lk, Condition, Region+1)] %>%
  # Region+1 for indexing purposes
  .[ItemID %in% c(4, 21), 
    # ^ these items have multi-word noun regions
    CR := mapply(CR_lk, paste0(Condition, "NN"), Region+1)] %>%
  .[ , .(SurprisalReweighted = mean(SurprisalReweighted)),
         # so we average LCS over repetitions
     .(SentenceID, Region, Word, 
       fname, language, delrate, model_id, run_id,
       Expt, ItemID, Condition, CR)] %>%
  .[ , .(SurprisalReweighted = sum(SurprisalReweighted),
         # then sum LCS over CRs
         Word = paste(unique(Word), collapse=" "),
         Region = paste(unique(Region), collapse=" ")),
     .(SentenceID, 
       fname, language, delrate, model_id, run_id,
       Expt, ItemID, Condition, CR)]



# Extract S10 predictions & label critical regions

s10_CR = list(obj_rc = c("RC_comp", "det", "noun", "verb", "main verb"),
              obj_rc18 = c("RC_comp", "det", "noun", "noun", "verb", "main verb"),
              obj_rc21 = c("RC_comp", "det", "noun", "verb", "verb", "main verb"),
           subj_rc = c("RC_comp", "verb", "det", "noun", "main verb"),
           subj_rc18 = c("RC_comp", "verb", "det", "noun", "noun", "main verb"),
           subj_rc21 = c("RC_comp", "verb", "verb", "det", "noun", "main verb"))

s10.lcs = lcs_rc_preds %>%
  .[stri_startswith_fixed(SentenceID, "Staub_2010")] %>%
  .[ , ItemID := as.numeric(ItemID)] %>%
  .[ , idx := Region + 1 - min(Region), .(ItemID, Condition)] %>%
  .[ , CR := mapply(CR_lk, Condition, idx, MoreArgs=list(CR=s10_CR))] %>%
  .[ItemID %in% c(18, 21), # items with multi-word CRs
    CR := mapply(CR_lk, paste0(Condition, ItemID), idx, MoreArgs=list(CR=s10_CR))]  %>%
  .[ , .(SurprisalReweighted = mean(SurprisalReweighted)), 
     # as above: average over repetitions
     .(Expt, ItemID, Condition, Word, Region, idx, CR, 
       delrate, model_id, run_id)] %>%
  .[ , .(SurprisalReweighted = sum(SurprisalReweighted),
         # then sum over multi-word CRs
         Word = paste(unique(Word), collapse = " "),
         Region = paste(unique(Region), collapse=" ")),
     .(Expt, ItemID, Condition, CR, delrate, model_id, run_id)]

# Isolate selected deletion rates (cf. paper + filler_analysis.R for motivation)

DRs = c(M=0.4, E=0.8) # Maze (M) vs. Eye-tracking (E)

v21.s10.lcs = s10.lcs %>%
  rbind(v21.lcs[ , names(s10.lcs), with=FALSE]) %>%
  .[, Condition := c(ORC="ORC", SRC="SRC", 
                     obj_rc="ORC", subj_rc="SRC", embed="embed")[Condition]] %>%
  .[ , task_match := paste((delrate == DRs[["E"]] & stri_startswith_fixed(Expt, "Staub")) | 
                             (delrate == DRs[["M"]] & !stri_startswith_fixed(Expt, "Staub")))]

# reading times for comparison plot
v21.s10.rt = data.table(sID = rep(c("Staub (2010)", "Vani et al. (2021)"), each=6),
                        Condition = rep(c("ORC", "SRC"), 6),
                        CR = rep(c("det", "noun", "verb"), 2, each=2),
                        RT = c(249, 239, 266, 316, 318, 270, # read from V et al Fig 4
                               794, 711, 894, 889, 958, 950)) 
                              # ^ calculated from V21 RT, cf. v21_test_means.R

ggsave("Fig4_v21_s10_CR.pdf",
       ggplot(v21.s10.lcs[Expt != "Vani_etal_2021-E2_S10" & 
                            delrate %in% DRs &
                            CR %in% c("det", "noun", "verb")]) +
         aes(CR, SurprisalReweighted, fill=Condition, alpha=paste(task_match)) +
         scale_alpha_manual(values = c("TRUE"=0.9, "FALSE"=0.4)) +
         stat_summary(geom="bar", position=position_dodge(width=0.9)) +
         stat_summary(geom="errorbar", position=position_dodge(width=0.9), width=0.2) +
         facet_grid(factor(ifelse(stri_startswith_fixed(Expt, "Staub_2010"), "Staub (2010)", "Vani et al. (2021)"), 
                           levels = c("Vani et al. (2021)", "Staub (2010)")) ~ 
                      paste0(100*(1-as.numeric(delrate)), "%")) +
         scale_fill_brewer(type="qual", palette="Dark2") +
         theme_minimal() +
         labs(x=NULL, y="Lossy Context Surprisal") +
         guides(alpha="none", fill="none") +
         ggplot(v21.s10.rt) +
         aes(CR, RT, fill=Condition) +
         geom_col(position="dodge") +
         scale_fill_brewer(type="qual", palette="Dark2") +
         facet_grid(factor(sID, levels = c("Vani et al. (2021)", "Staub (2010)")) ~ ., scales="free") +
         theme_minimal() +
         labs(x=NULL, y="Reading Time") +
         plot_layout(widths=c(2,1))
       , w=9, h=4)

#### Roland et al. (2021): Fig. 5

r21_CR = list("ORC" = c("Main Clause Subject", "Main Clause Subject", 
                             "That", "RC NP", "RC NP", "RC Verb", 
                             "Main Clause Verb", rep("_", 10)),
              "SRC" = c("Main Clause Subject", "Main Clause Subject", 
                             "That", "RC Verb", "RC NP", "RC NP", 
                             "Main Clause Verb", rep("_", 10)),
              "ORCMCS" = c("Main Clause Subject", "Main Clause Subject", 
                                "Main Clause Subject", "That", "RC NP", "RC NP", 
                                "RC Verb", "Main Clause Verb", rep("_", 10)),
              "SRCMCS" = c("Main Clause Subject", "Main Clause Subject", 
                                "Main Clause Subject", "That", "RC Verb", 
                                "RC NP", "RC NP", "Main Clause Verb", rep("_", 10)))

r21.lcs = lcs_rc_preds %>%
  .[ Expt == "RolandEtAl2021" &
      stri_endswith_fixed(Condition, "-noun")] %>%
  .[stri_endswith_fixed(SentenceID, "src-noun"), # fix silly coding error: 
    Condition := "ORC"] %>% # reversed ORC and SRC in prediction item labels
  .[stri_endswith_fixed(SentenceID, "orc-noun"),
    Condition := "SRC"] %>% 
  .[delrate %in% DRs] %>% # filter to deletion rates of interest
  .[ , CR := mapply(CR_lk, Condition, Region+1, MoreArgs=list(CR=r21_CR))] %>%
  .[ ItemID %in% c(1, 2, 14, 16), 
     CR := mapply(CR_lk, paste0(Condition, "MCS"), Region+1, 
                  MoreArgs=list(CR=r21_CR))] %>%
  .[ , .(SurprisalReweighted = mean(SurprisalReweighted)), # average over reps
     .(delrate, model_id, run_id, Expt, ItemID=as.numeric(ItemID), 
       Condition, Region, Word, CR)]  %>%
  .[ , .(SurprisalReweighted = sum(SurprisalReweighted), # then sum over CRs
         Word = paste(unique(Word), collapse = " "),
         Region = paste(unique(Region), collapse=" ")),
     .(Expt, ItemID, Condition, CR, delrate, model_id, run_id)]

# Original RT and spillover-adjusted RT for comparison
# From Roland et al. (2021), Experiment 2, full NP condition
# For RT data and analysis code see https://osf.io/4rq3m/

# Gaze duration CR averages from Experiment 2
# Following data cleaning &c from authors' code
# NB Fig. 5 in the paper has RT error bars calculated over full data -
# using computed averages here for expedience


r21.et = data.table(Condition = rep(c("ORC", "SRC"), each=3),
                    CR = rep(c("RC NP", "RC Verb", "Main Clause Verb"), 2),
                    prevCR = c("That", "RC NP", "RC Verb", 
                               "RC Verb", "That", "RC NP"),
                    RT = c(350, 309, 313, 374, 251, 294))

# Spillover adjustment
# Beta coefficients from Roland et al. (2021) Table 12

spillover_betas = data.table(prevCR = c("RC NP", 
                                        "RC Verb",
                                        "That"),
                             beta = c(28.59, 25.33,
                                      0)) # "that" = intercept

r21.et.adj = r21.et %>%
  .[spillover_betas, on=.(prevCR)] %>%
  .[ , RT.adj := RT - beta]

ggsave("Fig5_r21.pdf",
       ggplot(r21.et.adj) +
         aes(factor(CR, levels = c("RC NP", "RC Verb", "Main Clause Verb")), 
             RT, fill=Condition) +
         stat_identity(geom="bar", position=position_dodge(width=0.9), alpha=.5) +
         scale_fill_brewer(type="qual", palette="Dark2") +
         guides(fill="none") +
         theme_minimal() +
         labs(x="", y="RT") +
         theme(axis.text.x = element_text(angle = 25, hjust=.75)) +
         ggplot(r21.et.adj) +
         aes(factor(CR, levels = c("RC NP", "RC Verb", "Main Clause Verb")), 
             RT.adj, fill=Condition) +
         stat_identity(geom="bar", position=position_dodge(width=0.9)) +
         scale_fill_brewer(type="qual", palette="Dark2") +
         guides(fill="none") +
         theme_minimal() +
         labs(x="", y="Spillover-adjusted RT") +
         theme(axis.text.x = element_text(angle = 25, hjust=.75)) +
         ggplot(r21.lcs[CR %in% c("RC NP", "RC Verb", "Main Clause Verb")]) +
         aes(factor(CR, levels = c("RC NP", "RC Verb", "Main Clause Verb")), 
             SurprisalReweighted, fill=Condition, alpha=delrate) +
         stat_summary(geom="bar", position=position_dodge(width=0.9)) +
         stat_summary(geom="errorbar", position=position_dodge(width=0.9), width=0.2) +
         scale_fill_brewer(type="qual", palette="Dark2") +
         scale_alpha_manual(values=c("0.8"=1, "0.4"=.5)) +
         guides(alpha="none") +
         facet_grid(. ~ paste0((1-as.numeric(delrate))*100,"%"),
                    scales="free") +
         labs(x="", y="Lossy Context Surprisal", fill="Condition") +
         theme_minimal() +
         theme(axis.text.x = element_text(angle = 25, hjust=.75)) +
         plot_layout(widths=c(1,1,2))
       , w=9, h=3.5)

