library(ggplot2)
library(reshape2)
library(plyr)


dt <- read.table("../../Results/grid_lstm_wins.csv", header=TRUE, sep=',')
dt_melted <- melt(dt, id.vars=c("nsamples","nfeatures", "nseqfeatures", "seqlen"), variable.name='models', value.name = "accuracy")
dt_melted$nsamples <- as.factor(dt_melted$nsamples)

calc_confidence_interval <- function(data=dt_melted, feature_of_interest){
  command <- paste("dt_summary <- ddply(data, .(", feature_of_interest, ", models), summarize, mean_acc=round(mean(accuracy, na.rm = T),2), sd_acc=round(sd(accuracy, na.rm = T),2))", sep='')
  eval(parse(text=command))
  eval(parse(text=paste("dt_tot <- ddply(data, .(", feature_of_interest,", models), nrow)", sep='')))
  colnames(dt_tot)[3] <- "count"
  dt_summary <- merge(dt_summary, dt_tot, by=c(feature_of_interest, "models"), all.x=T)
  dt_summary$se <- dt_summary$sd_acc/sqrt(dt_summary$count)
  return(dt_summary)
}

dt_summary <- calc_confidence_interval(data=dt_melted, feature_of_interest="seqlen")
limits <- aes(ymax = mean_acc + 1.96*se, ymin=mean_acc - 1.96*se) #95% C.I.


p1 <- ggplot(dt_summary, aes(x=seqlen, y=mean_acc, fill=models)) + geom_bar(position='dodge', stat='identity', color="grey90") + 
  theme_bw(base_size=24) + geom_errorbar(limits, position='dodge')
p2 <- ggplot(dt_summary, aes(x=models, y=mean_acc, fill=as.factor(seqlen))) + geom_bar(position='dodge', stat='identity', color="grey90") + 
  theme_bw(base_size=24) + geom_errorbar(limits, position='dodge')

print(p2)

ggplot(dt_summary, aes(x=models, y=mean_acc, fill=nsamples)) + geom_bar(position='dodge', stat='identity', color="grey90") + 
  theme_bw(base_size=24) + geom_errorbar(limits, position='dodge')


# general plot of accuracy vs. models
ggplot(dt_melted, aes(x=models, y=accuracy)) + geom_boxplot() + theme_bw(base_size=24) + coord_flip() + 
  geom_point(position = position_jitter(width = 0.2)) +
  geom_boxplot(outlier.colour = NA, fill = NA) + facet_wrap(~nsamples)

