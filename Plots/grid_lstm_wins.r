library(ggplot2)
library(reshape2)
library(plyr)
setwd("/Users/annaleontjeva/Desktop/My_files/Generative-Models-in-Classification/")
dt <- read.table("Results/grid_lstm_wins.csv", header=TRUE, sep=',')
dt_melted <- melt(dt, id.vars=c("nsamples","nfeatures", "nseqfeatures", "seqlen"), variable.name='models', value.name = "accuracy")
dt_melted$dynamic_ratio <- dt_melted$nseqfeatures/(dt_melted$nfeatures + dt_melted$nseqfeatures)
dt_melted$nsamples <- as.factor(dt_melted$nsamples)
dt_melted$nseqfeatures <- as.factor(dt_melted$nseqfeatures)
dt_melted$nfeatures <- as.factor(dt_melted$nfeatures)
dt_melted$seqlen <- as.factor(dt_melted$seqlen)



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
  theme_bw(base_size=32) + geom_errorbar(limits, position='dodge')

dt_summary <- calc_confidence_interval(data=dt_melted, feature_of_interest="seqlen")
limits <- aes(ymax = mean_acc + 1.96*se, ymin=mean_acc - 1.96*se) #95% C.I.
p2 <- ggplot(dt_summary, aes(x=models, y=mean_acc, fill=seqlen)) + geom_bar(position='dodge', stat='identity', color="grey90") + 
  theme_bw(base_size=32) + geom_errorbar(limits, position='dodge') + 
  ylab("mean accuracy") + theme(axis.text.x = element_text(angle = 40, hjust = 1)) + labs(fill="sequence \nlength")

dt_summary <- calc_confidence_interval(data=dt_melted, feature_of_interest="nsamples")
limits <- aes(ymax = mean_acc + 1.96*se, ymin=mean_acc - 1.96*se) #95% C.I.
p3 <- ggplot(dt_summary, aes(x=models, y=mean_acc, fill=nsamples)) + geom_bar(position='dodge', stat='identity', color="grey90") + 
  theme_bw(base_size=32) + geom_errorbar(limits, position='dodge')

dt_summary <- calc_confidence_interval(data=dt_melted, feature_of_interest="nfeatures")
limits <- aes(ymax = mean_acc + 1.96*se, ymin=mean_acc - 1.96*se) #95% C.I.
p4 <- ggplot(dt_summary, aes(x=models, y=mean_acc, fill=nfeatures)) + geom_bar(position='dodge', stat='identity', color="grey90") + 
  theme_bw(base_size=32) + geom_errorbar(limits, position='dodge')

dt_summary <- calc_confidence_interval(data=dt_melted, feature_of_interest="nseqfeatures")
limits <- aes(ymax = mean_acc + 1.96*se, ymin=mean_acc - 1.96*se) #95% C.I.
p5 <- ggplot(dt_summary, aes(x=models, y=mean_acc, fill=nseqfeatures)) + geom_bar(position='dodge', stat='identity', color="grey90") + 
  theme_bw(base_size=32) + geom_errorbar(limits, position='dodge')

#dynamic ratio
p6 <- ggplot(dt_melted, aes(x=dynamic_ratio, y=accuracy))+ 
geom_point() + theme_bw(base_size=32) + facet_wrap(~models)
p6

png("/Users/annaleontjeva/Desktop/My_files/Generative-Models-in-Classification/Results/mean_acc_vs_seqlen.png", width = 1600, height = 1000)
print(p2)
dev.off()

png("/Users/annaleontjeva/Desktop/My_files/Generative-Models-in-Classification/Results/mean_acc_vs_nsamples.png", width = 1600, height = 1000)
print(p3)
dev.off()

png("/Users/annaleontjeva/Desktop/My_files/Generative-Models-in-Classification/Results/mean_acc_vs_nfeatures.png", width = 1600, height = 1000)
print(p4)
dev.off()

png("/Users/annaleontjeva/Desktop/My_files/Generative-Models-in-Classification/Results/mean_acc_vs_nseqfeatures.png", width = 1600, height = 1000)
print(p5)
dev.off()


setEPS()
postscript("/Users/annaleontjeva/Desktop/My_files/Generative-Models-in-Classification/Results/mean_acc_vs_seqlen.eps")
print(p2)
dev.off()

# general plot of accuracy vs. models
p7 <- ggplot(dt_melted, aes(x=models, y=accuracy)) + geom_boxplot() + theme_bw(base_size=24) + coord_flip() + 
  geom_point(position = position_jitter(width = 0.2)) +
  geom_boxplot(outlier.colour = NA, fill = NA) + facet_wrap(~nsamples)
multiplot(p2,p3,p4,p5)
p1
p2
p3
p4
p5
p6

ggplot(dt_melted, aes(x=feat_ratio, y=accuracy, size=nsamples, color=nsamples)) + 
  geom_point() + theme_bw(base_size=24) + geom_jitter(position = position_jitter()) + facet_wrap(~models)

ggplot(dt_melted, aes(x=feat_ratio)) + geom_histogram()

dt_sbs <- subset(dt_melted, nsamples==3000 & nfeatures==30 & nseqfeatures==1 & seqlen==50)

ggplot(dt_sbs, aes(x=models, y=accuracy)) + geom_bar(position="dodge", stat='identity')+theme_bw(base_size=24)
