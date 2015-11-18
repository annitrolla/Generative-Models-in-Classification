#load environment
library(reshape2)
library(plyr)
options("scipen" = 10)

# functions
seq_order_fun  <- function(x) {
  seq_along(x)
}
to_numeric <- function(data, feature){
  data[[feature]] <- as.factor(data[[feature]])
  data[[feature]]<- mapvalues(data[[feature]], from=levels(data[[feature]]), 
                              to = 1:length(levels(data[[feature]])))
  return(data)
} 

dynamic_dt_prep <- function(data, seq_length, nm=nms){
  length_dst <- as.data.frame(table(data$sessionId))
  sequences_long <- subset(length_dst, Freq>=seq_length)$Var1
  dt_long <- subset(data, sessionId %in% sequences_long)
  dt_longg <- dt_long[,-c('accountId',nm[-1],'gcName'),with=F]
  dt_long <- ddply(dt_longg, .(sessionId), mutate, seq_order = seq_order_fun(sessionId))
  dt_long_frst <- subset(dt_long, seq_order<=seq_length)
  dt_long_frst <- as.data.table(dt_long_frst)
  for(i in c("gcCause", "gcAction")){
    dt_long_frst <- to_numeric(dt_long_frst, i)
  }
  suppressWarnings(for (i in seq_along(dt_long_frst)) set(dt_long_frst, i=which(dt_long_frst[[i]]==-1), j=i, value=0))
  dt_wide <- dcast.data.table(dt_long_frst, sessionId ~ seq_order, value.var=colnames(dt_long_frst)[-c(1,2,21)], fill=NaN)
  return(dt_wide)
}
#---------------------------------------------------------------------------------#

# gcoverhead calculation
dt$duration <- as.numeric(dt$duration)
dt$sessionId <- as.factor(dt$sessionId)
dt$timestamp <- as.numeric(dt$timestamp)
gc_overhead <- ddply(dt, .(sessionId), summarize, gc_overhead=sum(duration)/(max(timestamp)-min(timestamp)))
gc_overhead$gc_overhead <- ifelse(is.finite(gc_overhead$gc_overhead)!=T, 0, gc_overhead$gc_overhead)
gc_overhead <- subset(gc_overhead, gc_overhead<=1)

#  0     1 
# 70597  9299 

# labels
gc_overhead$label <- ifelse(gc_overhead$gc_overhead>=0.0065,1,0)
gc_overhead$index <- c(1:nrow(gc_overhead))
table(gc_overhead$label)

# static
sds <- aggregate(data=dt_sample, .~sessionId, sd)
colSums(sds[,-c(1)])
nms <- as.character(c("sessionId, javaVersion, cpuCoreCount, maxAvailableMemory, PSPermGenmaxbefore, PSOldGenmaxbefore, PSPermGenmaxafter, PSOldGenmaxafter"))
nms <- strsplit(nms, split = ", ")[[1]]

dt_static <- dt[,nms, with=F]
dt_static_agg <- dt_static[!duplicated(dt_static$sessionId), ]
dt_static <- merge(dt_static_agg, gc_overhead[,c(1,4)], by="sessionId")
dt_static$javaVersion <- ifelse(dt_static$javaVersion==1.7,1,2)
dt_static$PSPermGenmaxbefore <- ifelse(dt_static$PSPermGenmaxbefore==-1,0,dt_static$PSPermGenmaxbefore)
dt_static$PSPermGenmaxafter <- ifelse(dt_static$PSPermGenmaxafter==-1,0,dt_static$PSPermGenmaxafter)
dt_static_final <- dt_static[,-c(1,9),with=F]

# ToDO:
# labels based on gc_overhead for each session
# for each session data: all numeric?
# static and dynamic separate
# how many events to take?
# each session - one line, list all features for one timestamp, then for another timestamp, etc
dt_wide <- dynamic_dt_prep(dt, seq_length=10)
dt_wide <- merge(dt_wide, gc_overhead[,c(1,4)], by="sessionId")
dt_dynamic <- dt_wide[order(dt_wide$index, decreasing=F),]

# writing results
# check consistency
dim(as.data.frame(gc_overhead$label))
dim(dt_static_final)
dim(dt_dynamic)

write.table(as.data.frame(gc_overhead$label), "/Users/annaleontjeva/Desktop/My_files/Generative-Models-in-Classification/Data/labels_piper.txt",
            col.names=F, row.names=F, sep=',',quote=F)

write.table(dt_static_final, "/Users/annaleontjeva/Desktop/My_files/Generative-Models-in-Classification/Data/static_piper.txt",
            col.names=F, row.names=F, sep=',',quote=F)

write.table(dt_wide_filtered, "/Users/annaleontjeva/Desktop/My_files/Generative-Models-in-Classification/Data/dynamic_piper.txt",
            col.names=F, row.names=F, sep=',',quote=F)

#------------------------#
# Plots and figures
# Gc overhead
ggplot(gc_overhead, aes(x=gc_overhead)) + geom_histogram() + theme_bw(base_size=24)
ggplot(gc_overhead, aes(x=log(gc_overhead+0.000001, base=10))) + geom_density() + theme_bw(base_size=24)

tmp <- subset(length_dst, Freq>=10)
gc_overhead_tmp <- subset(gc_overhead, sessionId %in% tmp$Var1)
table(gc_overhead_tmp$label)

ggplot(gc_overhead_tmp, aes(x=log(gc_overhead+0.000001, base=10))) + geom_density() + theme_bw(base_size=24)

# 0    1 
# 34 8451

# dynamic dataset
length_dst <- as.data.frame(table(dt$sessionId))
ggplot(length_dst, aes(x=log(Freq, base=10)))+geom_density()+theme_bw(base_size=24)
ggplot(gc_overhead_tmp, aes(x=gc_overhead)) + geom_histogram() + theme_bw(base_size=24)
