#!/usr/bin/env Rscript

# To be run on atlas

library(data.table)
library(ggplot2)
library(plyr)
library(reshape2)

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

dt  <- fread("dt  <- fread("/gpfs/hpchome/etais/hpc_anna1985/Research/Generative-Models-in-Classification/Data/corrected_dt.txt", header=FALSE)
nms <- as.character(c("accountId, sessionId, javaVersion,cpuCoreCount,maxAvailableMemory,timestamp,duration,gcCause,gcAction,gcName,PS Perm Gen used before,PS Perm Gen max before,PS Eden Space used before,PS Eden Space max before,PS Old Gen used before,PS Old Gen max before,PS Survivor Space used before,PS Survivor Space max before,PS Perm Gen used after,PS Perm Gen max after,PS Eden Space used after,PS Eden Space max after,PS Old Gen used after,PS Old Gen max after,PS Survivor Space used after,PS Survivor Space max after,allocationRate,promotionRate,maturityRate"))
nms <- strsplit(nms, split = ",")[[1]]
corrected_names <- gsub(" ", "", nms, fixed = TRUE)
setnames(dt, corrected_names)

dt$duration <- as.numeric(dt$duration)
dt$sessionId <- as.factor(dt$sessionId)
dt$timestamp <- as.numeric(dt$timestamp)
gc_overhead <- ddply(dt, .(sessionId), summarize, gc_overhead=sum(duration)/(max(timestamp)-min(timestamp)))
gc_overhead$gc_overhead <- ifelse(is.finite(gc_overhead$gc_overhead)!=T, 0, gc_overhead$gc_overhead)
gc_overhead <- subset(gc_overhead, gc_overhead<=1)

# static
#sds <- aggregate(data=dt_sample, .~sessionId, sd)
#colSums(sds[,-c(1)])
nms <- as.character(c("sessionId, javaVersion, cpuCoreCount, maxAvailableMemory, PSPermGenmaxbefore, PSOldGenmaxbefore, PSPermGenmaxafter, PSOldGenmaxafter"))
nms <- strsplit(nms, split = ", ")[[1]]

dt_static <- dt[,nms, with=F]
dt_static_agg <- dt_static[!duplicated(dt_static$sessionId), ]
dt_static <- merge(dt_static_agg, gc_overhead[,c(1,4)], by="sessionId")
dt_static$javaVersion <- ifelse(dt_static$javaVersion==1.7,1,2)
dt_static$PSPermGenmaxbefore <- ifelse(dt_static$PSPermGenmaxbefore==-1,0,dt_static$PSPermGenmaxbefore)
dt_static$PSPermGenmaxafter <- ifelse(dt_static$PSPermGenmaxafter==-1,0,dt_static$PSPermGenmaxafter)

dt$maturityRate <- NULL
dt_wide <- dynamic_dt_prep(dt, seq_length=10)
dt_wide <- merge(dt_wide, gc_overhead[,c(1,4)], by="sessionId")
dt_dynamic <- dt_wide[order(dt_wide$index, decreasing=F),]

#only those that are smaller than pre-defined length are kept

dt_static <- subset(dt_static, sessionId %in% dt_dynamic$sessionId)
dt_static_final <- dt_static[,-c(1,9),with=F]

# labels
gc_overhead$label <- ifelse(gc_overhead$gc_overhead>=0.0065,1,0)
gc_overhead$index <- c(1:nrow(gc_overhead))
gc_overhead <- subset(gc_overhead, sessionId %in% dt_dynamic$sessionId)

dt_dynamic$index <- NULL
dt_dynamic$sessionId <- NULL

# writing results
write.table(as.data.frame(gc_overhead$label), "/gpfs/hpchome/etais/hpc_anna1985/Research/Generative-Models-in-Classification/Data/labels_piper.txt",
            col.names=F, row.names=F, sep=',',quote=F)

write.table(dt_static_final, "/gpfs/hpchome/etais/hpc_anna1985/Research/Generative-Models-in-Classification/Data/static_piper.txt",
            col.names=F, row.names=F, sep=',',quote=F)

write.table(dt_dynamic, "/gpfs/hpchome/etais/hpc_anna1985/Research/Generative-Models-in-Classification/Data/dynamic_piper.txt",
            col.names=F, row.names=F, sep=',',quote=F)


