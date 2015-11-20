# static rf

dt <- rbind.data.frame(division[[1]],division[[2]], division[[3]])

to_exclude <- c('Start_date','End_date','time.timestamp', 'activity_name_short', 'Activity_code','group_short',
                'Number_of_executions','Producer_code','Section','Specialism_code','activity_nr')

dt_static <- dt[,!(colnames(dt) %in% to_exclude)]
length(unique(dt_static$sequence_nr))
dt_static <- unique(dt_static)

# splitting frequent_events to different factors (R cannot handle more than 33 levels on rf)
features_to_factorize <- function(data) {
  names(which(unlist(lapply(data, function(x) length(levels(x))>=25))==TRUE))
}

frequent_events_to_factors <- function(data, feature){
  feature_freq <- as.data.frame(table(data[[feature]]))
  feature_freq <- feature_freq[order(feature_freq$Freq, decreasing = T),]
  total_nr_values <- nrow(feature_freq)
  pieces <- seq(1, total_nr_values, by=25)
  if (total_nr_values > pieces[length(pieces)]){
    pieces <- c(pieces, total_nr_values)
  }
  for (j in 1:(length(pieces)-1)){
    name <- paste('freq_', feature, '_', j, sep='')
    data[[name]] <- ifelse(data[[feature]] %in% feature_freq$Var1[pieces[j]:(pieces[j+1]-1)], as.character(data[[feature]]), 'other')
    #test_data[[name]] <- ifelse(test_data[[feature]] %in% train_data[[name]], as.character(test_data[[feature]]), 'other')
    #validation_data[[name]] <- ifelse(validation_data[[feature]] %in% train_data[[name]], as.character(validation_data[[feature]]), 'other')
  }
  return(data)
}

features_factorize <- features_to_factorize(dt_static)
dt_t <- dt_static[,!(colnames(dt_static) %in% features_factorize)]
#dt_test <- dt_test_static[,!(colnames(dt_test_static) %in% features_factorize)]

for (feat in features_factorize) { #timestamp excluded here:
  print(feat)
  division_prep <- frequent_events_to_factors(data=dt_static, feature=feat)
  dt_t <- merge(dt_t, division_prep, by=c('sequence_nr','label','Age','Diagnosis_code'))
  #dt_test <- merge(dt_test, dt_basic_test, by=c('sequence_nr','label','Age','Diagnosis_code'))
}

dt_t <- dt_t[,-grep('.x',colnames(dt_t), fixed = T)]
dt_t <- dt_t[,-grep('.y',colnames(dt_t), fixed = T)]
dt_t <- dt_t[,!(colnames(dt_t) %in% features_factorize)]
dt_t <- as.data.frame(unclass(dt_t))

dt_train_idx <- sample(nrow(dt_t), 0.8*nrow(dt_t))  
dt_train <- dt_t[dt_train_idx,]
dt_test <- dt_t[-dt_train_idx,]
#subdf <- lapply(merged_res, function(x) if(is.factor(x) & nlevels(x)==1) x <- NULL else x)
#merged_res <- as.data.frame(subdf[-(which(sapply(subdf,is.null),arr.ind=TRUE))])

model <- randomForest(data=dt_train[,!names(dt_basic_train)%in% 'sequence_nr'], label~., na.action=na.omit)

predictions_bin <- predict(model, dt_test[,!names(dt_test)%in% 'sequence_nr']) 
table(pred=predictions_bin, real=dt_test$label)
predictions_p <- predict(model, dt_test[,!names(dt_test)%in% 'sequence_nr'], type='prob')
roc_crv <- roc(dt_test$label, predictions_p[,'true'])
