rm(list = ls())
setwd("C:/Users/jithin/Desktop/Data Science/Project/Churn Reduction")

## Read the data
df_train = read.csv("Train_data.csv", header = T)
df_test = read.csv("Test_data.csv", header = T)

#structure od data
str(df_train)

#Load Libraries
x = c("ggplot2", "corrgram", "DMwR", "caret", "randomForest", "unbalanced", "C50", "dummies", "e1071", "Information",
      "MASS", "rpart", "gbm", "ROSE", 'sampling', 'DataCombine', 'inTrees')

lapply(x, require, character.only = TRUE)
rm(x)


#Missing value analysis
missing_train = data.frame(apply(df_train,2,function(x){sum(is.na(x))}))
missing_test = data.frame(apply(df_test,2,function(x){sum(is.na(x))}))

##Data Manupulation; convert string categories into factor numeric
#assigning levels to categorical variables
fnames = c("state","international.plan","voice.mail.plan","Churn")

for(i in fnames){
  if(class(df_train[,i])== 'factor'){
    df_train[,i] = factor(df_train[,i], labels = (1:length(levels(factor(df_train[,i])))))
  }
}

for(i in fnames){
  if(class(df_test[,i])== 'factor'){
    df_test[,i] = factor(df_test[,i], labels = (1:length(levels(factor(df_test[,i])))))
  }
}
rm(fnames)
rm(i)


numeric_index = sapply(df_train,is.numeric) #selecting only numeric
numeric_data = df_train[,numeric_index]
cnames = colnames(numeric_data)
 
## Correlation Plot 
corrgram(df_train[,numeric_index], order = F,
         upper.panel=panel.pie, text.panel=panel.txt, main = "Correlation Plot")

## Chi-squared Test of Independence
factor_index = sapply(df_train,is.factor)
factor_data = df_train[,factor_index]

for (i in 1:4)
{
  print(names(factor_data)[i])
  print(chisq.test(table(factor_data$Churn,factor_data[,i]))) 
}

## Dimension Reduction
df_train = subset(df_train, select = -c(total.day.minutes, total.eve.minutes, total.night.minutes, total.intl.minutes, phone.number))
df_test = subset(df_test, select = -c(total.day.minutes, total.eve.minutes, total.night.minutes, total.intl.minutes, phone.number))

#normality check
hist(df_train$number.customer.service.calls)


#Normalisation
cnames = c("account.length","area.code","number.vmail.messages","total.day.calls","total.day.charge",
           "total.eve.calls","total.eve.charge","total.night.calls","total.night.charge","total.intl.calls", "total.intl.charge", 
           "number.customer.service.calls")

for(i in cnames){
  print(i)
  df_train[,i] = (df_train[,i] - min(df_train[,i]))/
    (max(df_train[,i] - min(df_train[,i])))
}

for(i in cnames){
  print(i)
  df_test[,i] = (df_test[,i] - min(df_test[,i]))/
    (max(df_test[,i] - min(df_test[,i])))
}


rmExcept(c("df_train","df_test"))

train = df_train
test = df_test

##Decision tree for classification
#Develop Model on training data
C50_model = C5.0(Churn ~., train, trials = 50, rules = TRUE)

#Summary of DT model
summary(C50_model)

#write rules into disk
write(capture.output(summary(C50_model)), "c50Rules.txt")

#Lets predict for test cases
C50_Predictions = predict(C50_model, test[,-16], type = "class")

##Evaluate the performance of classification model
ConfMatrix_C50 = table(test$Churn, C50_Predictions)
confusionMatrix(ConfMatrix_C50)

#False Negative rate
FNR = FN/FN+TP 

#Accuracy: 88.9%
#FNR: 32.58%

###Random Forest
RF_model = randomForest(Churn ~ ., train, importance = TRUE, ntree = 150)

#Extract rules fromn random forest
#transform rf object to an inTrees' format
treeList = RF2List(RF_model)  

#Extract rules
exec = extractRules(treeList, train[,-16])  # R-executable conditions
 
#Visualize some rules
exec[1:2,]
 
#Make rules more readable:
readableRules = presentRules(exec, colnames(train))
readableRules[1:2,]
 
#Get rule metrics
ruleMetric = getRuleMetric(exec, train[,-16], train$Churn)  # get rule metrics
 
#evaulate few rules
ruleMetric[1:2,]

#Presdict test data using random forest model
RF_Predictions = predict(RF_model, test[,-16])

##Evaluate the performance of classification model
ConfMatrix_RF = table(test$Churn, RF_Predictions)
confusionMatrix(ConfMatrix_RF)

#False Negative rate
FNR = FN/FN+TP 

#Accuracy = 90.58
#FNR = 66.96

#Logistic Regression
logit_model = glm(Churn ~ ., data = train, family = "binomial")

#summary of the model
summary(logit_model)

#predict using logistic regression
logit_Predictions = predict(logit_model, newdata = test, type = "response")

#convert prob
logit_Predictions = ifelse(logit_Predictions > 0.5, 1, 0)


##Evaluate the performance of classification model
ConfMatrix_LR = table(test$Churn, logit_Predictions)

#False Negative rate
FNR = FN/FN+TP 

#Accuracy: 86.38
#FNR: 70.53

##KNN Implementation
library(class)

#Predict test data
KNN_Predictions = knn(train[, 1:15], test[, 1:15], train$Churn, k = 7)

#Confusion matrix
ConfMatrix_Knn = table(KNN_Predictions, test$Churn)
confusionMatrix(ConfMatrix_Knn)

#False Negative rate
FNR = FN/FN+TP 

#Accuracy = 86.62
#FNR = 48.14

#naive Bayes
library(e1071)

#Develop model
NB_model = naiveBayes(Churn ~ ., data = train)

#predict on test cases #raw
NB_Predictions = predict(NB_model, test[,1:15], type = 'class')

#Look at confusion matrix
Confmatrix_NB = table(observed = test[,16], predicted = NB_Predictions)
confusionMatrix(Confmatrix_NB)

#Accuracy: 87.16
#FNR: 53.57
