#read dataset 
neural <- read.csv(file.choose())
neural <- neural[0:310,3:12]
summary(neural)
#data preprocessing
neural$Sex<-as.integer(as.factor(neural$Sex))
neural$MaritalDesc<-as.integer(as.factor(neural$MaritalDesc))
neural$PerformanceScore<-as.integer(as.factor(neural$PerformanceScore))

#custom normalization function
normalize<-function(x){
  return ( (x-min(x))/(max(x)-min(x)))
}
neural_norm<-as.data.frame(lapply(neural,normalize))
summary(neural_norm)

#divide the data into train and test
neural_train<-neural_norm[1:220,]
neural_test<-neural_norm[221:310,]

# Using multilayered feed forward nueral network
# package nueralnet
install.packages("neuralnet") #regresion
install.packages("nnet") #classifiction
library(neuralnet)
library(nnet)

#building model
neural_model <- neuralnet(PerformanceScore~.,data = neural_train)
#visualize the network topology
plot(neural_model)
# SSE sum of squared errors . least SSE best model
# Evaluating model performance
# compute function to generate ouput for the model prepared
model_results <- compute(neural_model,neural_test[1:9])
model_results

#obtain predicte output values
predicted_performance <- model_results$net.result
#accuracy
cor(predicted_performance,neural_test$PerformanceScore) #accuracy= 0.99
plot(predicted_performance,neural_test$PerformanceScore)


