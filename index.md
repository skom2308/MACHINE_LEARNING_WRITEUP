# Practical Machine Learning: Course Project
by José Carrasquero
Johns Hopkins University Data Science Specialization

##Introduction
In this Course project we will use data from accelerometers on the belt, forearm, arm, and dumbell of 6 participants performing one arm dumbell curls in five different ways (correct form, and four different variations with incorrect form):

* A: Correct form (Exactly according to the specification)
* B: Throwing the elbows to the front
* C: Lifting the dumbbell only halfway
* D: Lowering the dumbbell only halfway
* E: Throwing the hips to the front

and train machine learning algorithms able to predict which variation of the exerise is being performed.

##R Packages used:
We will use the following packages:


```r
library(caret)
library(rpart)
library(randomForest)
```

## The Data we use:
We proceed to download the training and testing data sets:

```r
if(!file.exists("pml-training.csv")){
        download.file("https://d396qusza40orc.cloudfront.net/predmachlearn/pml-training.csv", 
                      "/Users/skom/Desktop/coursera/MACHINE_LEARNING_WRITEUP/pml-training.csv",
                      method="curl")
}

if(!file.exists("pml-testing.csv")){
        download.file("https://d396qusza40orc.cloudfront.net/predmachlearn/pml-testing.csv", 
                      "/Users/skom/Desktop/coursera/MACHINE_LEARNING_WRITEUP/pml-testing.csv",
                      method="curl")
}
```

And we proceed to read the csv files into r, in this case we will need to specify that there are 3 different types of NA strings in the files:

```r
df  <- read.csv("pml-training.csv", na.strings = c(NA,"","#DIV/0!"))
df2 <- read.csv("pml-testing.csv" , na.strings = c(NA,"","#DIV/0!"))
```

Then I proceed to eliminate the first seven columns from the training dataset as they contain irrelevant information for the prediction exercise and all columns that have more than 90% of NA values (than we eliminate those same columns from the testing set). Next, the training data set is split into a smaller training and testing data set (60/40 split):

```r
df[1:7]<- list(NULL)
df <- df[,colSums(is.na(df))<nrow(df)*0.9]
set.seed(333)
inTrain <- createDataPartition(y=df$classe,
                               p=0.6, list=FALSE)
training <- df[inTrain,]
testing <- df[-inTrain,]

df2 <- read.csv("pml-testing.csv", na.strings = c(NA,"","#DIV/0!"))
df2<-df2[,names(df2) %in% names(df)]
df2$classe <- 1
```

## Regression tree model
Using the rpart function (from the package of the same name), we proceed to create a regression tree classification model. We see how the prediction model works out against the smaller training and testing datasets (I will only show results for testing sets):

```r
set.seed(333)
model1 <- rpart(classe ~. , data=training)

pred_test1 <-predict(model1,testing, type="class")
testing$predRight <- pred_test1==testing$classe
#table(pred_test1,testing$classe)
confusionMatrix(pred_test1,testing$classe)
```

```
## Confusion Matrix and Statistics
## 
##           Reference
## Prediction    A    B    C    D    E
##          A 2010  333   22  149   46
##          B   50  811   66   31   94
##          C   63  145 1119  213  163
##          D   77  124   82  799   81
##          E   32  105   79   94 1058
## 
## Overall Statistics
##                                          
##                Accuracy : 0.7388         
##                  95% CI : (0.729, 0.7485)
##     No Information Rate : 0.2845         
##     P-Value [Acc > NIR] : < 2.2e-16      
##                                          
##                   Kappa : 0.6682         
##  Mcnemar's Test P-Value : < 2.2e-16      
## 
## Statistics by Class:
## 
##                      Class: A Class: B Class: C Class: D Class: E
## Sensitivity            0.9005   0.5343   0.8180   0.6213   0.7337
## Specificity            0.9020   0.9619   0.9098   0.9445   0.9516
## Pos Pred Value         0.7852   0.7709   0.6571   0.6870   0.7734
## Neg Pred Value         0.9580   0.8959   0.9595   0.9271   0.9407
## Prevalence             0.2845   0.1935   0.1744   0.1639   0.1838
## Detection Rate         0.2562   0.1034   0.1426   0.1018   0.1348
## Detection Prevalence   0.3263   0.1341   0.2171   0.1482   0.1744
## Balanced Accuracy      0.9013   0.7481   0.8639   0.7829   0.8426
```

```r
pred1 <- predict(model1,df2, type="class")
pred1
```

```
##  1  2  3  4  5  6  7  8  9 10 11 12 13 14 15 16 17 18 19 20 
##  B  A  E  D  A  C  D  A  A  A  C  E  C  A  E  E  A  A  A  B 
## Levels: A B C D E
```

The regression tree model has an overall accuracy of 73.88% on the testing set (which was subset from the training dataset). The out of sample error is around 26.12%. Looking at the different sensitivities and specificities it is able to predict certain forms of performing the exercise  better than others.

## Random Forest Model
Than using the randomForest function (from the package of the same name), we proceed to create and random forest classification model:

```r
set.seed(333)
model2 <- randomForest(classe ~. , data=training)
model2
```

```
## 
## Call:
##  randomForest(formula = classe ~ ., data = training) 
##                Type of random forest: classification
##                      Number of trees: 500
## No. of variables tried at each split: 7
## 
##         OOB estimate of  error rate: 0.8%
## Confusion matrix:
##      A    B    C    D    E  class.error
## A 3346    1    1    0    0 0.0005973716
## B   18 2250   11    0    0 0.0127248793
## C    0   16 2034    4    0 0.0097370983
## D    0    0   31 1897    2 0.0170984456
## E    0    0    1    9 2155 0.0046189376
```

```r
pred_test2 <-predict(model2,testing); testing$predRight <- pred_test2==testing$classe
#table(pred_test2,testing$classe)
confusionMatrix(pred_test2,testing$classe)
```

```
## Confusion Matrix and Statistics
## 
##           Reference
## Prediction    A    B    C    D    E
##          A 2231    8    0    0    0
##          B    1 1507   14    0    0
##          C    0    3 1353   15    0
##          D    0    0    1 1271    6
##          E    0    0    0    0 1436
## 
## Overall Statistics
##                                           
##                Accuracy : 0.9939          
##                  95% CI : (0.9919, 0.9955)
##     No Information Rate : 0.2845          
##     P-Value [Acc > NIR] : < 2.2e-16       
##                                           
##                   Kappa : 0.9923          
##  Mcnemar's Test P-Value : NA              
## 
## Statistics by Class:
## 
##                      Class: A Class: B Class: C Class: D Class: E
## Sensitivity            0.9996   0.9928   0.9890   0.9883   0.9958
## Specificity            0.9986   0.9976   0.9972   0.9989   1.0000
## Pos Pred Value         0.9964   0.9901   0.9869   0.9945   1.0000
## Neg Pred Value         0.9998   0.9983   0.9977   0.9977   0.9991
## Prevalence             0.2845   0.1935   0.1744   0.1639   0.1838
## Detection Rate         0.2843   0.1921   0.1724   0.1620   0.1830
## Detection Prevalence   0.2854   0.1940   0.1747   0.1629   0.1830
## Balanced Accuracy      0.9991   0.9952   0.9931   0.9936   0.9979
```

```r
pred2 <- predict(model2,df2)
pred2
```

```
##  1  2  3  4  5  6  7  8  9 10 11 12 13 14 15 16 17 18 19 20 
##  B  A  B  A  A  E  D  B  A  A  B  C  B  A  E  E  A  B  B  B 
## Levels: A B C D E
```

This model has a 99.2% accuracy on the training subset and 99.39% accuracy on the testing subset.The out of sample error is around 0.61%.

## Conclusion
For this dataset the Random Forrest model predicts much better in which form the dumbell curl exercise is being performed.
