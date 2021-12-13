library(tidyverse)
library(caret)
library(data.table)
library(kableExtra)
library(ggplot2)
library(dplyr)
library(lubridate)


#Introduction
#Data Science: Capstone, is the final course in the HarvardX Professional Certificate in Data Science
#For this project, I will  create a movie recommendation system using the MovieLens dataset.  
#I will  create my own recommendation system using all the tools that have been shown  throughout the courses in this series. I will use the 10M version of the MovieLens dataset to make the computation a little easier.
#I will download the MovieLens data and run code  provided to generate my datasets.
# I will train a machine learning algorithm using the inputs in one subset to predict movie ratings in the validation set. 

#Loading Data from MovieLens Dataset zip file,

##########################################################
# Create edx set, validation set (final hold-out test set)
##########################################################

# Note: this process could take a couple of minutes

if(!require(tidyverse)) install.packages("tidyverse", repos = "http://cran.us.r-project.org")
if(!require(caret)) install.packages("caret", repos = "http://cran.us.r-project.org")
if(!require(data.table)) install.packages("data.table", repos = "http://cran.us.r-project.org")

library(tidyverse)
library(caret)
library(data.table)

# MovieLens 10M dataset:
# https://grouplens.org/datasets/movielens/10m/
# http://files.grouplens.org/datasets/movielens/ml-10m.zip

dl <- tempfile()
download.file("http://files.grouplens.org/datasets/movielens/ml-10m.zip", dl)

ratings <- fread(text = gsub("::", "\t", readLines(unzip(dl, "ml-10M100K/ratings.dat"))),
                 col.names = c("userId", "movieId", "rating", "timestamp"))

movies <- str_split_fixed(readLines(unzip(dl, "ml-10M100K/movies.dat")), "\\::", 3)
colnames(movies) <- c("movieId", "title", "genres")


# if using R 4.0 or later:
movies <- as.data.frame(movies) %>% mutate(movieId = as.numeric(movieId),
                                           title = as.character(title),
                                           genres = as.character(genres))


movielens <- left_join(ratings, movies, by = "movieId")

# Validation set will be 10% of MovieLens data
set.seed(1, sample.kind="Rounding") # if using R 3.5 or earlier, use `set.seed(1)`
test_index <- createDataPartition(y = movielens$rating, times = 1, p = 0.1, list = FALSE)
edx <- movielens[-test_index,]
temp <- movielens[test_index,]

# Make sure userId and movieId in validation set are also in edx set
validation <- temp %>% 
  semi_join(edx, by = "movieId") %>%
  semi_join(edx, by = "userId")

# Add rows removed from validation set back into edx set
removed <- anti_join(temp, validation)
edx <- rbind(edx, removed)

rm(dl, ratings, movies, test_index, temp, movielens, removed)

#Data exploration
summary(edx)

# histogram of ratings

ggplot(edx, aes(x= edx$rating)) +   geom_histogram( binwidth = 0.1) +
  labs(x="rating", y="number of ratings") +  ggtitle("Histogram")

# top 10 movies 
top_10 <- edx %>%  group_by(title) %>%  summarize(count=n()) %>%
  top_n(10,count) %>%  arrange(desc(count))
print(top_10)
ggplot(top_10, aes(x=reorder(title,count),y=count))+geom_bar(stat='identity', fill="cyan", width=0.2)+ coord_flip()+ylab("Count")+xlab("Title")

#The number of unique values for movieId and userId
unique=data.frame(movies=length(unique(edx$movieId)),users=length(unique(edx$userId)))
unique
tunique=as.data.frame( t(unique))
ggplot(tunique, aes(x=row.names(tunique),y=tunique[,1]))+geom_bar(stat='identity', fill="red", width=.5)+xlab("Var")+ylab("Unique")


# histogram  movie Versus ratings
edx %>%  count(movieId) %>%   ggplot(aes(n)) + geom_histogram(bins=20,color="Green")+ scale_x_log10() +xlab("Mov ID")+ylab("# of rates") 
 
# histogram  User Versus ratings
edx %>%  count(userId) %>%   ggplot(aes(n)) + geom_histogram(bins=20,color="Green")+ scale_x_log10() +xlab("User ID")+ylab("# of rates")                                                                 

#some movies get rated more than others, and some users are more active than others


#  the relation between rating versus  year
time=round_date(as_datetime(edx$timestamp), unit = "year")
timedat=data.frame(time,edx$rating)
timedat %>%  group_by(time) %>% summarize(edx.rating = mean(edx.rating)) %>% ggplot(aes(time, edx.rating)) +   geom_smooth( colour="red", size=0.5) 

#When compared to current times, the rating was high in the past.


# the RMSE function  -library(caret)  -> RMSE()

#Methods
#1.simple methods
#The mean of all movies 
meo=mean(edx$rating)
mean_only_RMSE=RMSE(validation$rating,meo)
mean_only_RMSE

# movie effect 
e_mov_mean = edx %>%  group_by(movieId) %>%  summarize(e_mov_avg = mean(rating-meo))
pred_rat_eachmov = validation %>% left_join(e_mov_mean, by='movieId') %>% mutate(pr1 = meo + e_mov_avg)
mean_each_movie_RMSE = RMSE(validation$rating,pred_rat_eachmov$pr1)
mean_each_movie_RMSE

# user+movie effect 
e_user_mean = edx %>% left_join(e_mov_mean, by = "movieId")  %>% group_by(userId) %>%  summarize(e_user_avg = mean(rating-meo-e_mov_avg))
pred_rat_eachusr = validation %>% left_join(e_mov_mean, by='movieId') %>%left_join(e_user_mean, by='userId') %>% mutate(pr2 = meo + e_mov_avg + e_user_avg)
mean_each_user_RMSE = RMSE(validation$rating,pred_rat_eachusr$pr2)
mean_each_user_RMSE

#Regularization
Lamda <- seq(1, 10, 1)
RMS_Regularization <- sapply(Lamda, function(Lam){
  meo=mean(edx$rating)
  e_mov_avg <- edx %>%  group_by(movieId) %>% summarize(e_mov_avg = sum(rating - meo)/(n()+Lam))
  e_user_mean <- edx %>% left_join(e_mov_avg, by="movieId") %>% group_by(userId) %>% summarize(e_user_mean = sum(rating - e_mov_avg - meo)/(n()+Lam))
  pred_rat <- validation %>%left_join(e_mov_avg, by = "movieId") %>%left_join(e_user_mean, by = "userId") %>%mutate(pr = meo + e_mov_avg + e_user_mean)
  
  return(RMSE(validation$rating, pred_rat$pr))
})
plot(Lamda, RMS_Regularization) 
min(RMS_Regularization)




#sampling

set.seed(1000, sample.kind="Rounding")
miniedx=edx[sample(9000055,10000),]

#Split the mini edx
indx = createDataPartition(miniedx$rating, times=1, p=0.8,list=FALSE)
train = miniedx[indx,]
test = miniedx[-indx,]

#Machine Learning Models
#Linear Model 
lm = train(rating~userId+movieId+timestamp, data=train, method="lm")
lm
pred1 = predict(lm, test)
lm_RMSE = RMSE(test$rating, pred1 )
lm_RMSE

#Decision Tree 
TRE = train(rating~userId+movieId+timestamp, data=train, method="rpart")
(TRE)
pred2 = predict(TRE, test)
TRE_RMSE = RMSE(test$rating, pred2)
TRE_RMSE


#k-Nearest Neighbors
knn = train(rating~userId+movieId+timestamp, data=train, method="knn")
knn
plot(knn)
pred3 = predict(knn, test)
knn_RMSE = RMSE(test$rating, pred3)
knn_RMSE

#Results
Rmse_Result=data.frame(method=c("mean_only","mean_each_movie","mean_each_user","Regularization","Linear Model","Decision Tree","k-Nearest Neighbors"  ),RMSE=c(mean_only_RMSE,mean_each_movie_RMSE,mean_each_user_RMSE,min(RMS_Regularization),lm_RMSE,TRE_RMSE,knn_RMSE))
arrange(Rmse_Result,(RMSE))
