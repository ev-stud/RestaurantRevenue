library(tidymodels)
library(vroom)
library(embed) # encoding
library(bonsai) # boost tree
library(lubridate) # date functionality



# Import Data -------------------------------------------------------------
train <- vroom("./RestaurantRevenue/train.csv", delim = ",")
test <- vroom("./RestaurantRevenue/test.csv", delim = ",")


# EDA ---------------------------------------------------------------------
head(train)
head(test)

glimpse(train)
DataExplorer::plot_correlation(train)
# categories:
# P14-P18, P19-P23, P24-P27, P30-P37
# are they factors? 
# create a month factor


# Recipe -------------------------------------------------------------------
my_recipe <- recipe(revenue~., data = train) %>% 
  step_mutate(Date = mdy(`Open Date`),
              Week = week(Date),
              Year = year(Date)) %>%
  step_rm(`Open Date`, Type, City, Date) %>% # test has additional Type, City Group covers City
  step_lencode_mixed(all_nominal_predictors(), outcome=vars(revenue))
  
# view the dataset to see the reduction of dimensions
prepped <- prep(my_recipe)
baked <- bake(prepped, train)
glimpse(baked)


# Boosted Trees -----------------------------------------------------------
boost_model <- boost_tree(mode = "regression",
                          engine = "lightgbm",
                          trees = 2500,
                          tree_depth = tune(),
                          learn_rate = tune())

boost_wf <- workflow() %>%
  add_recipe(my_recipe) %>%
  add_model(boost_model)

#try tune_bayes
tuning_grid <- grid_regular(tree_depth(),
                            learn_rate(),
                            levels = 5) # grid of L^2 tuning possibilities

folds <- vfold_cv(train, v = 5, repeats =1) # K-folds

cv_results <- boost_wf %>%
  tune_grid(resamples=folds,
            grid=tuning_grid,
            metrics=NULL)

bestTune <- cv_results %>%
  select_best(metric = "rmse")

final_wf <- boost_wf %>%
  finalize_workflow(bestTune) %>% 
  fit(data=train)

submit <- final_wf %>% 
  predict(new_data= test) %>%
  bind_cols(test) %>%
  rename(Prediction=.pred) %>% # pred_1 is prediction on response = 1, pred_0 for response=0
  select(Id,Prediction)

vroom_write(submit,"./RestaurantRevenue/submission.csv", delim = ",") # best


# BART --------------------------------------------------------------------

bart_model <- bart(trees = 500) %>%
  set_engine("dbarts") %>%
  set_mode("regression") 

bart_wf <- workflow() %>% 
  add_model(bart_model) %>%
  add_recipe(my_recipe) %>%
  fit(data=train)

submit <-bart_wf %>% 
  predict(new_data= test) %>%
  bind_cols(test) %>%
  rename(Prediction=.pred) %>% # pred_1 is prediction on response = 1, pred_0 for response=0
  select(Id,Prediction)

vroom_write(submit,"./RestaurantRevenue/submission.csv", delim = ",")


# KNN ---------------------------------------------------------------------
knn_model <- nearest_neighbor(neighbors=tune(), # round(sqrt(length(train)))
                              dist_power=tune()) %>% 
  set_mode("regression") %>%
  set_engine("kknn")

knn_wf <- workflow() %>%
  add_recipe(my_recipe) %>%
  add_model(knn_model)

tuning_grid <- grid_regular(neighbors(),
                            dist_power(),
                            levels = 5) # grid of L^2 tuning possibilities

folds <- vfold_cv(train, v = 5, repeats =1) # K-folds

cv_results <- knn_wf %>%
  tune_grid(resamples=folds,
            grid=tuning_grid,
            metrics=NULL)

bestTune <- cv_results %>%
  select_best(metric = "rmse")

final_knn <- knn_wf %>%
  finalize_workflow(bestTune) %>%
  fit(data=train)

submit <- final_knn %>% 
  predict(new_data= test) %>%
  bind_cols(test) %>%
  rename(Prediction=.pred) %>% 
  select(Id,Prediction)

vroom_write(submit,"./RestaurantRevenue/submission.csv", delim = ",")


