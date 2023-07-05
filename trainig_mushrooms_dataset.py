from sklearn import *
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

training_mush = pd.read_csv("C:\\Users\\Пользователь\\Downloads\\training_mush.csv") #Train Data
testing_mush = pd.read_csv("C:\\Users\\Пользователь\\Downloads\\testing_mush.csv") #Test data


y_train = training_mush['class'] #Target Variable
x_train = training_mush.iloc[:, :-1] #Features


# print(x_train.head())
# print(y_train.head())

parameters = {'n_estimators': range(10, 50, 10), 'max_depth': range(1, 12, 2),
              'min_samples_leaf': range(1, 7),
              'min_samples_split': range(2, 9, 2)} #Dictioanry of Parameters

clf_rf = ensemble.RandomForestClassifier(random_state=0)
grid_search_cv_clf_rf = model_selection.GridSearchCV(clf_rf, parameters, cv = 3)

grid_search_cv_clf_rf.fit(x_train, y_train)  #Fit of RF

best_params = grid_search_cv_clf_rf.best_params_ #Analyzed best parameters
#
# print(best_params)
model = grid_search_cv_clf_rf.best_estimator_ #Analyzed best estimator


predicted_count = model.predict(testing_mush) #predicts
# print(predicted_count)

feature_importance = model.feature_importances_ #Key variables of the model
feature_importance_df = pd.DataFrame({'features': list(x_train),
                        'features_importance': feature_importance})


feature_importance_df = feature_importance_df.sort_values('features_importance', ascending=True)
# print(feature_importance_df)

#scores of model
accuracy_score = metrics.accuracy_score(y_train, model.predict(x_train))
precision_score = metrics.precision_score(y_train, model.predict(x_train))
recall_score = metrics.recall_score(y_train, model.predict(x_train))

#There was not y_test in testing_mush database

print(accuracy_score)
print(precision_score)
print(recall_score)

#Definetly we got all metrics equals to 1.0 as a proving of model's efficiency





