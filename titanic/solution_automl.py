import pandas as pd

import titanic_utils

X_train = pd.read_csv('data/train.csv')
X_test = pd.read_csv('data/test.csv')
y_train = X_train['Survived']
X_train.drop('Survived', axis=1, inplace=True)

import autosklearn.classification as autosklearn_cls

automl = autosklearn_cls.AutoSklearnClassifier(time_left_for_this_task=600, per_run_time_limit=120)
automl.fit(X_train, y_train)
print(automl.show_models())

titanic_utils.evaluate_model_accuracy(automl, X_train, y_train, X_test)
