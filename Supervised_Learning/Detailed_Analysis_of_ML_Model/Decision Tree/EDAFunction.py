import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def percent_missing(df):
    percent_nan = 100*df.isnull().sum()/len(df)
    percent_nan = percent_nan[percent_nan>0].sort_values()
    return percent_nan

from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error, mean_squared_error
def report_regression_model(model):
    model.fit(X_train, y_train)
    model_pred = model.predict(X_test)
    print(f'the mean absolute percentage error of the {model} is {mean_absolute_percentage_error(model_pred, y_test)}')
    print(f'the mean absolute error of the {model} is {mean_absolute_error(model_pred, y_test)}')
    print(f'the mean squared error of the {model} is { np.sqrt(mean_squared_error(model_pred, y_test))}')


from sklearn.metrics import roc_curve, roc_auc_score
from sklearn.metrics import ConfusionMatrixDisplay, classification_report, confusion_matrix, RocCurveDisplay
import matplotlib.pyplot as plt

def report_classification_model(model,X_train,y_train,X_test,y_test):
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    print("Classification Report:")
    print(classification_report(y_test, y_pred))
    
    print("Confusion Matrix:")
    ConfusionMatrixDisplay(confusion_matrix(y_test, y_pred)).plot()
    plt.show()
    
    # Calculate the probabilities for each class
    if hasattr(model, "predict_proba"):
        y_prob = model.predict_proba(X_test)
        if len(model.classes_) == 2:
            # For binary classification, consider only the positive class probability
            y_prob = y_prob[:, 1]
        else:
            # For multi-class classification, use one-vs-rest strategy and calculate ROC for each class
            y_prob = y_prob[np.arange(len(y_prob)), y_test]
    else:
        raise AttributeError("Model does not have a 'predict_proba' method.")
    
    # Calculate the ROC curve
    fpr, tpr, thresholds = roc_curve(y_test, y_prob)
    
    # Calculate the AUC (Area Under the Curve)
    auc_score = roc_auc_score(y_test, y_prob)
    
    # Plot the ROC curve
    plt.figure()
    RocCurveDisplay(fpr=fpr, tpr=tpr, roc_auc=auc_score, estimator_name=type(model).__name__).plot()
    plt.show()


def categorical_plot(cat_columns,df):
    size = len(cat_columns)
    fig, axes = plt.subplots(nrows=(size+1)//2, ncols=2, figsize=(12,(size+1)//2*5))
    for i,col in enumerate(cat_columns):
       row_index, col_index = divmod(i,2)
       sns.countplot(x=col,data=df, ax=axes[row_index][col_index])
       axes[row_index][col_index].set_title(col , fontsize=10)

    if size%2 == 1:
      fig.delaxes(axes[-1,-1])