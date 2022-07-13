import numpy as np  
import pandas as pd  
import matplotlib.pyplot as plt
import seaborn as sns  
import shap
from sklearn import metrics
from numpy import interp
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.preprocessing import label_binarize, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score, recall_score, make_scorer, precision_score, brier_score_loss,roc_curve, roc_auc_score, auc, classification_report, precision_recall_curve, f1_score
from imblearn import under_sampling, over_sampling, combine
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from imblearn.ensemble import BalancedRandomForestClassifier
from collections import Counter
from catboost import CatBoostClassifier,Pool 
from lightgbm import LGBMClassifier
from sklearn.model_selection import train_test_split, StratifiedKFold,cross_validate, KFold
from sklearn.calibration import calibration_curve, CalibratedClassifierCV


Data = pd.read_csv('D:/..../D1~D7 Dataset.csv', encoding='utf_8_sig')

train_data, test_data = train_test_split(Data, random_state=5, train_size=0.8)


Features=['GCS (Day-1)','GCS (Day-2)','RASS (Day-1)','RASS (Day-2)','Urine (Day-1)','Urine (Day-2)','Injection (Day-1)','Injection (Day-2)',
'Diet (Day-1)','Diet (Day-2)','Ppeak (Day-1)','MAPS (Day-1)','Ppeak (Day-2)','MAPS (Day-2)','Respiratory rate (Day-1)','Respiratory rate (Day-2)',
'Heart rate (Day-1)','Heart rate (Day-2)','DAY','AGE']


X = pd.DataFrame(df[Features])
y = df['Label_adjust_8']

# estimate scale_pos_weight value
counter_train = Counter(y_train)
estimate_train = counter_train[0] / counter_train[1]
estimate_train

def classification_report_with_accuracy_score(y_true, y_pred):
    print (classification_report(y_true, y_pred,digits=3)) # print classification report
    return accuracy_score(y_true, y_pred) # return accuracy score
    

# CV model
scoring = {'report':    make_scorer(classification_report_with_accuracy_score),
           'accuracy' : make_scorer(accuracy_score), 
           'precision' : make_scorer(precision_score),
           'sensitivity'  : make_scorer(recall_score),
           'specificity': make_scorer(recall_score,pos_label=0),
           'F-1': make_scorer(f1_score),
           'auc': make_scorer(roc_auc_score, needs_proba=True),
           'brier_score':make_scorer(brier_score_loss, needs_proba=True)
          }

#kfold = StratifiedKFold(n_splits=5, random_state=None)
kfold = KFold(n_splits=5, random_state=0, shuffle=True)

###########################################LOGISTIC REGRESSION######################################################################

          
LR_model = LogisticRegression(solver='liblinear', max_iter=10000, class_weight="balanced")

results = cross_validate(LR_model, X_train, Y_train, cv=kfold,scoring=scoring)

LR_model.fit(X_train, Y_train)
LR_y_pred_test = LR_model.predict(X_test)
LR_y_preds_proba_test = LR_model.predict_proba(X_test)

# # #計算auc
LR_auc_test = roc_auc_score(Y_test, LR_y_preds_proba_test[:, 1])
LR_fpr_test, LR_tpr_test, LR_thresholds_test = roc_curve(Y_test, LR_y_preds_proba_test[:, 1])


# performance
print('Model: Logistic Regression\n')
print(classification_report(Y_test, LR_y_pred_test,
      target_names=['Survive (Class 0)', 'Death (Class 1)'],digits=3))
print(f'Accuracy Score: {accuracy_score(Y_test,LR_y_pred_test)}')
print(f'Confusion Matrix: \n{confusion_matrix(Y_test, LR_y_pred_test)}')
print(f'Area Under Curve: {roc_auc_score(Y_test, LR_y_preds_proba_test[:, 1])}')
print(f'Recall score: {recall_score(Y_test,LR_y_pred_test)}')
print(f'Brier score: {brier_score_loss(Y_test, LR_y_preds_proba_test[:, 1])}')
print("###########################################################\n")


###########################################RANDOM FOREST######################################################################

BRF_model =BalancedRandomForestClassifier(n_estimators=3000, criterion='entropy', verbose=1)
results = cross_validate(BRF_model, X_train, y_train, cv=kfold,scoring=scoring)
print("Accuracy: %.2f%% (%.2f%%)" % (results['test_accuracy'].mean()*100, results['test_accuracy'].std()*100))
print("Precision: %.2f%% (%.2f%%)" % (results['test_precision'].mean()*100, results['test_precision'].std()*100))
print("Sensitivity: %.2f%% (%.2f%%)" % (results['test_sensitivity'].mean()*100, results['test_sensitivity'].std()*100))
print("Specificity: %.2f%% (%.2f%%)" % (results['test_specificity'].mean()*100, results['test_specificity'].std()*100))
print("AUC: %.2f%% (%.2f%%)" % (results['test_auc'].mean()*100, results['test_auc'].std()*100))
BRF_model.fit(X_train, y_train)
BRF_y_pred_test = BRF_model.predict(X_test)
BRF_y_preds_proba_test = BRF_model.predict_proba(X_test)

# #計算auc
BRF_auc_test = roc_auc_score(y_test, BRF_y_preds_proba_test[:, 1])
BRF_fpr_test, BRF_tpr_test, BRF_thresholds_test = roc_curve(y_test, BRF_y_preds_proba_test[:, 1])



# performance
print('Model: Balanced Random Forest\n')
print(classification_report(y_test, BRF_y_pred_test, target_names=['Not Ready (Class 0)', 'Ready (Class 1)'],digits=3))
print(f'Accuracy Score: {accuracy_score(y_test,BRF_y_pred_test)}')
print(f'Confusion Matrix: \n{confusion_matrix(y_test, BRF_y_pred_test)}')
print(f'Area Under Curve: {roc_auc_score(y_test, BRF_y_preds_proba_test[:,1])}')
print(f'Recall score: {recall_score(y_test,BRF_y_pred_test)}')
print(f'Brier score: {brier_score_loss(y_test, BRF_y_preds_proba_test[:,1])}')
print("###########################################################\n")


###########################################CATBOOST######################################################################
