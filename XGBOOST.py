import numpy as np  
import pandas as pd  
import matplotlib.pyplot as plt
import seaborn as sns  
import xgboost as xgb
import xgboost
import shap
from sklearn import metrics
from numpy import interp
from sklearn.preprocessing import label_binarize, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score, recall_score, make_scorer, precision_score, brier_score_loss,roc_curve, roc_auc_score, auc, classification_report, precision_recall_curve, f1_score
from collections import Counter
from xgboost import XGBClassifier
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



# fit model no training data 
xgb_model =  xgb.XGBClassifier(objective='binary:logistic', colsample_bytree=0.5, learning_rate=0.005, 
                              max_depth=12, n_estimators=900, scale_pos_weight=estimate_train)

kfold = StratifiedKFold(n_splits=5, random_state=None)
results = cross_validate(xgb_model, X_train, Y_train, cv=kfold,scoring=scoring)
print("Accuracy: %.2f%% (%.2f%%)" % (results['test_accuracy'].mean()*100, results['test_accuracy'].std()*100))
print("Precision: %.2f%% (%.2f%%)" % (results['test_precision'].mean()*100, results['test_precision'].std()*100))
print("Sensitivity: %.2f%% (%.2f%%)" % (results['test_sensitivity'].mean()*100, results['test_sensitivity'].std()*100))
print("Specificity: %.2f%% (%.2f%%)" % (results['test_specificity'].mean()*100, results['test_specificity'].std()*100))
print("F-1: %.2f%% (%.2f%%)" % (results['test_F-1'].mean()*100, results['test_F-1'].std()*100))
print("AUC: %.2f%% (%.2f%%)" % (results['test_auc'].mean()*100, results['test_auc'].std()*100))

########################################################################################################################
eval_set = [(X_train, Y_train), (X_test, Y_test)]
xgb_model.fit(X_train, Y_train, eval_metric=["error", "logloss"], eval_set=eval_set, verbose=True)

xgb_y_pred_test = xgb_model.predict(X_test)
xgb_y_preds_proba_test = xgb_model.predict_proba(X_test)

xgb_predictions_test = []
for value in xgb_y_preds_proba_test[:, 1]:
    if value < 0:
        value = 0
    if value > 1:
        value = 1
    else:
        value
    xgb_predictions_test.append(value)
    

# #計算auc
xgbc_auc_test = roc_auc_score(Y_test, xgb_y_preds_proba_test[:, 1])
xgb_fpr_test, xgb_tpr_test, xgb_thresholds_test = roc_curve(Y_test, xgb_y_preds_proba_test[:, 1])   
    
# performance 
print('Model: XGBOOST\n')
print(classification_report(Y_test, xgb_y_pred_test,
      target_names=['Not Ready (Class 0)', 'Ready (Class 1)'],digits=3))
print(f'Accuracy Score: {accuracy_score(Y_test,xgb_y_pred_test)}')
print(f'Confusion Matrix: \n{confusion_matrix(Y_test, xgb_y_pred_test)}')
print(f'Area Under Curve: {roc_auc_score(Y_test, xgb_predictions_test)}')
print(f'Recall score: {recall_score(Y_test,xgb_y_pred_test)}')
print(f'Brier score: {brier_score_loss(Y_test, xgb_predictions_test)}')
print("###########################################################\n")


from matplotlib import pyplot
# retrieve performance metrics
results = xgb_model.evals_result()
epochs = len(results['validation_0']['error'])
x_axis = range(0, epochs)
# plot log loss
fig, ax = pyplot.subplots()
ax.plot(x_axis, results['validation_0']['logloss'], label='Train')
ax.plot(x_axis, results['validation_1']['logloss'], label='Test')
ax.legend()
pyplot.ylabel('Log Loss')
pyplot.title('XGBoost Log Loss')
pyplot.show()
# plot classification error
fig, ax = plt.subplots()
ax.plot(x_axis, results['validation_0']['error'], label='Train')
ax.plot(x_axis, results['validation_1']['error'], label='Test')
ax.legend()
pyplot.ylabel('Classification Error')
pyplot.title('XGBoost Classification Error')
pyplot.show()


# ROC CURVE
plt.figure()
plt.figure(figsize=(12, 12))
plt.plot(xgb_fpr_test, xgb_tpr_test, 'black', label='XGBoost (AUC = %0.3f)' %Xgbc_auc_test, color='r',lw=6)
plt.plot([0, 1], [0, 1], color='black', lw=6, linestyle='--', alpha=.8)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xticks(fontsize=30)
plt.yticks(fontsize=30)
# plt.xlabel('False Positive Rate', fontsize=40)
# plt.ylabel('True Positive Rate', fontsize=40)
# plt.title('ROC CURVE (30 DAY)', fontsize=50)
plt.legend(loc="lower right",fontsize=30)
sns.set(style='white') 
plt.rcParams["font.weight"] = "bold"
# sns.despine(top=True, right= True) 
# plt.grid(False)
# plt.show() 
#plt.savefig('ROC(Testing,365).tif', format='tif', dpi=300, bbox_inches='tight')


## Decision curve

plt.figure(figsize=(16, 10))
pt_arr = []
net_bnf_arr = []
treatall = []
pred_ans = XGB_y_preds_proba_test[:, 1]


for i in range(0,100,1):
    pt = i /100
    #compiute TP FP
    pred_ans_clip = np.zeros(pred_ans.shape[0])
    for j in range(pred_ans.shape[0]):
        if pred_ans[j] >= pt:
            pred_ans_clip[j] = 1
        else:
            pred_ans_clip[j] = 0
    TP = np.sum((y_test) * np.round(pred_ans_clip))
    FP = np.sum((1 - y_test) * np.round(pred_ans_clip))
    net_bnf = ( TP-(FP * pt/(1-pt)) )/y_test.shape[0]
    print('pt {}, TP {}, FP {}, net_bf {}'.format(pt,TP,FP,net_bnf))
    pt_arr.append(pt)
    net_bnf_arr.append(net_bnf)
    treatall.append((sum(y_test)-(len(y_test)-sum(y_test))*pt/(1-pt))/len(y_test))



plt.plot(pt_arr, net_bnf_arr, color='red', lw=4,label='XGBoost')             
plt.plot(pt_arr, np.zeros(len(pt_arr)), color='k', lw=3, linestyle='--',label='Treat None')
pt_np = np.array(pt_arr)
plt.plot(pt_arr, treatall , color='black', lw=3 ,label='Treat ALL')
plt.xlim([0.0, 1.0])
plt.ylim([-0.1, 0.4])#plt.legend(loc="right")
#ax=plt.gca()
#x_major_locator=MultipleLocator(0.1)
#ax.xaxis.set_major_locator(x_major_locator)
plt.xlabel('Threshold Probability',fontsize=30)
plt.ylabel('Net Benefit',fontsize=30)
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)
plt.legend(loc="upper right", fontsize=15)
plt.grid()


## Plot calibration plots

plt.figure(figsize=(12, 12))
ax1 = plt.subplot2grid((3, 1), (0, 0), rowspan=2)
ax2 = plt.subplot2grid((3, 1), (2, 0))

colors = ['r']

clf_list = [(xgb_model, 'XGBOOST')]

ax1.plot([0, 1], [0, 1], "k:", label="Perfectly calibrated")
for i,(clf, name) in enumerate(clf_list):

    # predict probabilities
    if hasattr(clf, "predict_proba"):
        prob_pos = clf.predict_proba(X_test)[:, 1]
    else:  # use decision function
        prob_pos = clf.decision_function(X_test)
        prob_pos = \
            (prob_pos - prob_pos.min()) / (prob_pos.max() - prob_pos.min())
    fraction_of_positives, mean_predicted_value = \
        calibration_curve(y_test, prob_pos, n_bins=10)

    ax1.plot(mean_predicted_value, fraction_of_positives, "s-",
             label="%s" % (name, )+ ' (B.S = %0.3f )' % (brier_score_loss(y_test, prob_pos)), lw=5, color=colors[i])

    ax2.hist(prob_pos, range=(0, 1), bins=10, label=name,
             histtype="step", lw=5, color=colors[i])

    print('------------------------------------------------')
    print(name)
    print('accuracy = ' + str(clf.score(X_test, y_test)))
    print('brier score = ' + str(brier_score_loss(y_test, prob_pos)))

ax1.set_xlabel("Predicted probability" ,fontsize=30)
ax1.set_ylabel("True probability in each bin" ,fontsize=30)
    
#plt.xlabel("Predicted probability" ,fontsize=40, fontweight='bold')
#plt.ylabel("True probability in each bin" ,fontsize=40, fontweight='bold')
ax1.set_ylim([-0.05, 1.05])
ax1.legend(loc="upper left" ,fontsize=20)
#ax1.legend(loc="upper left" ,fontsize=27)
#ax1.set_title('Calibration plots' ,fontsize=30)

ax2.set_xlabel("Mean predicted value" ,fontsize=20)
ax2.set_ylabel("Count" ,fontsize=30)
ax2.legend(loc="upper center", ncol=2 ,fontsize=15)

ax1.tick_params(labelsize=20)
ax2.tick_params(labelsize=20)
plt.tight_layout()
sns.set(style='white') 
plt.rcParams["font.weight"] = "bold"
#plt.show()



## Partial SHAP dependence plot
fig, ((ax1, ax2, ax3), (ax4, ax5, ax6 )) = plt.subplots(nrows=2, ncols=3, figsize=(18, 10.8))
#ax1.grid(), ax2.grid(), ax3.grid(), ax4.grid(), ax5.grid(), ax6.grid()
shap.dependence_plot('GCS_24', shap_values, X_train[Features], interaction_index=None, xmin=None, xmax=None, show=False, ax=ax1)
#ax1.hlines(0,0,50,color="red")
ax1.set_ylabel('SHAP value')
ax1.set_xlabel('')
ax1.set_xticks(np.arange(2, 18, 2))

shap.dependence_plot('Urine_24', shap_values, X_train[Features], interaction_index=None, xmin=500, xmax=3500, show=False, ax=ax3)
#ax3.hlines(0,-5,15000,color="red")
ax3.set_ylabel('SHAP value')
ax3.set_xlabel('')
ax3.set_xticks(np.arange(500, 4000, 1000))

shap.dependence_plot('Injection_24', shap_values, X_train[Features], interaction_index=None, xmin=500, xmax=3500, show=False, ax=ax4)
#ax4.hlines(0,-5,15000,color="red")
ax4.set_ylabel('SHAP value')
ax4.set_xlabel('')
ax4.set_xticks(np.arange(500, 4000, 1000))

shap.dependence_plot('RASS_24', shap_values, X_train[Features], interaction_index=None, xmin=-5, xmax=5, show=False, ax=ax2)
#ax2.hlines(0,-6,60,color="red")
ax2.set_ylabel('SHAP value')
ax2.set_xlabel('')
ax2.set_xticks(np.arange(-5, 6, 2))
ax2.set_yticks(np.arange(-1.5, 1.0, 0.5))

shap.dependence_plot('PAW_24', shap_values, X_train[Features], interaction_index=None, xmin=10, xmax=40, show=False, ax=ax5)
#ax5.hlines(0,-5,145,color="red")
ax5.set_ylabel('SHAP value')
ax5.set_xlabel('')

shap.dependence_plot('MAPS_24', shap_values, X_train[Features], interaction_index=None, xmin=None, xmax=30, show=False, ax=ax6)
#ax6.hlines(0,-5,145,color="red")
ax6.set_ylabel('SHAP value')
ax6.set_xlabel('')
ax6.set_yticks(np.arange(-1.5, 1.0, 0.5))
plt.rcParams["font.weight"] = "bold"
