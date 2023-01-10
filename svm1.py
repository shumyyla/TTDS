

import pandas as pd
from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import sklearn.metrics


df = pd.read_csv(r'''heart.csv''')
# df['sex'] = df['sex'].map({'M': 0, 'F': 1})
# df['address'] = df['address'].map({'U': 0, 'R': 1})

# for using it on standard dataset change blow value as predictors = df.values[:, 0:10]
predictors = df.values[:, 0:12]
targets = df.values[:,13]


pred_train, pred_test, targ_train, targ_test = train_test_split(predictors, targets, test_size=0.33)

clf =svm.SVC(kernel='rbf', C=1000, gamma=1000)
clf.fit(pred_train,targ_train)

pred = clf.predict(pred_test)

#accuracy
print("Accuracy is",accuracy_score(targ_test, pred, normalize = True))
#classification error
print("Classification error is",1- accuracy_score(targ_test, pred, normalize = True))
#sensitivity
print("sensitivity is", sklearn.metrics.recall_score(targ_test, pred, labels=None, average =  'micro', sample_weight=None))
#specificity
print("specificity is", 1 - sklearn.metrics.recall_score(targ_test, pred,labels=None, average =  'micro', sample_weight=None))
print("f1 is", sklearn.metrics.f1_score (targ_test, pred,labels=None, average =  'micro', sample_weight=None))