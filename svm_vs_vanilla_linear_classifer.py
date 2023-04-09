import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets, svm, metrics, model_selection
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix
import pandas as pd
import seaborn as sns
from sklearn.metrics import accuracy_score

digits = datasets.load_digits()
target = digits.target
flatten_digits = digits.images.reshape((len(digits.images), -1))

# _, axes = plt.subplots(nrows=1, ncols=5, figsize=(10, 4))
# for ax, image, label in zip(axes, digits.images, target):
#     ax.set_axis_off()  # turns off the axis lines and labels for better visualization.
#     ax.imshow(image, cmap=plt.cm.gray_r, interpolation='nearest')
#     ax.set_title("%i" % label)
# plt.show()

X_train, X_test, y_train, y_test = train_test_split(flatten_digits, target, test_size=0.2)

# Standardize the dataset
scaler = StandardScaler()
X_train_logistic = scaler.fit_transform(X_train)
X_test_logistic = scaler.transform(X_test)

# Create the logistic regression and fit the logistic regression and use the l1 penalty.
# Note here that since this is a multiclass problem the Logistic Regression parameter multi_class is set to multinomial.
# C = regularization strength: a technique used to prevent overfitting by adding a penalty term to the cost function.
# l1 = Lazzo Regularization
# Saga =  a solver that supports both L1 and L2 penalties and is particularly well-suited for large datasets.
# tol = Tolerance, which specifies the stopping criterion for the optimization algorithm.
# multinomial = model will use a softmax activation, this allows the model to predict more than two classes
logit = LogisticRegression(C=0.01, penalty='l1', solver='saga', tol=0.1, multi_class='multinomial')
logit.fit(X_train_logistic, y_train)
y_pred_logistic = logit.predict(X_test_logistic)


# print("Accuracy: " + str(logit.score(X_test_logistic, y_test)))


def show_heat_map(y_true, y_predicted, ax):
    label_names = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    cmx = confusion_matrix(y_true, y_predicted, labels=label_names)
    df_cm = pd.DataFrame(cmx)
    sns.set(font_scale=1.4)
    sns.heatmap(df_cm, annot=True, annot_kws={"size": 16}, ax=ax)


# show_heat_map(y_test, y_pred_logistic, title="Confusion Matrix for Logistic Regression results")

svm_classifier = svm.SVC(gamma='scale')
svm_classifier.fit(X_train, y_train)
y_pred_svm = svm_classifier.predict(X_test)

# print("Accuracy: " + str(svm_classifier.score(X_test, y_test)))
# print("Accuracy: "+str(accuracy_score(y_test, y_pred_svm)))
# show_heat_map(y_test, y_pred_svm, title="Confusion Matrix for SVM results")

_, axes = plt.subplots(ncols=2, figsize=(12, 6))
model_predictions = {"Confusion Matrix for Multi LRegression results": y_pred_logistic,
                     "Confusion Matrix for SVM results": y_pred_svm}
for ax, predicted in zip(axes, model_predictions.items()):
    title, y_pred = predicted
    show_heat_map(y_test, y_pred, ax=ax)
    ax.set_title(title)
plt.show()

algorithm = []
algorithm.append(('SVM', svm_classifier))
algorithm.append(('Logistic_L1', logit))
algorithm.append(
    ('Logistic_L2', LogisticRegression(C=0.01, penalty='l2', solver='saga', tol=0.1, multi_class='multinomial')))

results = []
names = []
y = digits.target
for name, algo in algorithm:
    k_fold = model_selection.KFold(n_splits=10)
    if name == 'SVM':
        X = flatten_digits
        cv_results = model_selection.cross_val_score(algo, X, y, cv=k_fold, scoring='accuracy')
    else:
        scaler = StandardScaler()
        X = scaler.fit_transform(flatten_digits)
        cv_results = model_selection.cross_val_score(algo, X, y, cv=k_fold, scoring='accuracy')

    results.append(cv_results)
    names.append(name)

fig = plt.figure()
fig.suptitle('Compare Logistic and SVM results')
ax = fig.add_subplot()
plt.boxplot(results)
plt.ylabel('Accuracy')
ax.set_xticklabels(names)
plt.show()

if __name__ == '__main__':
    print()
