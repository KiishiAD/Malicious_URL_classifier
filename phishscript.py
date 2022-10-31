import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_selection import mutual_info_classif
from sklearn.linear_model import LogisticRegression
from cuml.ensemble import RandomForestClassifier as cuRfc
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


phishdata = pd.read_csv("Phishing_Legitimate_full.csv")

float32 = phishdata.select_dtypes('float64').columns
for c in float32:
    phishdata[c] = phishdata[c].astype('float32')

int32 = phishdata.select_dtypes('int64').columns
for c in int32:
    phishdata[c] = phishdata[c].astype('int32')

phishdata.info()



phishdata.rename(columns={'CLASS_LABEL': 'label'}, inplace=True)

phishdata['label'].value_counts().plot(kind='bar',color = 'Purple')
plt.show()


def corr_heatmap(data, idx_s, idx_e):
    y = data['label']
    temp = data.iloc[:, idx_s:idx_e]
    if 'id' in temp.columns:
        del temp['id']
    temp['label'] = y
    sns.heatmap(temp.corr(), annot=True, fmt='.2f')
    plt.show()


corr_heatmap(phishdata, 0, 10)
corr_heatmap(phishdata, 30, 40)
corr_heatmap(phishdata, 40, 50)




X = phishdata.drop(['id', 'label'], axis=1)
y = phishdata['label']
discrete = X.dtypes == int


mi = mutual_info_classif(X, y, discrete=discrete)
mi = pd.Series(mi, name='MI Scores', index=X.columns)
mi = mi.sort_values(ascending=False)
print(mi)


def miplot(scores):
    scores = scores.sort_values(ascending=True)
    width = np.arange(len(scores))
    ticks = list(scores.index)
    plt.barh(width, scores)
    plt.yticks(width, ticks)
    plt.title("MI Scores")


plt.figure(dpi=100, figsize=(12, 12))
miplot(mi)


def logisticreg(data, top_n):
    top_features = mi.sort_values(ascending=False).head(top_n).index.tolist()
    X = data[top_features]
    y = data['label']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=True)

    lr = LogisticRegression(max_iter=10000)
    lr.fit(X_train, y_train)

    y_pred = lr.predict(X_test)

    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    accuracy = accuracy_score(y_test, y_pred)

    return precision, recall, f1, accuracy

list = []
for i in range(20,51,1):
    precision, recall, f1, accuracy = logisticreg(phishdata, i)
    print("Performance for Logistic Model with Top {} features is precision : {}, recall : {}, f1 score : {}, accuracy : {}".format(i, precision, recall, f1, accuracy))
    list.append([i, precision, recall, f1, accuracy])

df = pd.DataFrame(list, columns=['num_of_features', 'precision', 'recall', 'f1_score', 'accuracy'])
print(df)

sns.lineplot(x='num_of_features', y='precision', data=df, label='Precision Score')
sns.lineplot(x='num_of_features', y='recall', data=df, label='Recall Score')
sns.lineplot(x='num_of_features', y='f1_score', data=df, label='F1 Score')
sns.lineplot(x='num_of_features', y='accuracy', data=df, label='Acc Score')


def random_forest_classifier(data, top_n):
    top_n_features = mi.sort_values(ascending=False).head(top_n).index.tolist()
    X = data[top_n_features]
    y = data['label']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=True)

    rfc = cuRfc(n_estimators=500,
                split_criterion=1,
                max_depth=32,
                max_leaves=-1,
                max_features=1.0,
                n_bins=128)

    rfc.fit(X_train, y_train)

    y_pred = rfc.predict(X_test, predict_model='CPU')

    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    accuracy = accuracy_score(y_test, y_pred)

    return precision, recall, f1, accuracy

list2 = []
for i in range(20,51,1):
    precision, recall, f1, accuracy = random_forest_classifier(phishdata, i)
    print("Performance for RFC Model with Top {} features is precision : {}, recall : {}, f1 score : {}, accuracy : {}".format(i, precision, recall, f1, accuracy))
    list2.append([i, precision, recall, f1, accuracy])

df = pd.DataFrame(list2, columns=['num_of_features', 'precision', 'recall', 'f1_score', 'accuracy'])
df.head()


sns.lineplot(x='num_of_features', y='precision', data=df, label='Precision Score')
sns.lineplot(x='num_of_features', y='recall', data=df, label='Recall Score')
sns.lineplot(x='num_of_features', y='f1_score', data=df, label='F1 Score')
sns.lineplot(x='num_of_features', y='accuracy', data=df, label='Acc Score')

top_n_features = mi.sort_values(ascending=False).head(32).index.tolist()
X = phishdata[top_n_features]
y = phishdata['label']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=True)

rfc = cuRfc(n_estimators=500,
            split_criterion=1,
            max_depth=32,
            max_leaves=-1,
            max_features=1.0,
            n_bins=128)

rfc.fit(X_train, y_train)

y_pred = rfc.predict(X_test, predict_model='CPU')

precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
accuracy = accuracy_score(y_test, y_pred)

print("Performance for RFC Model with Top {} features is precision : {}, recall : {}, f1 score : {}, accuracy : {}".format(27, precision, recall, f1, accuracy))







