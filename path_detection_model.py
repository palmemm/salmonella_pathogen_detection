from pathogen_detection_data import df_encoded
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, accuracy_score

y = df_encoded['serovars_le']
X = df_encoded.drop(columns=['serovars_le'])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25)

models = [['Decision Tree', DecisionTreeClassifier(random_state =12, max_leaf_nodes = max(y))],
          ['AdaBoost', AdaBoostClassifier(n_estimators = 100, learnnig_rate = 1.2, random_state = 12)],
          ['Random Forest', RandomForestClassifier(n_estimators=100, random_state = 12)],
          ['SVC', SVC(probability = True, random_state = 12)],
          ['KNN', KNeighborsClassifier(weights= 'distance')],
          ['Logistic Regression', LogisticRegression(multi_class = 'multinomial', max_iter=1000, random_state =12)]
]

results = []
for model_name, model in models:
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_score = model.predict_proba(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    roc_auc_s = roc_auc_score(y_test, y_score)
    results.append([model_name, 'accuracy:',accuracy, 'ROC AUC score:', roc_auc_s])

print(results)


