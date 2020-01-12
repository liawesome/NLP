from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn import svm
from sklearn.svm import LinearSVC
#from sklearn.ensemble import GradientBoostingClassifier
# ================== PARAMETERS =======================
randomState = None

# ========== Classifiers for model building ==========
logistic_regression = LogisticRegression(penalty="l2", random_state=randomState)
decision_tree_classifier = DecisionTreeClassifier(criterion="entropy", random_state=randomState)
MultinomialNB = MultinomialNB()
svc = svm.SVC(gamma='scale')
lsvc = LinearSVC(random_state=100, tol=1e-5)

#clf = GradientBoostingClassifier(n_estimators=100, learning_rate=1.0, max_depth=1)
