
from sklearn.svm import SVC

class SVM_SKLEARN:
    def __init__(self, C=1.0, kernel="linear"):
        self.model = SVC(C=C, kernel=kernel)

    def fit(self, X, y):
        self.model.fit(X, y)

    def predict(self, X):
        return self.model.predict(X)
