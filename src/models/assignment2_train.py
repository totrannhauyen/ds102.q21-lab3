
import sys
sys.path.append('/content/drive/MyDrive/[DS102.Q21]/Lab3/Lab3')

from src.data.load_data import load_data
from src.data.preprocess import normalize
from src.models.assignment2 import SVM_SKLEARN

from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle


def train_sklearn_svm(data_path):

    X, y = load_data(data_path)

    X, y = shuffle(X, y, random_state=42)

    X = X[:500]
    y = y[:500]

    X = normalize(X)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    model = SVM_SKLEARN(C=1.0, kernel="linear")
    model.fit(X_train, y_train)

    return model, X_test, y_test
