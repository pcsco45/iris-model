from ucimlrepo import fetch_ucirepo
from numpy import ravel
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

class IrisModel:
    def __init__(self):
        iris = fetch_ucirepo(id=53)
        X = iris.data.features
        y = ravel(iris.data.targets)

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=50
        )


        self.model = make_pipeline(StandardScaler(), LogisticRegression())
        self.model.fit(X_train, y_train)


        self.train_acc = self.model.score(X_train, y_train)
        self.test_acc = self.model.score(X_test, y_test)

    def predict(self, features):
        return self.model.predict([features])[0]
