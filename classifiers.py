from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, f1_score, classification_report
import xgboost as xgb


class BinaryClassifier:
    def __init__(self, model_type="rf"):
        if model_type == "rf":
            self.model = RandomForestClassifier(
                n_estimators=100, max_depth=20, random_state=42, n_jobs=-1
            )
            self.name = "Random Forest"
        elif model_type == "xgb":
            self.model = xgb.XGBClassifier(
                n_estimators=100,
                max_depth=10,
                learning_rate=0.1,
                random_state=42,
                eval_metric="logloss",
            )
            self.name = "XGBoost"
        elif model_type == "nb":
            self.model = GaussianNB()
            self.name = "Naive Bayes"

    def train(self, X_train, y_train):
        self.model.fit(X_train, y_train)

    def predict(self, X_test):
        return self.model.predict(X_test)

    def evaluate(self, X_test, y_test, target_names=["BENIGN", "ATTACK"]):
        y_pred = self.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, average="weighted")
        report = classification_report(y_test, y_pred, target_names=target_names)

        return {
            "accuracy": accuracy,
            "f1_score": f1,
            "predictions": y_pred,
            "report": report,
        }


class MulticlassClassifier:
    def __init__(self, model_type="rf"):
        if model_type == "rf":
            self.model = RandomForestClassifier(
                n_estimators=100, max_depth=20, random_state=42, n_jobs=-1
            )
            self.name = "Random Forest"
        elif model_type == "xgb":
            self.model = xgb.XGBClassifier(
                n_estimators=100,
                max_depth=10,
                learning_rate=0.1,
                random_state=42,
                eval_metric="mlogloss",
            )
            self.name = "XGBoost"
        elif model_type == "nb":
            self.model = GaussianNB()
            self.name = "Naive Bayes"

    def train(self, X_train, y_train):
        self.model.fit(X_train, y_train)

    def predict(self, X_test):
        return self.model.predict(X_test)

    def evaluate(self, X_test, y_test, target_names):
        y_pred = self.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, average="weighted")
        report = classification_report(y_test, y_pred, target_names=target_names)

        return {
            "accuracy": accuracy,
            "f1_score": f1,
            "predictions": y_pred,
            "report": report,
        }

    def get_feature_importances(self):
        if hasattr(self.model, "feature_importances_"):
            return self.model.feature_importances_
        return None
