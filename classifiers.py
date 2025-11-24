from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, f1_score, classification_report
import xgboost as xgb


class Classifier:
    def __init__(self, model_type="rf", num_classes=2, model_params=None):
        if model_params is None:
            model_params = {}

        if model_type == "rf":
            self.model = RandomForestClassifier(**model_params)
            self.name = "Random Forest"
        elif model_type == "xgb":
            eval_metric = "logloss" if num_classes == 2 else "mlogloss"
            params = model_params.copy()
            params['eval_metric'] = eval_metric
            self.model = xgb.XGBClassifier(**params)
            self.name = "XGBoost"
        elif model_type == "nb":
            self.model = GaussianNB(**model_params)
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
