import sys, os, pandas as pd, numpy as np, matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.exceptions import NotFittedError
from sklearn.utils.validation import check_is_fitted
from sklearn.feature_selection import RFE, RFECV

parent_folder = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
sys.path.append(parent_folder)



class RandomForestFeatureSelector:
    def __init__(self, model_subtype=None, n_estimators=100, random_state=42):
        self.model_subtype = model_subtype
        self.n_estimators = n_estimators
        self.random_state = random_state
        self.model = None
        self.feature_importances_ = None
        # self.actual_model_subtype = None  # Stores resolved model_subtype
        self.train_score = None
        self.feature_count = 0

    def train(self, X, y):
        # self.actual_model_subtype = infer_model_subtype(y) if not self.model_subtype else self.model_subtype
        self.feature_count = X.shape[1]

        # if self.actual_model_subtype == 'regression':
        if self.model_subtype == 'regression':
            self.model = RandomForestRegressor(n_estimators=self.n_estimators, random_state=self.random_state)
        else:
            self.model = RandomForestClassifier(n_estimators=self.n_estimators, random_state=self.random_state)

        self.model.fit(X, y)
        self.train_score = self.model.score(X, y)

    def get_feature_importance(self, feature_names, top_n=None):
        if self.model is None:
            raise ValueError("Train the model first.")

        try:
            check_is_fitted(self.model)
        except NotFittedError:
            print("⚠️ Model not fitted. Skipping feature importance calculation.")
            return pd.Series(dtype='float64')  # Return empty series

        importances = pd.Series(self.model.feature_importances_, index=feature_names)
        self.feature_importances_ = importances.sort_values(ascending=False)

        if top_n:
            return self.feature_importances_.head(top_n)
        else:
            return self.feature_importances_.to_string()

    def plot_feature_importance(self, additional_title='', top_n=50):
        if self.feature_importances_ is None:
            raise ValueError("Train the model and extract importances before plotting.")

        if top_n:
            self.feature_importances_.head(top_n).plot(kind='barh', figsize=(10, 8))
        else:
            self.feature_importances_.plot(kind='barh', figsize=(10, 8))

        plt.title("Top Feature Importances" + additional_title)
        plt.gca().invert_yaxis()
        plt.tight_layout()
        plt.show()

    def summary(self):
        print("Model Summary")
        print("=" * 40)
        print(f"Model Subtype type:       {self.model_subtype}")
        print(f"Model used:      {type(self.model).__name__}")
        print(f"Features used:   {self.feature_count}")
        if self.model_subtype == 'regression':
            print(f"Train R² score:  {self.train_score:.4f}")
        else:
            print(f"Train accuracy:  {self.train_score:.4f}")
        print("=" * 40)


class RFEFeatureSelector:
    def __init__(self, estimator, n_features_to_select=10, step=1, cv=None):
        self.estimator = estimator
        self.n_features_to_select = n_features_to_select
        self.step = step
        self.cv = cv
        self.rfe = None
        self.support_ = None
        self.ranking_ = None

    def train(self, X, y):
        self.rfe = RFE(estimator=self.estimator, n_features_to_select=self.n_features_to_select, step=self.step)
        self.rfe.fit(X, y)
        self.support_ = self.rfe.support_
        self.ranking_ = self.rfe.ranking_

        X_selected = X.loc[:, self.support_]

    def get_feature_importance(self, feature_names, top_n=None):
        selected_features = [f for f, s in zip(feature_names, self.support_) if s]
        ranking = pd.Series(self.ranking_, index=feature_names).sort_values()
        return ranking.head(top_n)

    def summary(self):
        print("📋 RFE Summary")
        print("Selected Features:", np.array(self.rfe.support_).sum())
        print("Ranking (1 is best):", self.ranking_)


class RFECVFeatureSelector:
    def __init__(self, estimator, step=1, min_features_to_select=5, cv_splits=5, scoring='accuracy'):
        self.estimator = estimator
        self.cv = TimeSeriesSplit(n_splits=cv_splits)
        self.step = step
        self.min_features_to_select = min_features_to_select
        self.scoring = scoring
        self.rfecv = None

    def train(self, X, y):
        self.rfecv = RFECV(estimator=self.estimator, step=self.step,
                           min_features_to_select=self.min_features_to_select,
                           cv=self.cv, scoring=self.scoring)
        self.rfecv.fit(X, y)
        support = self.rfecv.support_
        self.support_ = support
        self.ranking_ = self.rfecv.ranking_
        X_selected = X.loc[:, support]

    def get_feature_importance(self, feature_names, top_n=None):
        ranking = pd.Series(self.ranking_, index=feature_names).sort_values()
        return ranking.head(top_n)

    def summary(self):
        print("📋 RFECV Summary")
        print(f"Optimal number of features: {self.rfecv.n_features_}")
        print("Ranking:", self.ranking_)
