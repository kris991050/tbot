import sys, os, datetime, pandas as pd, numpy as np, matplotlib.pyplot as plt, joblib, glob, xgboost, shap
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, StackingRegressor, StackingClassifier
from sklearn.metrics import (classification_report, mean_squared_error, accuracy_score, confusion_matrix,
                             recall_score, precision_score, f1_score, mean_absolute_error, r2_score,
                             ConfusionMatrixDisplay)
from sklearn.model_selection import TimeSeriesSplit, cross_val_score, train_test_split
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.dummy import DummyRegressor


parent_folder = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
sys.path.append(parent_folder)

from utils import logs, helpers
from utils.constants import PATHS
from ml import features_processor, features_selector, model_explainer
from execution import trade_manager


class ModelTrainer:
    # model types: 'xgboost', 'rf', 'linear'
    # selector types: 'rf', 'rfe', 'rfecv'
    def __init__(self, model_type:str='xgboost', selector_type:str='rf', label_column:str='label_binary', strategy_name:str='',
                 target:str='', base_models:str=None, base_folder:str=None, top_n_features:int=10, train_size:float=0.7, validation_size:float=0.15,
                 depth_surrogate:int=3, clip_outliers:bool=True, show_figures:bool=False, shap_types:list=None, drawdown:bool=False, **preprocessor_kwargs):

        self.model_type = model_type
        self.base_models = base_models or ['rf', 'xgboost', 'linear']
        self.strategy_name = strategy_name
        self.timeframe_str = trade_manager.get_strategy_instance(self.strategy_name).timeframe.pandas
        self.target = target
        self.label_column = label_column
        self.top_n_features = top_n_features
        self.train_size = train_size
        self.validation_size = validation_size
        self.depth_surrogate = depth_surrogate
        self.shap_types = shap_types or ['dot', 'bar'] # 'violin'

        self.preprocessor = features_processor.FeaturePreprocessor(label_column=label_column, clip_outliers=clip_outliers,
                                                transform_features=True, **preprocessor_kwargs)

        # self.selector = RandomForestFeatureSelector(model_subtype='auto')
        self.explainer = None  # will initialize after model is ready
        self.selector_type = selector_type
        self.model = None
        self.model_subtype = None
        self.feature_names = []
        self.trained = False
        self.show_figures = show_figures
        self.X_raw = None
        self.X_transformed = None
        self.X_train = None
        self.X_val = None
        self.X_test = None
        self.y_train = None
        self.y_val = None
        self.y_test = None
        self.shap_values = None
        self.drawdown_mode = drawdown
        self.folder_names = {'models': 'models', 'shaps': 'shaps', 'figures': 'figures'}
        self._create_folders(base_folder)
        self.create_paths()
        self.surrogate_tree = None

    def _create_folders(self, base_folder):
        self.base_folder = base_folder or os.path.join(PATHS.folders_path['strategies_data'], self.strategy_name)
        self.outputs_folder = os.path.join(self.base_folder, f'ml_outputs_{self.model_type}_{self.selector_type}')
        self.models_folder = os.path.join(self.outputs_folder, self.folder_names['models'])
        self.shaps_folder = os.path.join(self.outputs_folder, self.folder_names['shaps'])
        self.figures_folder = os.path.join(self.outputs_folder, self.folder_names['figures'])
        # self.test_set_folder = os.path.join(self.base_folder, 'test_set')
        # os.makedirs(self.models_folder, exist_ok=True)
        # os.makedirs(self.shaps_folder, exist_ok=True)
        # os.makedirs(self.figures_folder, exist_ok=True)
        # os.makedirs(self.test_set_folder, exist_ok=True)

    def create_paths(self):
        d = 'dd_' if self.drawdown_mode else ''
        self.paths = {}
        self.paths['data'] = os.path.join(self.base_folder, f"{d}results_{self.strategy_name}_{self.target}.csv")
        # rev = '' if not self.revised else '_revised'
        if self.figures_folder:
            self.paths['confusion_matrix'] = os.path.join(self.figures_folder, f"{d}confusion_matrix_{self.strategy_name}_{self.target}.png")
            self.paths['dependence_plot'] = os.path.join(self.figures_folder, f"{d}dependence_{self.strategy_name}_{self.target}.png")
            self.paths['surrogate_tree'] = os.path.join(self.figures_folder, f"{d}surrogate_tree_{self.strategy_name}_{self.target}.png")
            for shap_type in self.shap_types:
                self.paths['shap_' + shap_type] = os.path.join(self.figures_folder, f"{d}shap_{shap_type}_{self.strategy_name}_{self.target}.png")

        if self.models_folder: self.paths['models'] = os.path.join(self.models_folder,f"{d}model_{self.strategy_name}_{self.target}.pkl")
        if self.shaps_folder: self.paths['shaps'] = os.path.join(self.shaps_folder, f"{d}shap_{self.strategy_name}_{self.target}.pkl")
        # if self.test_set_folder: self.paths['test_set'] = os.path.join(self.test_set_folder, f"test_set_{self.strategy_name}_{self.target}.csv")

    def _check_model_type_subtype_compatibility(self):
        if self.model_subtype in ['classification', 'multi'] and self.model_type == 'linear':
            raise ValueError(f"'{self.model_type}' model cannot be used for classification model_subtypes.")
        if self.model_subtype == 'regression' and self.model_type in ['xgboost', 'rf']:
            pass  # XGBoost/RF support regression depending on model init
        # Could expand here if needed for stricter validation

    def _init_model(self, model_type):
        if model_type == 'xgboost':
            if self.model_subtype == 'regression':
                return xgboost.XGBRegressor()
            else: #return xgboost.XGBClassifier()
                return xgboost.XGBClassifier(tree_method='hist', enable_categorical=True, n_estimators=300, learning_rate=0.05, max_depth=4,
                                             min_child_weight=10, gamma=1.0, subsample=0.7, colsample_bytree=0.7, random_state=42, verbosity=0)

        elif model_type == 'rf':
            if self.model_subtype == 'regression':
                return RandomForestRegressor()
            else: #return RandomForestClassifier()
                return RandomForestClassifier(n_estimators=100, max_depth=6, random_state=42)

        elif model_type == 'linear':
            return LinearRegression()#(max_iter=1000)

        elif model_type == 'stacking':
            estimators = []

            for base in self.base_models:
                model = self._init_model(base)
                estimators.append((base, model))

            if self.model_subtype == 'regression':
                final_estimator = LinearRegression()
                return StackingRegressor(estimators=estimators, final_estimator=final_estimator, passthrough=True)
            else:
                final_estimator = LogisticRegression(max_iter=1000)
                return StackingClassifier(estimators=estimators, final_estimator=final_estimator, passthrough=True)

        else:
            raise ValueError(f"Unsupported model type: {model_type}")

    @staticmethod
    def data_set_split(X, y, train_size=0.7, validation_size=0.15, shuffle=False):
        try:
            if train_size + validation_size >= 1:
                raise ValueError("Train size + Validation size must be < 1")

            num_cols = X.select_dtypes(include=[np.number]).columns
            X[num_cols] = X[num_cols].replace([np.inf, -np.inf], np.nan).fillna(0)

            temp_size = (1-train_size)
            test_size = (temp_size-validation_size)/temp_size

            # First split: train vs. temp (val + test)
            X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=temp_size, shuffle=shuffle)  # shuffle = False: Important for time series!

            # Second split: val vs. test
            X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=test_size, shuffle=shuffle)
            # X_train = X_train.replace([np.inf, -np.inf], np.nan).fillna(0) # Making sure no unwanted value has slipped through the cracks

            return X_train, X_val, X_test, y_train, y_val, y_test

        except ValueError as e:
            print(f"⚠️ Skipping feature selection due to small sample size: {e}")
            # Fallback: use all data for training and testing to avoid breaking pipeline
            return X, X, X, y, y, y

    def _create_selector(self):
        estimator = self._init_model(self.model_type)

        if self.selector_type == 'rf':
            return features_selector.RandomForestFeatureSelector(model_subtype=self.model_subtype)

        elif self.selector_type in ['rfe', 'rfecv']:

            if self.model_type == 'stacking':
                # Check if estimator exposes feature importance
                supports_importance = (hasattr(estimator, 'coef_') or hasattr(estimator, 'feature_importances_'))

                if not supports_importance:
                    raise ValueError(
                        f"Selector type '{self.selector_type}' requires the model to have "
                        f"either `.coef_` or `.feature_importances_`, but model_type '{self.model_type}' "
                        f"(e.g., StackingClassifier) does not support it. "
                        f"Consider using selector_type='rf' or changing model_type.")
            else:
                if self.selector_type == 'rfe':
                    return features_selector.RFEFeatureSelector(estimator=estimator, n_features_to_select=self.top_n_features)
                elif self.selector_type == 'rfecv':
                    return features_selector.RFECVFeatureSelector(estimator=estimator, min_features_to_select=self.top_n_features, cv_splits=5)
        else:
            raise ValueError(f"Unknown selector type: {self.selector_type}")

    def _create_explainer(self):
        self.explainer = model_explainer.ModelExplainer(model=self.model, trained=self.trained, X_train=self.X_train, X_raw=self.X_raw,
                                                        X_transformed=self.X_transformed, y_train=self.y_train, feature_names=self.feature_names,
                                                        model_subtype=self.model_subtype, shap_types=self.shap_types, preprocessor=self.preprocessor,
                                                        paths=self.paths, show_figures=self.show_figures, figures_folder=self.figures_folder)

    def select_features(self, X, y):
        print(f"🔍 Selecting most relevant features using {self.selector_type} selector...")

        self.selector.train(X, y)
        importances = self.selector.get_feature_importance(X.columns, top_n=self.top_n_features)

        if importances.empty:
            print("⚠️ No important features selected due to fitting issues.")

        self.feature_names = importances.index.tolist()
        if 'market_cap_cat' not in self.feature_names: self.feature_names.append('market_cap_cat')

    def train_model(self, X_train, y_train):
        print(f"📚 Training final model using: {self.model_type} {self.model_subtype}")

        # Model and scaling should already be set in `fit()`
        if self.model is None:
            raise RuntimeError("Model has not been initialized. Call fit() or set self.model first.")

        # ✅ Check label diversity before fitting
        unique_classes = np.unique(y_train)
        if len(unique_classes) < 2:
            print(f"⚠️ Skipping training: only one class found in labels: {unique_classes}")
            return  # Skip training step

        try:
            self.model.fit(X_train[self.feature_names], y_train)
            self.trained = True
            print(f"✅ Model trained successfully on {len(X_train)} samples.")
        except Exception as e:
            print(f"❌ Error during model training: {type(e).__name__}: {e}")
            self.trained = False

    def cross_validate(self, X, y, cv=5, scoring=None):
        print(f"🔁 Running {cv}-fold cross-validation...")
        model = self._init_model(self.model_type)
        # if len(X[self.feature_names]) < cv: cv = len(X[self.feature_names]) - 1
        scores = cross_val_score(model, X[self.feature_names], y, cv=TimeSeriesSplit(n_splits=cv), scoring=scoring)
        print(f"📊 Cross-validation scores: {scores}")
        print(f"📈 Mean CV score: {scores.mean():.4f}")
        return scores

    def summarize_model_training(self, X_train, X_val):
        # print(f"\n📊 Model Summary for strategy {self.strategy_name} and timeframe {self.timeframe_str}")
        print(f"\n📊 Model Summary for strategy {self.strategy_name}, timeframe {self.timeframe_str}, target {self.target}")

        self.selector.summary()
        if self.show_figures:
            self.selector.plot_feature_importance(additional_title=f" {self.strategy_name} {self.timeframe_str} {self.target}")
        print("=" * 40)
        print(f"Model subtype type:       {self.model_subtype}")
        print(f"Model used:      {type(self.model).__name__}")
        print(f"Top features:    {len(self.feature_names)}")
        print(f"Train sample size:    {len(X_train)}")
        print(f"Validation sample size:    {len(X_val)}")
        print(f"Trained?:        {'✅ Yes' if self.trained else '❌ No'}")
        if self.trained:
            print(f"Features:        {self.feature_names}")
        print("=" * 40)

    def _build_shap_explainer(self, model, model_type, background_data=None):
        if model_type == 'xgboost':
            return shap.Explainer(model)
        elif model_type == 'rf':
            return shap.TreeExplainer(model)
        elif model_type == 'linear':
            background = background_data or shap.sample(self.X_train[self.feature_names], 100)
            return shap.Explainer(model.predict, background)
        else:
            raise ValueError(f"SHAP explainer not implemented for model type: {model_type}")

    def predict(self, df, return_proba=False, shift_features=False):
        """
        Predict using the trained model.

        Args:
            df (pd.DataFrame): Input dataframe with raw features (can be 1 row or more).
            return_proba (bool): If True, return prediction probabilities instead of class labels.
            shift_features (bool): If True, shift the feature dataframe by 1 to avoid lookahead bias.

        Returns:
            np.array: Predictions (class labels or probabilities for class 1).
        """
        if not self.trained:
            raise RuntimeError("Model must be trained before prediction.")

        df_input = df.shift(1) if shift_features else df

        # X_raw, _, _ = self.preprocessor.prepare_data_set(df_input, self.model_subtype)
        df_input = self.preprocessor.encode_categoricals(df_input)
        missing_features = [f for f in self.feature_names if f not in df.columns]
        if missing_features:
            X = self.preprocessor.prepare_features(df_input, self.model, display_features=False)
            missing_features = [f for f in self.feature_names if f not in X.columns]
            if missing_features:
                raise ValueError(f"🚫 Missing features in input data: {missing_features}")
        else:
            X = df_input

        if return_proba:
            if hasattr(self.model, "predict_proba"):
                return self.model.predict_proba(X[self.feature_names])[:, 1]  # probability of class 1
            else:
                raise ValueError("Model does not support probability predictions.")
        else:
            return self.model.predict(X[self.feature_names])

    def evaluate_model(self, X_train, y_train, X_val, y_val):
        if not self.trained:
            print("⚠️ Evaluation skipped: model is not trained.")
            return

        try:
            print("📈 Evaluation Results")
            print("🔍 Predicting on train and validation sets...")
            y_pred_train = self.model.predict(X_train[self.feature_names])
            y_pred_val = self.model.predict(X_val[self.feature_names])

            if self.model_subtype in ['classification', 'multi']:
                print(f"Train Accuracy: {accuracy_score(y_train, y_pred_train):.4f}")
                print(f"Validation Accuracy:  {accuracy_score(y_val, y_pred_val):.4f}")
                print("\nClassification Report:\n", classification_report(y_val, y_pred_val))
            else:
                print(f"Train MSE: {mean_squared_error(y_train, y_pred_train):.4f}")
                print(f"Train r2: {r2_score(y_train, y_pred_train):.4f}")
                print(f"Validation MSE:  {mean_squared_error(y_val, y_pred_val):.4f}")
                print(f"Validation r2: {r2_score(y_val, y_pred_val):.4f}")

                # Compare to Baseline
                y_baseline = DummyRegressor(strategy="mean").fit(X_train, y_train).predict(X_val)
                print(f"Baseline MSE: {mean_squared_error(y_val, y_baseline):.4f}")
                print(f"Baseline R²: {r2_score(y_val, y_baseline):.4f}")
        except Exception as e:
            print(f"❌ Error during evaluation: {type(e).__name__}: {e}")

    def get_training_results(self):
        X_val = self.X_val[self.feature_names]
        y_true = self.y_val
        y_pred_raw = self.model.predict(X_val)

        results = {
            "strategy": self.strategy_name,
            "timeframe": self.timeframe_str,
            "target": self.target,
        }

        # Automatically determine metrics based on task type
        if self.model_type == 'regression':
            results.update({
                "rmse": round(mean_squared_error(y_true, y_pred_raw, squared=False), 4),
                "mae": round(mean_absolute_error(y_true, y_pred_raw), 4),
                "r2": round(r2_score(y_true, y_pred_raw), 4),
            })

        elif self.model_type == 'classification':
            # Binary classification: threshold regression outputs if necessary
            if y_pred_raw.ndim == 1 and np.issubdtype(y_pred_raw.dtype, np.floating):
                y_pred = (y_pred_raw >= 0.5).astype(int)
            else:
                y_pred = y_pred_raw

            results.update({
                "accuracy": round(accuracy_score(y_true, y_pred), 4),
                "precision": round(precision_score(y_true, y_pred), 4),
                "recall": round(recall_score(y_true, y_pred), 4),
                "f1": round(f1_score(y_true, y_pred), 4),
            })

        elif self.model_type == 'multi':
            y_pred = y_pred_raw
            results.update({
                "accuracy": round(accuracy_score(y_true, y_pred), 4),
                "precision_macro": round(precision_score(y_true, y_pred, average='macro'), 4),
                "recall_macro": round(recall_score(y_true, y_pred, average='macro'), 4),
                "f1_macro": round(f1_score(y_true, y_pred, average='macro'), 4),
            })

        # Optionally include cross-validation score
        if hasattr(self, "cross_validate"):
            cv_score = self.cross_validate(self.X_train, self.y_train)
            results["cv_score"] = round(np.mean(cv_score), 4)

        return results

    def fit(self, df):

        # if df.shape[0] < 5:
        #     raise ValueError(f"🚫 Not enough rows in input DataFrame ({df.shape[0]}) to train a model.")

        print("🧪 Full training pipeline started...\n" + "=" * 40)

        # Reorder data per trigger time, in case not done
        df_sorted = df.sort_values("trig_time").reset_index(drop=True)

        self.X_raw, y, self.model_subtype = self.preprocessor.prepare_data_set(df_sorted, self.model_subtype)
        self._check_model_type_subtype_compatibility()
        if y.nunique() < 2:
            print(f"⚠️ Not enough label diversity for training. Label counts:\n{y.value_counts()}. Skipping.")
            return

        self.model = self._init_model(self.model_type)
        self.selector = self._create_selector()

        # Features Preparation and selection
        self.X_transformed = self.preprocessor.prepare_features(self.X_raw, self.model)
        self.X_train, self.X_val, self.X_test, self.y_train, self.y_val, self.y_test = self.data_set_split(
            self.X_transformed, y, self.train_size, self.validation_size)
        self.select_features(self.X_train, self.y_train)

        # Model training and validation
        self.train_model(self.X_train, self.y_train)
        self._create_explainer()
        self.cross_validate(self.X_train, self.y_train)
        self.summarize_model_training(self.X_train, self.X_val)
        self.plot_confusion_matrix(self.X_val, self.y_val)
        self.evaluate_model(self.X_train, self.y_train, self.X_val, self.y_val)

        # SHAP and surrogate tree analysis
        if not self.model_type == 'stacking':
            self.shap_values, X_sample = self.explainer.compute_shap()
            self.explainer.plot_shaps(self.strategy_name, self.timeframe_str, self.target, X_sample)
        self.surrogate_tree = self.explainer.train_surrogate_tree()

        print("=" * 40 + "\n✅ Training complete.\n")

        # self.X_test = self.populate_predictions_on_test(return_proba=True)
        self.save(df_sorted)

        return self.get_training_results()

    def plot_confusion_matrix(self, X, y, title="Confusion Matrix", save_plot=True):
        if self.model_subtype not in ['classification', 'multi']:
            print("⚠️ Confusion matrix only valid for classification model_subtypes.")
            return

        y_pred = self.model.predict(X[self.feature_names])
        cm = confusion_matrix(y, y_pred)

        title = title + f" startegy {self.strategy_name}, timeframe {self.timeframe_str}, target {self.target}"
        disp = ConfusionMatrixDisplay(confusion_matrix=cm)
        disp.plot(cmap='Blues')
        plt.title(title)
        plt.title(title)
        plt.tight_layout()
        plot_name = 'confusion_matrix'
        if save_plot and plot_name in self.paths:
            os.makedirs(self.figures_folder, exist_ok=True)
            # plt.savefig(os.path.join(self.figures_folder, f"confusion_matrix_{self.strategy_name}_{self.timeframe_str}_{self.target}.png"))
            plt.savefig(self.paths[plot_name])
            print(f"📊 Saved confusion matrix to: {self.paths[plot_name]}")
        if self.show_figures: plt.show()
        plt.close()

    def save(self, df):
        df_train, df_val, df_test, _, _, _ = self.data_set_split(df, df, self.train_size, self.validation_size)
        df_train = helpers.format_df_date(df_train, col='trig_time')
        df_val = helpers.format_df_date(df_val, col='trig_time')
        df_test = helpers.format_df_date(df_test, col='trig_time')

        save_models_name = self.folder_names['models']
        save_shaps_name = self.folder_names['shaps']
        # save_test_set_name = 'test_set'
        if save_models_name in self.paths:
            os.makedirs(self.models_folder, exist_ok=True)
            joblib.dump({
                'model': self.model,
                'features': self.feature_names,
                'scaler': self.preprocessor.scaler,
                'model_subtype': self.model_subtype,
                'model_type': self.model_type,
                'selector_type': self.selector_type,
                'transform_features': self.preprocessor.transform_features,
                'params': self.model.get_params() if hasattr(self.model, 'get_params') else {},
                'shap_type': 'xgboost' if self.model_type == 'xgboost' else 'tree',
                'strategy': self.strategy_name,
                'timeframe': self.timeframe_str,
                'target': self.target,
                'label_column': self.label_column,
                'data_ranges': {'training': [df_train['trig_time'].iloc[0], df_train['trig_time'].iloc[-1]],
                                'validation': [df_val['trig_time'].iloc[0], df_val['trig_time'].iloc[-1]],
                                'test': [df_test['trig_time'].iloc[0], df_test['trig_time'].iloc[-1]]}
            }, self.paths[save_models_name])
            print(f"💾 Model saved to: {self.paths[save_models_name]}")
        else:
            print(f"⚠️ model not saved as path is not defined")

        if save_shaps_name in self.paths:
            os.makedirs(self.shaps_folder, exist_ok=True)
            joblib.dump(self.shap_values, self.paths[save_shaps_name])
            print(f"💾 SHAP values saved to: {self.paths[save_shaps_name]}")
        else:
            print(f"⚠️ shaps not saved as path is not defined")

        # if save_test_set_name in self.paths:
        #     helpers.save_df_to_file(self.X_test, self.paths[save_test_set_name])#, file_format='parquet')
        #     print(f"💾 Test Set values saved to: {self.paths[save_test_set_name]}")
        # else:
        #     print(f"⚠️ Test Set not saved as path is not defined")

    def load(self):
        save_models_name = self.folder_names['models']
        if save_models_name in self.paths:
            data = joblib.load(self.paths[save_models_name])
            self.model = data['model']
            self.feature_names = data['features']
            self.preprocessor.scaler = data['scaler']
            self.model_type = data['model_type']
            self.model_subtype = data['model_subtype']
            self.trained = True
            self.model_type = data.get('model_type', self.model_type)
            self.selector = data['selector_type']
            self.strategy_name = data['strategy']
            self.timeframe_str = data['timeframe']
            self.target = data['target']
            self.label_column = data['label_column']
            self.data_ranges = data['data_ranges']
            self.preprocessor.transform_features = data.get('transform_features', True)
            self._create_explainer()
            print(f"📦 Model loaded from: {self.paths[save_models_name]}")


class StrategyModelComparator:
    def __init__(self, strategy_name:str, clip_outliers:bool=True, base_folder:str=None, model_type:str='xgboost', selector_type:str='rf', 
                 label_column:str='label_binary', depth_surrogate:int=3):
        self.strategy_name = strategy_name
        self.clip_outliers=clip_outliers
        self.trainer = ModelTrainer(model_type=model_type, selector_type=selector_type, label_column=label_column,
                               strategy_name=strategy_name, target='', clip_outliers=clip_outliers, base_folder=base_folder,
                               depth_surrogate=depth_surrogate, show_clipping_report=False, show_figures=False, plot_trim_distributions=False)
        self.trainer_dd = ModelTrainer(model_type='xgboost', selector_type=selector_type, label_column='max_drawdown_pct',
                               strategy_name=strategy_name, target='', clip_outliers=clip_outliers, base_folder=base_folder,
                               depth_surrogate=depth_surrogate, show_clipping_report=False, show_figures=False, drawdown=True,
                               plot_trim_distributions=False)

        # self.base_folder = base_folder or os.path.join(PATHS.folders_path['strategies_data'], self.strategy_name)
        self.base_folder = base_folder or self.trainer.base_folder
        # self.trainer.base_folder = self.base_folder
        # self.logs_folder = os.path.join(self.base_folder, 'logs')
        self.logs_folder = os.path.join(self.trainer.outputs_folder, 'logs')
        self.model_type = model_type
        self.selector_type = selector_type
        self.label_column = label_column
        self.depth_surrogate = depth_surrogate
        self.results = []
        self.results_dd = []
        self.comparison_file_name = f"comparison_{self.strategy_name}.csv"
        self.comparison_file_name_dd = f"comparison_dd_{self.strategy_name}.csv"
        # self.comparison_file_path = os.path.join(self.base_folder, self.comparison_file_name)
        self.comparison_file_path = os.path.join(self.trainer.outputs_folder, self.comparison_file_name)
        self.comparison_file_path_dd = os.path.join(self.trainer_dd.outputs_folder, self.comparison_file_name_dd)

    def run(self):

        # Match pattern: results_breakout_up_1D.csv or results_breakout_down_60min.csv
        pattern = os.path.join(self.base_folder, 'results_*.csv')
        files = glob.glob(pattern)
        # regex = FORMATS.RESULTS_STRATEGY_REGEX

        # === Loop through each file ===
        for file_path in files:
            filename = os.path.basename(file_path)
            # match = regex.match(filename)
            # if not match:
                # print(f"Skipping unrecognized file: {filename}")
                # continue
            
            # strategy_name = match.group('strategy')
            # timeframe = Timeframe(match.group('timeframe'))
            # target = match.group('target')

            if self.strategy_name not in filename:
                print(f"Skipping file {filename} as it doesn't match the strategy name {self.strategy_name}.")
                continue

            base_name = filename.replace('results_', '').replace('.csv', '')
            target = base_name[len(strategy_name):].strip('_')  # Remove any leading underscores, target will be everything that comes after the strategy_name part

            # trainer = ModelTrainer(model_type=self.model_type, selector_type=self.selector_type, label_column=self.label_column,
            #                         strategy=strategy_name, timeframe=timeframe, target=target, clip_outliers=self.clip_outliers,
            #                         depth_surrogate=self.depth_surrogate, show_clipping_report=False, show_figures=False, plot_trim_distributions=False)
            for t in [self.trainer, self.trainer_dd]:
                t.target = target
                t.create_paths()

            log_file_path = os.path.join(self.logs_folder, f"training_log_{strategy_name}_{self.trainer.timeframe_str}_{target}.txt")

            df = pd.read_csv(file_path)

            # Create per-model log file path
            with logs.LogContext(log_file_path, overwrite=True):  # Logging starts here

                print(f"⏰ Date: {datetime.datetime.now().strftime("%Y%m%d_%H%M%S")}\n")
                print(f"🔍 Training model for strategy: {strategy_name}, timeframe: {self.trainer.timeframe_str}, target: {target}")
                if not df.empty:
                    print(f"Tickers analyzed:\n{df['symbol'].unique().tolist()}")
                    print(f"\nDate range from  {df['data_from_time'].iloc[0]}  to  {df['data_to_time'].iloc[0]}\n")
                else: print("No data, Dataframe is empty.")

                # try:
                # Full training pipeline
                results = self.trainer.fit(df)
                results_dd = self.trainer_dd.fit(df)

                for t in [self.trainer, self.trainer_dd]:
                    if not t.model_type == 'stacking':
                        t.explainer.plot_dependence(t.shap_values.feature_names[0])
                        t.explainer.plot_dependence(t.shap_values.feature_names[1])
                        t.explainer.plot_dependence(t.shap_values.feature_names[2])
                        # t.explainer.plot_dependence("market_cap_cat")
                        if "market_cap_log" in t.shap_values.feature_names: t.explainer.plot_dependence("market_cap_cat")

                    rules = t.explainer.extract_tree_rules()
                    revised_logic = t.explainer.generate_revised_logic_code(rules, class_label="Class 1", prob_threshold=0.7)
                    print(revised_logic)

                self.results.append(results)
                self.results_dd.append(results_dd)
                # except Exception as e:
                #     print(f"⚠️ Failed to train model for {self.strategy_name} {timeframe} {target}. Error: {e}")

        self._save()

    def _save(self):
        df = pd.DataFrame(self.results)
        os.makedirs(self.trainer.outputs_folder, exist_ok=True)
        df.to_csv(self.comparison_file_path, index=False)
        print(f"✅ Saved comparison results at {self.comparison_file_path}")

        df_dd = pd.DataFrame(self.results_dd)
        os.makedirs(self.trainer_dd.outputs_folder, exist_ok=True)
        df_dd.to_csv(self.comparison_file_path_dd, index=False)
        print(f"✅ Saved comparison results drawdowns at {self.comparison_file_path_dd}")


if __name__ == "__main__":

    args = sys.argv

    pd.options.mode.chained_assignment = None # Disable Pandas warnings

    # Args Setup
    args = sys.argv
    # display_res = 'display' in args
    revised = 'revised' in args
    strategy_name = next((arg[9:] for arg in args if arg.startswith('strategy=')), '')
    selector = next((arg[9:] for arg in args if arg.startswith('selector=') and arg[9:] in ['rf', 'rfe', 'rfecv']), 'rfe')

    # Setup
    model_type='xgboost'
    # model_type='linear'
    # model_type='rf'
    # model_type='stacking'
    label_column='label_binary'
    # label_column='label_multi'
    # label_column='label_regression'
    # label_column='label_R-R'

    rev = '' if not revised else '_R'
    strategy = strategy_name + rev

    strategy_model_comparator = StrategyModelComparator(strategy_name, clip_outliers=True, model_type=model_type, selector_type=selector , label_column=label_column)
    strategy_model_comparator.run()
    # strategies_folder = PATHS.folders_path['strategies_data']
    # strategy_folder = os.path.join(strategies_folder, strategy)











# def compute_shap(self, X_sample=None, max_display=20, default_sample_size=100, random_state=42):
#         """Compute SHAP values for current trained model(s), returns dict."""
#         if not self.trained or self.model is None:
#             raise RuntimeError("Model must be trained before running SHAP.")

#         print("📊 Generating SHAP explanations...")

#         if X_sample is None:
#             X_sample = shap.utils.sample(
#                 self.X_train[self.feature_names],
#                 default_sample_size,
#                 random_state=random_state
#             )

#         shap_dict = {}

#         if self.model_type == 'stacked' and hasattr(self, 'base_models'):
#             for base_model in self.base_models:
#                 try:
#                     model_type = getattr(base_model, 'model_type', self.model_type)  # Optional: get type from model
#                     explainer = self._build_shap_explainer(base_model, model_type, background_data=X_sample)
#                     shap_values = explainer(X_sample)
#                     shap_dict[base_model] = shap_values
#                     print(f"✅ SHAP values computed for {base_model}")
#                 except Exception as e:
#                     print(f"⚠️ Failed SHAP for {base_model}: {e}")
#         else:
#             try:
#                 explainer = self._build_shap_explainer(self.model, self.model_type, background_data=X_sample)
#                 shap_values = explainer(X_sample)
#                 shap_dict[self.model_type] = shap_values
#                 print(f"✅ SHAP values computed for {self.model_type}")
#             except Exception as e:
#                 print(f"⚠️ Failed SHAP for {self.model_type}: {e}")

#         # Save to instance
#         self.shap_values = shap_dict

#         # Generate plots
#         for shap_type in self.shap_types:
#             self._plot_shap(shap_dict, X_sample, plot_type=shap_type,
#                             title_prefix=f"SHAP summary: {self.strategy} {self.timeframe} {self.target}",
#                             max_display=max_display)

#         return shap_dict

# def _plot_shap(self, shap_values_dict, X_sample, plot_type, title_prefix, max_display=20, save_plot=True):
    # for model_key, shap_values in shap_values_dict.items():
    #     title = f"{title_prefix} - {model_key}"
    #     shap.summary_plot(
    #         shap_values, X_sample,
    #         plot_type=plot_type,
    #         max_display=max_display,
    #         show=self.show_figures
    #     )
    #     plt.title(title)
    #     plt.tight_layout()

    #     plot_name = f"shap_{plot_type}_{model_key}"
    #     if save_plot and plot_name in self.paths:
    #         plt.savefig(self.paths[plot_name])
    #         print(f"📊 Saved SHAP {plot_type} graph for {model_key} to: {self.paths[plot_name]}")
    #     if self.show_figures:
    #         plt.show()
    #     plt.close()

# def populate_predictions_on_test(self, return_proba=True, shift_features=False):
    #     """
    #     Populate self.X_test with model predictions (either class or probability).

    #     Appends a new column:
    #     - 'model_pred_proba' if return_proba is True
    #     - 'model_pred' if return_proba is False
    #     """

    #     if self.X_test is None or self.X_test.empty:
    #         raise ValueError("🚫 self.X_test is empty. Run fit() first.")

    #     X_test_pred = self.X_test.copy() # Create a DataFrame version of X_test for safe modification

    #     # Reconstruct the original features for the test set (raw input needed for predict)
    #     # Assumes self.X_raw or df is still accessible and indexing consistency

    #     if not hasattr(self, "X_raw") or self.X_raw is None:
    #         raise ValueError("Raw input data (self.X_raw) is required to re-generate test set features.")

    #     # Align raw input with test set indices
    #     X_test_raw = self.X_raw.loc[self.X_test.index]

    #     # Add predictions to a new copy of X_test
    #     preds = self.predict(X_test_raw, return_proba=return_proba, shift_features=shift_features)
    #     pred_col = "model_pred_proba" if return_proba else "model_pred"
    #     X_test_pred[pred_col] = preds

    #     return X_test_pred

    # def get_training_results(self):
    #     y_pred = self.model.predict(self.X_val[self.feature_names])
    #     return {"strategy": self.strategy,
    #             "timeframe": self.timeframe,
    #             "target": self.target,
    #             "validation_accuracy": round(accuracy_score(self.y_val, y_pred), 4),
    #             "recall": round(recall_score(self.y_val, y_pred), 4),
    #             "precision": round(precision_score(self.y_val, y_pred), 4),
    #             "cv_score": round(self.cross_validate(self.X_train, self.y_train).mean(), 4)}



# def train_surrogate_tree(self, max_depth=3, random_state=42, save_plot=True):
    #     """Train a surrogate decision tree to approximate the trained model."""
    #     if not self.trained:
    #         raise RuntimeError("Train model before fitting surrogate tree.")

    #     y_pred = self.model.predict(self.X_train[self.feature_names])
    #     surrogate = DecisionTreeClassifier(max_depth=max_depth, random_state=random_state)
    #     surrogate.fit(self.X_train[self.feature_names], y_pred)

    #     print(f"Surrogate approximation accuracy: {accuracy_score(y_pred, surrogate.predict(self.X_train[self.feature_names])):.4f}")
    #     print(f"Surrogate vs true labels accuracy: {accuracy_score(self.y_train, surrogate.predict(self.X_train[self.feature_names])):.4f}")

    #     plt.figure(figsize=(14, 6))
    #     plot_tree(surrogate, feature_names=self.feature_names, class_names=['Class 0', 'Class 1'], filled=True, rounded=True, max_depth=max_depth, fontsize=10)
    #     plt.title("Surrogate Decision Tree")
    #     plot_name = 'surrogate_tree'
    #     if save_plot and plot_name in self.paths:
    #         # plt.savefig(os.path.join(self.figures_folder, f"surrogate_tree_{self.strategy}_{self.timeframe}_{self.target}.png"))
    #         plt.savefig(self.paths[plot_name])
    #         print(f"📊 Saved surrogate tree to: {self.paths[plot_name]}")
    #     if self.show_figures: plt.show()
    #     plt.close()

    #     return surrogate

# def extract_tree_rules(self, tree=None, feature_names=None, class_names=None, decimals=4, display_rules=True):

#         tree = tree or self.surrogate_tree
#         feature_names = feature_names or self.feature_names
#         class_names=class_names or ['Class 0', 'Class 1'] if self.model_subtype != 'regression' else None

#         tree_ = tree.tree_
#         feature_name = [feature_names[i] if i != _tree.TREE_UNDEFINED else 'undefined!' for i in tree_.feature]

#         rules = []

#         def recurse(node, depth, rule_path, rule_path_original):
#             if tree_.feature[node] != _tree.TREE_UNDEFINED:
#                 name = feature_name[node]
#                 threshold = tree_.threshold[node]
#                 rounded_threshold = round(threshold, decimals)
#                 left_rule = rule_path + [f"{name} <= {round(threshold, decimals)}"]
#                 right_rule = rule_path + [f"{name} > {round(threshold, decimals)}"]
#                 recurse(tree_.children_left[node], depth + 1, left_rule)
#                 recurse(tree_.children_right[node], depth + 1, right_rule)
#             else:
#                 # value = tree_.value[node][0]
#                 # sample_count = int(np.sum(value))

#                 sample_count = int(tree_.n_node_samples[node])  # total samples at this node
#                 value = tree_.value[node][0] * sample_count

#                 if class_names is not None:
#                     probs = value / value.sum()
#                     prediction_idx = np.argmax(probs)
#                     prediction = class_names[prediction_idx]
#                     prob = round(probs[prediction_idx], decimals)
#                     full_prob_dist = {class_names[i]: round(p, decimals) for i, p in enumerate(probs)}
#                 else:
#                     # prediction = round(value[0], decimals)
#                     # prob = None
#                     prediction = int(np.argmax(value))
#                     prob = round(np.max(value) / np.sum(value), decimals)
#                     full_prob_dist = None

#                 rules.append({"conditions": rule_path, "prediction": prediction, "probability": prob,
#                               "prob_distribution": full_prob_dist, "samples": sample_count})

#         recurse(0, 1, [])

#         if display_rules:
#             # Print or export rules
#             for i, rule in enumerate(rules):
#                 cond_str = "IF " + " AND ".join(rule["conditions"])
#                 prob_str = f" with probability {rule['probability']}" if rule["probability"] is not None else ""
#                 print(f"Rule {i + 1}: {cond_str} THEN Predict: {rule['prediction']}{prob_str} [Samples: {rule['samples']}]")

#         return rules

    # # Match pattern: results_breakout_up_1D.csv or results_breakout_down_60min.csv
    # pattern = os.path.join(strategy_folder, 'results_*.csv')
    # # pattern = os.path.join(strategy_folder, 'results_*.parquet')
    # files = glob.glob(pattern)

    # # === Regex to extract strategy, direction, timeframe and target from filename ===
    # # regex = re.compile(r'results_(?P<strategy>.+)_(?P<timeframe>[^_]+)\.csv$')
    # regex = re.compile(r'results_(?P<strategy>.+)_(?P<timeframe>[^_]+)_(?P<target>[^.]+)\.csv$')


    # # === Loop through each file ===
    # for file_path in files:
    #     filename = os.path.basename(file_path)
    #     match = regex.match(filename)
    #     if not match:
    #         print(f"Skipping unrecognized file: {filename}")
    #         continue

    #     strategy_name = match.group('strategy')
    #     timeframe = match.group('timeframe')
    #     target = match.group('target')

    #     # === Load Data ===
    #     df = pd.read_csv(file_path)


    #     # Create per-model log file path
    #     timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    #     log_file_name = f"training_log_{strategy_name}_{timeframe}_{target}.txt"
    #     log_file_path = os.path.join(figures_folder, log_file_name)
    #     with logs.LogContext(log_file_path, overwrite=True):  # Logging starts here

    #         print(f"⏰ Date: {datetime.datetime.now}\n")
    #         print(f"🔍 Training model for strategy: {strategy_name}, timeframe: {timeframe}, target: {target}")
    #         if not df.empty:
    #             print(f"Tickers analyzed:\n{df['symbol'].unique().tolist()}")
    #             print(f"\nDate range from  {df['data_from_time'].iloc[0]}  to  {df['data_to_time'].iloc[0]}\n")
    #         else: print("No data, Dataframe is empty.")

    #         trainer = ModelTrainer(
    #             model_type='xgboost',
    #             # model_type='linear',
    #             # model_type='rf',
    #             label_column='label_binary',
    #             # label_column='label_multi',
    #             # label_column='label_regression',
    #             # label_column='label_R-R',
    #             strategy=strategy_name,
    #             timeframe=timeframe,
    #             target=target,
    #             clip_outliers=True,
    #             show_clipping_report=False,
    #             show_figures=False,
    #             plot_trim_distributions=False
    #         )

    #         # Full training pipeline
    #         trainer.fit(df)

    #         # Save the model
    #         trainer.save()

    #         # Load it later
    #         # trainer.load()

    #         # post = ModelPostAnalysis(trainer.model, trainer.X_train[trainer.feature_names], trainer.y_train,
    #         #                          trainer.strategy, trainer.timeframe, trainer.target,
    #         #                          feature_names=trainer.feature_names, figures_folder=figures_folder,
    #         #                          show_figures=False)

    #         # SHAP dependence plot
    #         # post.compute_shap()
    #         # print("Post Analyser shap features:\n")
    #         # print(post.shap_values.feature_names)
    #         show_post_pots = False
    #         trainer.plot_dependence(trainer.shap_values.feature_names[0])
    #         trainer.plot_dependence(trainer.shap_values.feature_names[1])
    #         trainer.plot_dependence(trainer.shap_values.feature_names[2])
    #         # post.plot_dependence("target")
    #         if "market_cap_log" in trainer.shap_values.feature_names: trainer.plot_dependence("market_cap_log")
    #         # post.plot_dependence("sr_1W_dist_to_next_down_pct")

    #         # Decision Tree Surrogate
    #         surrogate = trainer.train_surrogate_tree(max_depth=3)

    #         # # Evaluate on live validation data
    #         # trainer.predict(df_live)

    #         # # Optional: cross-validate before training
    #         # X, y, _ = trainer.prepare_features(df_train)
    #         # X_train, X_val, y_train, y_val = trainer.select_features(X, y)
    #         # trainer.cross_validate(X_train, y_train)

    #         # # Optional: visualize confusion matrix
    #         # trainer.plot_confusion_matrix(X_val, y_val)





        # trainer = ModelTrainer(
        #     # model_type='xgboost',
        #     model_type='linear',
        #     # model_type='rf',
        #     model_subtype='auto',
        #     # label_column='label_binary',
        #     # label_column='label_multi',
        #     # label_column='label_regression',
        #     label_column='label_R-R',
        #     plot_feature_importance=False,
        #     clip_outliers=True,
        #     show_clipping_report=False,
        #     plot_trim_distributions=False
        # )

        # print("\nFeature preprocessing...\n")
        # trainer.fit(df)



# class ModelPostAnalysis:
#     def __init__(self, model, X, y, strategy, timeframe, target, feature_names=None,
#                  figures_folder=None, show_figures=False):
#         self.model = model
#         self.X = X
#         self.y = y
#         self.strategy = strategy
#         self.timeframe = timeframe
#         self.target = target
#         self.feature_names = feature_names if feature_names is not None else X.columns
#         self.shap_values = None
#         self.figures_folder = figures_folder
#         if self.figures_folder: os.makedirs(self.figures_folder, exist_ok=True)
#         self.show_figures = show_figures

#     def compute_shap(self):
#         explainer = shap.Explainer(self.model, self.X)
#         self.shap_values = explainer(self.X)
#         return self.shap_values

#     # def plot_dependence(self, feature):
#     #     if self.shap_values is None:
#     #         self.compute_shap()
#     #     title = f"SAP dependence for {self.strategy} {self.timeframe}"
#     #     shap.plots.scatter(self.shap_values[:, feature], color=self.shap_values, title=title)

#     def plot_dependence(self, feature, save_plot=True):
#         if self.shap_values is None:
#             self.compute_shap()

#         title = f"SHAP dependence for {self.strategy} {self.timeframe} {self.target}"

#         # Get column index from feature name
#         if isinstance(feature, str):
#             try:
#                 feature_index = self.shap_values.feature_names.index(feature)
#             except ValueError:
#                 raise ValueError(f"Feature '{feature}' not found in SHAP feature names.")
#         else:
#             feature_index = feature  # If it's already an integer
#             feature = self.shap_values.feature_names[feature_index]

#         # shap.plots.scatter(
#         #     self.shap_values[:, feature_index],
#         #     color=self.shap_values,
#         #     title=title
#         # )

#         # Create plot
#         shap.dependence_plot(feature, self.shap_values.values, self.X, feature_names=self.X.columns, show=self.show_figures)

#         # Optionally save plot
#         if save_plot:
#             filename = f"dependence_{self.strategy}_{self.timeframe}_{self.target}_{feature}.png"
#             save_path = os.path.join(self.figures_folder, filename)
#             plt.title(title)
#             plt.tight_layout()
#             plt.savefig(save_path)
#             plt.close()
#             print(f"📊 Saved SHAP dependence plot: {save_path}")

#     def train_surrogate_tree(self, max_depth=3, random_state=42):

#         y_pred = self.model.predict(self.X)
#         surrogate = DecisionTreeClassifier(max_depth=max_depth, random_state=random_state)
#         surrogate.fit(self.X, y_pred)

#         print(f"Surrogate approximation accuracy: {accuracy_score(y_pred, surrogate.predict(self.X)):.4f}")
#         print(f"Surrogate vs true labels accuracy: {accuracy_score(self.y, surrogate.predict(self.X)):.4f}")

#         plt.figure(figsize=(14, 6))
#         plot_tree(surrogate, feature_names=self.feature_names, class_names=['Class 0', 'Class 1'], filled=True,
#                   rounded=True, max_depth=max_depth, fontsize=10)
#         plt.title("Surrogate Decision Tree")
#         if self.figures_folder:
#             plt.savefig(os.path.join(self.figures_folder, f"surrogate_tree_{self.strategy}_{self.timeframe}_{self.target}.png"))
#         if self.show_figures: plt.show()

#         return surrogate

















# class FeaturePreprocessor:
#     def __init__(self, label_column='label_binary', remove_cols=None, transform_features=False,
#                  clip_outliers=False, clip_range=(0.01, 0.99),
#                  show_clipping_report=False, plot_trim_distributions=False):

#         self.label_column = label_column
#         # self.scale = scale
#         # self.model_type = model_type
#         self.transform_features = transform_features
#         self.clip_outliers = clip_outliers
#         self.clip_range = clip_range
#         self.should_scale = False # ✅ Auto-determine whether scaling is needed based on model type
#         self.scaler = None
#         self.show_clipping_report = show_clipping_report
#         self.plot_trim_distributions = plot_trim_distributions
#         self.clip_log = {}

#         self.leakage_keywords = [
#             'target', 'label', 'rtn', 'profit', 'drawdown', 'sharpe', 'sortino', 'recovery',
#             'event_duration', 'target_volatility', 'first_event', 'post_trig', 'end_'
#         ]

#         # Manually added leakage columns not caught by keyword logic
#         self.manual_leakage_cols = remove_cols or [
#             'time_to_max', 'time_to_max_ratio', 'time_to_min', 'time_to_min_ratio', 'time_to_recovery',
#             'time_to_recovery_ratio', 'max_drawdown_per_min', 'open', 'low', 'high', 'trig_close',
#             'pml', 'pmh', 'pdl', 'pdh', 'pdl_D', 'pdh_D', 'pdc', 'do', 'pMl', 'pMh', 'pMc', 'Mo', 'pivots',
#             'pivots_D', 'pivots_M', 'session', 'bb_width', 'atr_D_band_high', 'atr_D_band_low'
#         ]

#     # @staticmethod
#     # def detect_leakage(df, target_col, threshold=0.95):
#     #     suspicious = []
#     #     for col in df.columns:
#     #         if col == target_col or not pd.api.types.is_numeric_dtype(df[col]):
#     #             continue
#     #         corr = df[col].corr(df[target_col])
#     #         if abs(corr) > threshold:
#     #             suspicious.append((col, corr))
#     #     return sorted(suspicious, key=lambda x: abs(x[1]), reverse=True)

#     def prepare_features(self, df: pd.DataFrame):
#         if self.label_column not in df.columns:
#             raise ValueError(f"Label column '{self.label_column}' not found in dataframe.")


#         all_remove_cols = set(self.manual_leakage_cols)
#         # all_remove_cols.update(self.manual_leakage_cols)

#         for col in df.columns:
#             if col == self.label_column:
#                 continue
#             if any(kw in col.lower() for kw in self.leakage_keywords):
#                 all_remove_cols.add(col)

#         print(f"🧹 Removing {len(all_remove_cols)} potential leakage columns: {sorted(all_remove_cols)}")

#         feature_cols = [
#             col for col in df.columns
#             if col not in all_remove_cols
#             and col != self.label_column
#             and df[col].dtype in [np.float64, np.int64]
#         ]

#         X = df[feature_cols].copy()
#         X = X.replace([np.inf, -np.inf], np.nan).fillna(0)

#         y = df[self.label_column]

#         if self.clip_outliers:
#             self._clip_outliers(X)

#         if self.transform_features:
#             self._apply_feature_transformations(X)

#         # # ✅ Auto-determine whether scaling is needed based on model type
#         # should_scale = self.scale

#         if self.should_scale:
#             self.scaler = StandardScaler()
#             # X = pd.DataFrame(self.scaler.fit_transform(X), columns=feature_cols)
#             X = pd.DataFrame(self.scaler.fit_transform(X), columns=X.columns, index=X.index)


#         # print("\n📌 Using features:", feature_cols, f"\n({len(feature_cols)} total)\n")
#         print(f"🧪 Scaling Applied: {self.should_scale}\n")
#         print(f"🧬 Feature Transformation Applied: {self.transform_features}\n")
#         print("\n📌 Using features:", X.columns.to_list(), f"\n({len(X.columns)} total)\n")

#         return X, y, list(X.columns)

#     def _apply_feature_transformations(self, X):

#         # Transformation definitions
#         _apply_pct_diff_transformation = lambda col_sub, col_den: (X['close'] - X[col_sub]) / X[col_den]
#         _apply_pct_transformation = lambda col_num: X[col_num] / X['close']

#         # 🔹 Log transform for skewed volume-based features
#         log_cols = [col for col in X.columns if any(re.search(feature, col) for feature in ['volume', 'vpa', 'avg_vol', 'vol_change', 'vol_ratio'])] # 'market_cap'
#         # log_cols = [col for col in X.columns if ('volume' in col or 'vpa' in col or 'avg_vol' in col or 'market_cap' in col or 'vol_change' in col or 'vol_ratio' in col)]
#         for col in log_cols:
#             if col in X.columns and (X[col] > 0).all():
#                 # X[col] = np.log1p(X[col])
#                 X[f'{col}_log'] = np.log1p(X[col])
#                 X.drop(columns=col, inplace=True)

#         # 🔹 Apply category type to category features
#         for col in X.columns:
#             if col.endswith('_cat'):
#                 X[col] = X[col].astype('category')


#         # 🔹 Cyclic transform for time-based features
#         cyclic_cols = {'hour_of_day': 24, 'day_of_week': 7}
#         for col, period in cyclic_cols.items():
#             if col in X.columns:
#                 X[f'{col}_sin'] = np.sin(2 * np.pi * X[col] / period)
#                 X[f'{col}_cos'] = np.cos(2 * np.pi * X[col] / period)
#                 X.drop(columns=col, inplace=True)


#         # 🔹 EMA/SMA and levels % difference features
#         if 'close' in X.columns:
#             # Price Change Ratios or Log Returns
#             X['log_return'] = np.log(X['close'] / X['close'].shift(1)).fillna(0)

#             for col in ['ema9', 'ema20', 'sma50', 'sma200', 'vwap']:
#                 if col in X.columns:
#                     X[f'{col}_pct_diff'] = _apply_pct_diff_transformation(col, 'close')#(X['close'] - X[col]) / X['close']
#                     X[f'{col}_slope'] = X[col].diff()

#                     if 'atr' in X.columns:
#                         X[f'{col}_dist_pct_atr'] = _apply_pct_diff_transformation(col, 'atr').replace([np.inf, -np.inf], 0)#((X['close'] - X[col]) / X['atr']).replace([np.inf, -np.inf], 0)
#                         X[f'{col}_slope_pct_of_atr'] = X[f'{col}_slope'] / X['atr']

#                 X.drop(columns=col, inplace=True)

#             for col in ['low_of_day', 'high_of_day']:
#                 if col in X.columns:
#                     X[f'{col}_pct_diff'] = _apply_pct_diff_transformation(col, 'close')
#                     X.drop(columns=col, inplace=True)

#             for col in ['atr', 'macd', 'macd_signal', 'macd_diff', 'body', 'bband_h', 'bband_l', 'bband_mavg']:
#                 if col in X.columns:
#                     X[f'{col}_pct'] = _apply_pct_transformation(col)
#                     X.drop(columns=col, inplace=True)

#             # 🔹 Add SR and Pivot distance features
#             for col in X.columns:
#                 if '_dist_to_next_' in col:
#                     X[f'{col}_pct'] = _apply_pct_transformation(col)
#                     X.drop(columns=col, inplace=True)
#                 # if col.startswith(('sr_', 'pivots_', 'levels_')) and not any(x in col for x in [
#                 #     '_pos_in_range', '_dist_to_next_up', '_dist_to_next_down', '_D_', '_M_']):
#                 #     X[f'{col}_dist_from_close'] = _apply_pct_diff_transformation(col, 'close')
#                 #     X.drop(columns=col, inplace=True)

#     def _clip_outliers(self, X):
#         lower = X.quantile(self.clip_range[0])
#         upper = X.quantile(self.clip_range[1])
#         clip_log = {}

#         for col in X.columns:
#             original = X[col].copy()
#             X[col] = X[col].clip(lower=lower[col], upper=upper[col])
#             clipped_low = (original < lower[col]).sum()
#             clipped_high = (original > upper[col]).sum()
#             clip_log[col] = {
#                 'clipped_low': int(clipped_low),
#                 'clipped_high': int(clipped_high),
#                 'total_clipped': int(clipped_low + clipped_high)
#             }

#             if self.plot_trim_distributions and (clipped_low + clipped_high > 0):
#                 self._plot_trim_distribution(original, X[col], col)

#         self.clip_log = pd.DataFrame(clip_log).T.sort_values(by='total_clipped', ascending=False)

#         if self.show_clipping_report:
#             print("\n📊 Clipping Summary:")
#             print(self.clip_log.to_string())

#     def set_scale(self, model):
#         """Deduces if scaling is necessary based on model class."""

#         scale_required_models = (
#             LinearRegression, LogisticRegression,
#             SVC, SVR,
#             KNeighborsClassifier, KNeighborsRegressor
#         )
#         self.should_scale = isinstance(model, scale_required_models)

#     def _plot_trim_distribution(self, before, after, col_name):
#         before = pd.Series(before).replace([np.inf, -np.inf], np.nan).dropna()
#         after = pd.Series(after).replace([np.inf, -np.inf], np.nan).dropna()

#         if len(before) == 0 or len(after) == 0:
#             print(f"⚠️ Skipping plot for {col_name} — all values are non-finite.")
#             return

#         plt.figure(figsize=(10, 4))
#         plt.hist(before, bins=50, alpha=0.5, label='Before Clipping')
#         plt.hist(after, bins=50, alpha=0.5, label='After Clipping')
#         plt.title(f"Distribution Before vs After Clipping: {col_name}")
#         plt.legend()
#         plt.tight_layout()
#         plt.show()


# class RandomForestFeatureSelector:
#     def __init__(self, model_subtype='auto', n_estimators=100, test_size=0.2, random_state=42):
#         self.model_subtype = model_subtype
#         self.n_estimators = n_estimators
#         self.random_state = random_state
#         self.test_size = test_size
#         self.model = None
#         self.feature_importances_ = None
#         self.actual_model_subtype = None  # Stores resolved model_subtype
#         self.train_score = None
#         self.feature_count = 0

#     def _infer_model_subtype(self, y):
#         if self.model_subtype != 'auto':
#             return self.model_subtype

#         if y.dtype.kind in 'f' or (y.dtype.kind in 'i' and len(np.unique(y)) > 10):
#             return 'regression'
#         elif y.dtype.kind in 'i' and len(np.unique(y)) > 2:
#             return 'multi'
#         else:
#             return 'classification'

#     def train(self, X, y):
#         self.actual_model_subtype = self._infer_model_subtype(y)
#         self.feature_count = X.shape[1]

#         if self.actual_model_subtype == 'regression':
#             self.model = RandomForestRegressor(n_estimators=self.n_estimators, random_state=self.random_state)
#         else:
#             self.model = RandomForestClassifier(n_estimators=self.n_estimators, random_state=self.random_state)

#         try:
#             X_train, X_val, y_train, y_val = train_test_split(
#                 X, y, test_size=self.test_size, shuffle=False
#             )
#         except ValueError as e:
#             print(f"⚠️ Skipping feature selection due to small sample size: {e}")
#             # Fallback: use all data for training and validation to avoid breaking pipeline
#             return X, X, y, y

#         X_train = X_train.replace([np.inf, -np.inf], np.nan).fillna(0) # Making sure no unwanted value hhas slipped through the cracks
#         self.model.fit(X_train, y_train)
#         self.train_score = self.model.score(X_train, y_train)
#         return X_train, X_val, y_train, y_val

#     def get_feature_importance(self, feature_names, top_n=None):
#         if self.model is None:
#             raise ValueError("Train the model first.")

#         try:
#             check_is_fitted(self.model)
#         except NotFittedError:
#             print("⚠️ Model not fitted. Skipping feature importance calculation.")
#             return pd.Series(dtype='float64')  # Return empty series

#         importances = pd.Series(self.model.feature_importances_, index=feature_names)
#         self.feature_importances_ = importances.sort_values(ascending=False)

#         if top_n:
#             return self.feature_importances_.head(top_n)
#         else:
#             return self.feature_importances_.to_string()

#     def plot_feature_importance(self, additional_title='', top_n=50):
#         if self.feature_importances_ is None:
#             raise ValueError("Train the model and extract importances before plotting.")

#         if top_n:
#             self.feature_importances_.head(top_n).plot(kind='barh', figsize=(10, 8))
#         else:
#             self.feature_importances_.plot(kind='barh', figsize=(10, 8))

#         plt.title("Top Feature Importances" + additional_title)
#         plt.gca().invert_yaxis()
#         plt.tight_layout()
#         plt.show()

#     def summary(self):
#         print("Model Summary")
#         print("=" * 40)
#         print(f"Model subtype:       {self.actual_model_subtype}")
#         print(f"Model used:      {type(self.model).__name__}")
#         print(f"Features used:   {self.feature_count}")
#         if self.actual_model_subtype == 'regression':
#             print(f"Train R² score:  {self.train_score:.4f}")
#         else:
#             print(f"Train accuracy:  {self.train_score:.4f}")
#         print("=" * 40)