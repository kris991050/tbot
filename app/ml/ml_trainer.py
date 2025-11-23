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
                 target:str='', base_models:str=None, base_folder:str=None, top_n_features:int=10, train_size:float=0.7, 
                 validation_size:float=0.15, depth_surrogate:int=3, clip_outliers:bool=True, show_figures:bool=False, shap_types:list=None, 
                 drawdown:bool=False, **preprocessor_kwargs):

        self.model_type = model_type
        self.base_models = base_models or ['rf', 'xgboost', 'linear']
        self.strategy_name = strategy_name
        timeframe = trade_manager.get_strategy_instance(self.strategy_name).timeframe
        self.timeframe_str = timeframe.pandas
        self.target = target
        self.entry_delay = helpers.get_entry_delay_from_timeframe(timeframe)
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
        filename_pattern = f"{self.strategy_name}_{self.target}_delay{self.entry_delay}"
        self.paths['data'] = os.path.join(self.base_folder, f"{d}results_{filename_pattern}.csv")
        # rev = '' if not self.revised else '_revised'
        if self.figures_folder:
            self.paths['confusion_matrix'] = os.path.join(self.figures_folder, f"{d}confusion_matrix_{filename_pattern}.png")
            self.paths['dependence_plot'] = os.path.join(self.figures_folder, f"{d}dependence_{filename_pattern}.png")
            self.paths['surrogate_tree'] = os.path.join(self.figures_folder, f"{d}surrogate_tree_{filename_pattern}.png")
            for shap_type in self.shap_types:
                self.paths['shap_' + shap_type] = os.path.join(self.figures_folder, f"{d}shap_{shap_type}_{filename_pattern}.png")

        if self.models_folder: self.paths['models'] = os.path.join(self.models_folder,f"{d}model_{filename_pattern}.pkl")
        if self.shaps_folder: self.paths['shaps'] = os.path.join(self.shaps_folder, f"{d}shap_{filename_pattern}.pkl")
        # if self.test_set_folder: self.paths['test_set'] = os.path.join(self.test_set_folder, f"test_set_{filename_pattern}.csv")

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
            "entry_delay": self.entry_delay
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
            self.explainer.plot_shaps(self.strategy_name, self.timeframe_str, self.target, self.entry_delay, X_sample)
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

        title = title + f" startegy {self.strategy_name}, timeframe {self.timeframe_str}, target {self.target}, delay {self.entry_delay}"
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
                'entry_delay': self.entry_delay, 
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
        if save_models_name in self.paths and os.path.exists(self.paths[save_models_name]):
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
            self.entry_delay = data['entry_delay']
            self.label_column = data['label_column']
            self.data_ranges = data['data_ranges']
            self.preprocessor.transform_features = data.get('transform_features', True)
            self._create_explainer()
            print(f"📦 Model loaded from: {self.paths[save_models_name]}")
        else:
            # raise ValueError(f"❌ No model found at {self.paths[save_models_name]}. No model loaded.")
            print(f"⚠️ No model found at {self.paths[save_models_name]}. No model loaded.")


class StrategyModelComparator:
    def __init__(self, strategy_name:str, clip_outliers:bool=True, base_folder:str=None, model_type:str='xgboost', selector_type:str='rf', 
                 label_column:str='label_binary', depth_surrogate:int=3, dd_mode:bool=False):
        self.strategy_name = strategy_name
        self.clip_outliers=clip_outliers
        model_type = model_type if not dd_mode else 'xgboost'
        label_column = label_column if not dd_mode else 'max_drawdown_pct'
        self.trainer = ModelTrainer(model_type=model_type, selector_type=selector_type, label_column=label_column,
                               strategy_name=strategy_name, target='', clip_outliers=clip_outliers, base_folder=base_folder,
                               depth_surrogate=depth_surrogate, show_clipping_report=False, show_figures=False, plot_trim_distributions=False)
        # self.trainer_dd = ModelTrainer(model_type='xgboost', selector_type=selector_type, label_column='max_drawdown_pct',
        #                        strategy_name=strategy_name, target='', clip_outliers=clip_outliers, base_folder=base_folder,
        #                        depth_surrogate=depth_surrogate, show_clipping_report=False, show_figures=False, drawdown=True,
        #                        plot_trim_distributions=False)

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
        dd_path = '' if not dd_mode else '_d'
        self.comparison_file_name = f"comparison{dd_path}_{self.strategy_name}.csv"
        # self.comparison_file_name_dd = f"comparison_dd_{self.strategy_name}.csv"
        # self.comparison_file_path = os.path.join(self.base_folder, self.comparison_file_name)
        self.comparison_file_path = os.path.join(self.trainer.outputs_folder, self.comparison_file_name)
        # self.comparison_file_path_dd = os.path.join(self.trainer_dd.outputs_folder, self.comparison_file_name_dd)

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
            # target = base_name[len(strategy_name):].strip('_')  # Remove any leading underscores, target will be everything that comes after the strategy_name part

            # Extract the target by removing strategy_name and anything after "_delay"
            target_start = len(self.strategy_name) # Start after strategy_name
            delay_pos = base_name.find('_delay') # Check if '_delay' exists and find its position
            target = base_name[target_start:delay_pos].strip('_')
            entry_delay = int(base_name[delay_pos + 6:])  # 6 is the length of '_delay'

            # trainer = ModelTrainer(model_type=self.model_type, selector_type=self.selector_type, label_column=self.label_column,
            #                         strategy=strategy_name, timeframe=timeframe, target=target, clip_outliers=self.clip_outliers,
            #                         depth_surrogate=self.depth_surrogate, show_clipping_report=False, show_figures=False, plot_trim_distributions=False)
            for t in [self.trainer]:#, self.trainer_dd]:
                t.target = target
                t.entry_delay = entry_delay
                t.create_paths()

            log_file_path = os.path.join(self.logs_folder, f"training_log_{strategy_name}_{self.trainer.timeframe_str}_{target}_delay{entry_delay}.txt")

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
                # results_dd = self.trainer_dd.fit(df)

                for t in [self.trainer]:#, self.trainer_dd]:
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
                # self.results_dd.append(results_dd)
                # except Exception as e:
                #     print(f"⚠️ Failed to train model for {self.strategy_name} {timeframe} {target}. Error: {e}")

        self._save()

    def _save(self):
        df = pd.DataFrame(self.results)
        os.makedirs(self.trainer.outputs_folder, exist_ok=True)
        df.to_csv(self.comparison_file_path, index=False)
        print(f"✅ Saved comparison results at {self.comparison_file_path}")

        # df_dd = pd.DataFrame(self.results_dd)
        # os.makedirs(self.trainer_dd.outputs_folder, exist_ok=True)
        # df_dd.to_csv(self.comparison_file_path_dd, index=False)
        # print(f"✅ Saved comparison results drawdowns at {self.comparison_file_path_dd}")


if __name__ == "__main__":

    args = sys.argv

    pd.options.mode.chained_assignment = None # Disable Pandas warnings

    # Args Setup
    args = sys.argv
    # display_res = 'display' in args
    revised = 'revised' in args
    dd_mode = 'dd' in args
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

    strategy_model_comparator = StrategyModelComparator(strategy_name, clip_outliers=True, model_type=model_type, selector_type=selector, 
                                                        label_column=label_column, dd_mode=dd_mode)
    strategy_model_comparator.run()
    # strategies_folder = PATHS.folders_path['strategies_data']
    # strategy_folder = os.path.join(strategies_folder, strategy)
