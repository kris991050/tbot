import sys, os, pandas as pd, numpy as np, matplotlib.pyplot as plt, re
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.svm import SVC, SVR
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor

parent_folder = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
sys.path.append(parent_folder)

from utils import constants


def apply_feature_transformations(X, transformation_log=None, drop=True):
    print(f"🐛 Applying feature transformations...")
    if transformation_log is None: transformation_log = [] # Mutable default arguments retain state between function calls—can lead to bugs when used in pipelines or tests.

    # X = X.copy() if isinstance(X, pd.DataFrame) else X.copy().to_frame().T if isinstance(X, pd.Series) else pd.DataFrame()
    if isinstance(X, pd.DataFrame):
        # X = X.copy()
        pass
    elif isinstance(X, pd.Series):
        X = X.to_frame().T  # Convert Series to single-row DataFrame
    else:
        X = pd.DataFrame()

    # Transformation definitions
    _apply_close_pct_diff_transformation = lambda col_sub, col_den: (X['close'] - X[col_sub]) / X[col_den]
    _apply_close_pct_transformation = lambda col_num: X[col_num] / X['close']

    # 🔹 Log transform for skewed volume-based features
    # Xcolumns = X.columns #if isinstance(X, pd.DataFrame) else X.index if isinstance(X, pd.Series) else []
    log_cols = [col for col in X.columns if any(re.search(feature, col) for feature in ['volume', 'vpa', 'avg_vol', 'vol_change', 'vol_ratio'])] # 'market_cap'
    new_cols = {}
    # log_cols = [col for col in X.columns if ('volume' in col or 'vpa' in col or 'avg_vol' in col or 'market_cap' in col or 'vol_change' in col or 'vol_ratio' in col)]
    for col in log_cols:
        if col in X.columns and (X[col] >= 0).all():
            new_col = f"{col}_log"
            # X[col] = np.log1p(X[col])
            # X[col] = pd.to_numeric(X[col], errors='coerce')
            new_cols[new_col] = np.log1p(pd.to_numeric(X[col]))
            transformation_log.append({'transformation': 'log1p', 'features': [col], 'output': new_col})
            X.drop(columns=col, inplace=True)

    # 🔹 Apply category type to category features
    for col in X.columns:
        if col.endswith('_cat'):
            new_cols[col] = X[col].astype('category')
        # if col in categorical_columns:
        #     X[f"{col}_cat"] = X[col].astype('category')
        #     X.drop(columns=col, inplace=True)


    # 🔹 Cyclic transform for time-based features
    cyclic_cols = {'hour_of_day': 24, 'day_of_week': 7}
    for col, period in cyclic_cols.items():
        if col in X.columns:
            sin_col, cos_col = f"{col}_sin", f"{col}_cos"
            # X[col] = pd.to_numeric(X[col], errors='coerce')
            new_cols[sin_col] = np.sin(2 * np.pi * pd.to_numeric(X[col]) / period)
            new_cols[cos_col] = np.cos(2 * np.pi * pd.to_numeric(X[col]) / period)
            transformation_log.append({'transformation': 'cyclic', 'features': [col],
                                            'output': [sin_col, cos_col], 'period': period})
            X.drop(columns=col, inplace=True)


    # 🔹 EMA/SMA and levels % difference features
    if 'close' in X.columns:

        # Price Change Ratios or Log Returns
        new_cols['log_return'] = np.log(pd.to_numeric(X['close']) / pd.to_numeric(X['close']).shift(1)).fillna(0)
        transformation_log.append({'transformation': 'log_return', 'features': ['close'], 'output': 'log_return'})

        # for col in ['ema9', 'ema20', 'sma50', 'sma200', 'vwap', 'bband_h', 'bband_l', 'bband_mavg']:
        for col in [col for col in X.columns if any(re.search(feature, col) for feature
                            in ['ema9', 'ema20', 'sma50', 'sma200', 'vwap', 'bband_h', 'bband_l', 'bband_mavg'])
                            and not any(exclude in col for exclude in ['slope', 'pct_diff'])]:
            if col in X.columns:
                new_col_pct_diff = f'{col}_pct_diff'
                new_col_slope = f'{col}_slope'
                new_cols[new_col_pct_diff] = _apply_close_pct_diff_transformation(col, 'close')#(X['close'] - X[col]) / X['close']
                new_cols[f'{col}_slope'] = X[col].diff()

                transformation_log.append({'transformation': 'pct_diff', 'features': [col],
                                                'denominator': 'close', 'output': new_col_pct_diff})
                transformation_log.append({'transformation': 'slope', 'features': [col],
                                                'output': new_col_slope})

                if 'atr' in X.columns:
                    new_col_dist_pct_atr = f'{col}_dist_pct_atr'
                    new_col_slope_pct_of_atr = f'{col}_slope_pct_of_atr'
                    # new_cols[new_col_dist_pct_atr] = _apply_close_pct_diff_transformation(col, 'atr').replace([np.inf, -np.inf], 0).infer_objects(copy=False)#((X['close'] - X[col]) / X['atr']).replace([np.inf, -np.inf], 0)

                    new_cols[new_col_dist_pct_atr] = (
                        _apply_close_pct_diff_transformation(col, 'atr')
                        .astype(float)
                        .replace([np.inf, -np.inf], 0)
                    )


                    new_cols[new_col_slope_pct_of_atr] = new_cols[f'{col}_slope'] / X['atr']
                    transformation_log.append({'transformation': 'pct_diff', 'features': [col],
                                                    'denominator': 'atr', 'output': new_col_dist_pct_atr})
                    transformation_log.append({'transformation': 'ratio', 'features': [f'{col}_slope', 'atr'],
                                                    'output': new_col_slope_pct_of_atr})

            X.drop(columns=col, inplace=True)

        for col in ['low_of_day', 'high_of_day']:
            if col in X.columns:
                new_col_pct_diff_day = f'{col}_pct_diff'
                new_cols[new_col_pct_diff_day] = _apply_close_pct_diff_transformation(col, 'close')
                transformation_log.append({'transformation': 'pct_diff', 'features': [col],
                                                'denominator': 'close', 'output': new_col_pct_diff_day})
                X.drop(columns=col, inplace=True)

        for col in ['atr', 'macd', 'macd_signal', 'macd_diff', 'body']:
            if col in X.columns:
                new_col_pct = f'{col}_pct'
                new_cols[new_col_pct] = _apply_close_pct_transformation(col)
                transformation_log.append({'transformation': 'pct', 'features': [col],
                                                'denominator': 'close', 'output': new_col_pct})
                X.drop(columns=col, inplace=True)

        # 🔹 Add SR and Pivot distance features
        for col in X.columns:
            if '_dist_to_next_' in col:
                new_col_pct_dist_next = f'{col}_pct'
                new_cols[new_col_pct_dist_next] = _apply_close_pct_transformation(col)
                transformation_log.append({'transformation': 'pct', 'features': [col],
                                                'denominator': 'close', 'output': new_col_pct_dist_next})
                X.drop(columns=col, inplace=True)
            # if col.startswith(('sr_', 'pivots_', 'levels_')) and not any(x in col for x in [
            #     '_pos_in_range', '_dist_to_next_up', '_dist_to_next_down', '_D_', '_M_']):
            #     X[f'{col}_dist_from_close'] = _apply_close_pct_diff_transformation(col, 'close')
            #     X.drop(columns=col, inplace=True)
    # X = X.replace([np.inf, -np.inf], np.nan).fillna(0)
    X = pd.concat([X, pd.DataFrame(new_cols, index=X.index)], axis=1)
    X = X.loc[:, ~X.columns.duplicated()] # Remove potential dupllicate columns

    num_cols = X.select_dtypes(include=[np.number]).columns
    X[num_cols] = X[num_cols].replace([np.inf, -np.inf], np.nan).fillna(0)
    # for col in X.select_dtypes(include=[np.number]).columns:
    #     X[col] = X[col].replace([np.inf, -np.inf], np.nan).fillna(0).infer_objects(copy=False)

    return X, transformation_log


class FeaturePreprocessor:
    def __init__(self, label_column='label_binary', remove_cols=None, transform_features=False,
                 clip_outliers=True, clip_range=(0.01, 0.99), model_subtype=None,
                 show_clipping_report=False, plot_trim_distributions=False):

        self.label_column = label_column
        self.model_subtype = model_subtype  # Actual resolved model_subtype
        self.transform_features = transform_features
        self.clip_outliers = clip_outliers
        self.clip_range = clip_range
        self.scaler = None
        self.show_clipping_report = show_clipping_report
        self.plot_trim_distributions = plot_trim_distributions
        self.clip_log = {}
        self.transformation_log = []

        self.leakage_keywords = [
            'target', 'label', 'rtn', 'profit', 'drawdown', 'sharpe', 'sortino', 'recovery', 'r-r', 
            'event_duration', 'target_volatility', 'first_event', 'post_trig', '_list', '~'
        ]

        # Manually added leakage columns not caught by keyword logic
        self.manual_leakage_cols = remove_cols or [
            'symbol', 'timeframe', 'strategy', 'target', 'data_to_time', 'data_from_time', 'low_of_day', 'high_of_day'
            'trig_time', 'date_D', 'date_M', 'time_to_max', 'time_to_max_ratio', 'time_to_min', 'time_to_min_ratio',
            'time_to_recovery', 'time_to_recovery_ratio', 'max_drawdown_per_min', 'open', 'low', 'high', 'trig_close',
            'session',
        ]

        # self.categorical_cols = ['market_cap_cat', 'session', 'first_event']
        self.categorical_cols = {'market_cap_cat': constants.CONSTANTS.MARKET_CAP_CATEGORIES.keys(),#['Unknown', 'Nano', 'Micro', 'Small', 'Mid', 'Large', 'Mega'],
                                   'session': None, 'first_event': None}

        # self.transformation_log = [
        #     {'transformation': 'log1p', 'features': ['volume', 'vpa', 'avg_vol', 'vol_change', 'vol_ratio']},
        #     {'transformation': "astype('category')", 'features': "endswith('_cat')"}
        #     {'transformation': "{'hour_of_day': 24, 'day_of_week': 7}"}
        #     {'transformation': 'pct_diff', 'features': ['ema9'], 'denominator': 'close', 'output': 'ema9_pct_diff'},
        #     {'transformation': 'cyclic', 'features': ['hour_of_day'], 'period': 24, 'output': ['hour_of_day_sin', 'hour_of_day_cos']},
        # ...
        # ]

    # def infer_model_subtype(self, y):
    #     if y.dtype.kind in 'f' or (y.dtype.kind in 'i' and len(np.unique(y)) > 10):
    #         return 'regression'
    #     elif y.dtype.kind in 'i' and len(np.unique(y)) > 2:
    #         return 'multi'
    #     else:
    #         return 'classification'

    def infer_model_subtype(self, y, max_length=10):
        unique_vals = np.unique(y)
        if np.issubdtype(y.dtype, np.floating):
            return 'regression'
        elif np.issubdtype(y.dtype, np.integer):
            if len(unique_vals) <= 2:
                return 'classification'
            elif len(unique_vals) <= max_length:
                return 'multi'
            else:
                return 'regression'
        else:
            # Handle string or object types
            if len(unique_vals) <= 2:
                return 'classification'
            else:
                return 'multi'

    def prepare_data_set(self, df:pd.DataFrame, model_subtype:str):
        if self.label_column not in df.columns:
            raise ValueError(f"Label column '{self.label_column}' not found in dataframe.")

        # Encode any string/categorical columns
        df = self.encode_categoricals(df)

        all_remove_cols = set(self.manual_leakage_cols)
        # all_remove_cols.update(self.manual_leakage_cols)

        for col in df.columns:
            if col == self.label_column:
                continue
            if any(kw in col.lower() for kw in self.leakage_keywords):
                all_remove_cols.add(col)

        print(f"🧹 Removing {len(all_remove_cols)} potential leakage columns: {sorted(all_remove_cols)}")

        feature_cols = [
            col for col in df.columns
            if col not in all_remove_cols
            and col != self.label_column
            and df[col].dtype in [np.float64, np.int64, np.int8]]

        X = df[feature_cols].copy()
        y = df[self.label_column]
        if not model_subtype: model_subtype = self.infer_model_subtype(y)

        return X, y, model_subtype

    def prepare_features(self, X, model, display_features=True, drop=True):

        # cat_cols = X.select_dtypes(include='category').columns
        # num_cols = X.columns.difference(cat_cols)
        # X[num_cols] = X[num_cols].replace([np.inf, -np.inf], np.nan).fillna(0) # Exclude categorica columns

        if self.clip_outliers:
            X = self._clip_outliers(X)

        if self.transform_features:
            X, self.transformation_log = apply_feature_transformations(X, self.transformation_log, drop=drop)

        # # ✅ Auto-determine whether scaling is needed based on model type
        should_scale = self._set_scale(model)
        if should_scale:
            self.scaler = StandardScaler()
            # X = pd.DataFrame(self.scaler.fit_transform(X), columns=feature_cols)
            X = pd.DataFrame(self.scaler.fit_transform(X), columns=X.columns, index=X.index)


        # print("\n📌 Using features:", feature_cols, f"\n({len(feature_cols)} total)\n")
        print(f"🧪 Scaling Applied: {should_scale}\n")
        print(f"🧬 Feature Transformation Applied: {self.transform_features}\n")
        if display_features: print("\n📌 Using features:", X.columns.to_list(), f"\n({len(X.columns)} total)\n")

        return X#, list(X.columns)

    def encode_categoricals(self, X):
        """Encode categorical string columns with optional ordering."""
        for col, category_order in self.categorical_cols.items():
            if col in X.columns:
                print(f"🔤 Encoding categorical column: {col}")
                try:
                    if category_order is not None:
                        # Ordered categorical encoding
                        X[col] = pd.Series(pd.Categorical(X[col], categories=category_order, ordered=True)).cat.codes
                        # cat = pd.Categorical(X[col], categories=category_order, ordered=True)
                        # X[col] = pd.Series(cat.codes.astype(np.int32), index=X.index)
                        # if 0 not in X[col].cat.categories: X[col] = X[col].cat.add_categories(-1)
                        # X[col]
                    else:
                        # Nominal categorical encoding using LabelEncoder
                        le = LabelEncoder()
                        X[col] = le.fit_transform(X[col].astype(str))  # Ensure string type

                except Exception as e:
                    print(f"⚠️ Could not encode column {col}: {e}")
        return X

    def reverse_transform_feature(self, threshold_value, feature_name, transformed_X=None,
                                  original_X=None, epsilon=1e-3):
        """
        Reverse a transformed feature threshold into its original-scale value.

        Args:
            threshold_value (float): The threshold from the transformed feature.
            feature_name (str): The transformed feature name (e.g., 'ema20_pct_diff').
            transformed_X (pd.DataFrame, optional): Transformed dataset used to train surrogate.
            original_X (pd.DataFrame, optional): Original untransformed dataset (before feature engineering).
            epsilon (float): Tolerance when finding rows near the threshold.

        Returns:
            dict or None: {
                'reversed_value': float,
                'feature': str (original name),
                'approx_from': 'local_mean' or 'global_mean'
            }
        """
        for entry in self.transformation_log:
            out = entry['output']
            if isinstance(out, list):
                if feature_name not in out:
                    continue
                out = feature_name

            if out != feature_name:
                continue

            t = entry['transformation']
            f = entry['features'][0]  # Original feature name
            d = entry.get('denominator', None)

            try:
                if t == 'log1p':
                    reversed_value = np.expm1(threshold_value)
                    return {'reversed_value': reversed_value, 'feature': f, 'approx_from': 'direct'}

                elif t == 'log_return':
                    reversed_value = np.exp(threshold_value)
                    return {'reversed_value': reversed_value, 'feature': f, 'approx_from': 'direct'}

                elif t in ['pct_diff', 'pct']:
                    if transformed_X is not None and original_X is not None:
                        # Find local sample around the threshold
                        mask = np.abs(transformed_X[feature_name] - threshold_value) < epsilon
                        subset = original_X.loc[mask]

                        if not subset.empty:
                            if t == 'pct_diff':
                                denom_val = subset[d].mean()
                                close_val = subset['close'].mean()
                                reversed_value = close_val - threshold_value * denom_val
                                approx_from = 'local_mean'

                            elif t == 'pct':
                                close_val = subset['close'].mean()
                                reversed_value = threshold_value * close_val
                                approx_from = 'local_mean'
                        else:
                            # Fall back to global average
                            denom_val = original_X[d].mean() if d else None
                            close_val = original_X['close'].mean()

                            if t == 'pct_diff' and denom_val is not None:
                                reversed_value = close_val - threshold_value * denom_val
                                approx_from = 'global_mean'
                            elif t == 'pct':
                                reversed_value = threshold_value * close_val
                                approx_from = 'global_mean'
                            else:
                                return None
                    else:
                        print(f"⚠️ Cannot reverse '{feature_name}' without transformed and original datasets.")
                        return None

                    return {'reversed_value': reversed_value, 'feature': f, 'approx_from': approx_from}

                else:
                    print(f"⚠️ Transformation '{t}' not supported for reversal.")
                    return None

            except Exception as e:
                print(f"⚠️ Failed to reverse {feature_name}: {e}")
                return None

        print(f"⚠️ No matching transformation found for: {feature_name}")
        return None

    def _clip_outliers(self, X):
        print(f"✂️ Clipping outliers...")
        numeric_X = X.select_dtypes(include=[np.number])
        lower = numeric_X.quantile(self.clip_range[0])
        upper = numeric_X.quantile(self.clip_range[1])
        # X_clipped = numeric_X.clip(lower, upper, axis=1)

        # lower = X.quantile(self.clip_range[0])
        # upper = X.quantile(self.clip_range[1])
        clip_log = {}

        for col in numeric_X.columns:
            original = numeric_X[col].copy()
            numeric_X[col] = numeric_X[col].clip(lower=lower[col], upper=upper[col])
            clipped_low = (original < lower[col]).sum()
            clipped_high = (original > upper[col]).sum()
            clip_log[col] = {
                'clipped_low': int(clipped_low),
                'clipped_high': int(clipped_high),
                'total_clipped': int(clipped_low + clipped_high)
            }

            if self.plot_trim_distributions and (clipped_low + clipped_high > 0):
                self._plot_trim_distribution(original, numeric_X[col], col)

        self.clip_log = pd.DataFrame(clip_log).T.sort_values(by='total_clipped', ascending=False)

        if self.show_clipping_report:
            print("\n📊 Clipping Summary:")
            print(self.clip_log.to_string())

        X[numeric_X.columns] = numeric_X

        return X

    def _set_scale(self, model):
        """Deduces if scaling is necessary based on model class."""

        scale_required_models = (LinearRegression, LogisticRegression, SVC, SVR, KNeighborsClassifier, KNeighborsRegressor)
        return isinstance(model, scale_required_models)

    def _plot_trim_distribution(self, before, after, col_name):
        before = pd.Series(before).replace([np.inf, -np.inf], np.nan).dropna()
        after = pd.Series(after).replace([np.inf, -np.inf], np.nan).dropna()

        if len(before) == 0 or len(after) == 0:
            print(f"⚠️ Skipping plot for {col_name} — all values are non-finite.")
            return

        plt.figure(figsize=(10, 4))
        plt.hist(before, bins=50, alpha=0.5, label='Before Clipping')
        plt.hist(after, bins=50, alpha=0.5, label='After Clipping')
        plt.title(f"Distribution Before vs After Clipping: {col_name}")
        plt.legend()
        plt.tight_layout()
        plt.show()


    # @staticmethod
    # def detect_leakage(df, target_col, threshold=0.95):
    #     suspicious = []
    #     for col in df.columns:
    #         if col == target_col or not pd.api.types.is_numeric_dtype(df[col]):
    #             continue
    #         corr = df[col].corr(df[target_col])
    #         if abs(corr) > threshold:
    #             suspicious.append((col, corr))
    #     return sorted(suspicious, key=lambda x: abs(x[1]), reverse=True)



    # def encode_categoricals(self, X):
    #     """Encode safe categorical string columns."""
    #     categorical_cols = self.categorical_cols if self.categorical_cols else X.select_dtypes(include='object').columns
    #     for col in categorical_cols:
    #         if col in X.columns:
    #             print(f"🔤 Encoding categorical column: {col}")
    #             le = LabelEncoder()
    #             try:
    #                 X[col] = le.fit_transform(X[col])
    #             except Exception as e:
    #                 print(f"⚠️ Could not encode column {col}: {e}")
    #     return X


    # def _apply_feature_transformations(self, X):

    #     X = X.copy()
    #     # Transformation definitions
    #     _apply_pct_diff_transformation = lambda col_sub, col_den: (X['close'] - X[col_sub]) / X[col_den]
    #     _apply_pct_transformation = lambda col_num: X[col_num] / X['close']

    #     # 🔹 Log transform for skewed volume-based features
    #     log_cols = [col for col in X.columns if any(re.search(feature, col) for feature in ['volume', 'vpa', 'avg_vol', 'vol_change', 'vol_ratio'])] # 'market_cap'
    #     # log_cols = [col for col in X.columns if ('volume' in col or 'vpa' in col or 'avg_vol' in col or 'market_cap' in col or 'vol_change' in col or 'vol_ratio' in col)]
    #     for col in log_cols:
    #         if col in X.columns and (X[col] >= 0).all():
    #             new_col = f"{col}_log"
    #             # X[col] = np.log1p(X[col])
    #             X[new_col] = np.log1p(X[col])
    #             self.transformation_log.append({'transformation': 'log1p', 'features': [col], 'output': new_col})
    #             X.drop(columns=col, inplace=True)

    #     # 🔹 Apply category type to category features
    #     for col in X.columns:
    #         if col.endswith('_cat'):
    #             X[col] = X[col].astype('category')


    #     # 🔹 Cyclic transform for time-based features
    #     cyclic_cols = {'hour_of_day': 24, 'day_of_week': 7}
    #     for col, period in cyclic_cols.items():
    #         if col in X.columns:
    #             sin_col, cos_col = f"{col}_sin", f"{col}_cos"
    #             X[sin_col] = np.sin(2 * np.pi * X[col] / period)
    #             X[cos_col] = np.cos(2 * np.pi * X[col] / period)
    #             self.transformation_log.append({'transformation': 'cyclic', 'features': [col],
    #                                             'output': [sin_col, cos_col], 'period': period})
    #             X.drop(columns=col, inplace=True)


    #     # 🔹 EMA/SMA and levels % difference features
    #     if 'close' in X.columns:

    #         # Price Change Ratios or Log Returns
    #         X['log_return'] = np.log(X['close'] / X['close'].shift(1)).fillna(0)
    #         self.transformation_log.append({'transformation': 'log_return', 'features': ['close'], 'output': 'log_return'})

    #         # for col in ['ema9', 'ema20', 'sma50', 'sma200', 'vwap', 'bband_h', 'bband_l', 'bband_mavg']:
    #         for col in [col for col in X.columns if any(re.search(feature, col) for feature in ['ema9', 'ema20', 'sma50', 'sma200', 'vwap', 'bband_h', 'bband_l', 'bband_mavg'])]:
    #             if col in X.columns:
    #                 new_col_pct_diff = f'{col}_pct_diff'
    #                 new_col_slope = f'{col}_slope'
    #                 X[new_col_pct_diff] = _apply_pct_diff_transformation(col, 'close')#(X['close'] - X[col]) / X['close']
    #                 X[f'{col}_slope'] = X[col].diff()

    #                 self.transformation_log.append({'transformation': 'pct_diff', 'features': [col],
    #                                                 'denominator': 'close', 'output': new_col_pct_diff})
    #                 self.transformation_log.append({'transformation': 'slope', 'features': [col],
    #                                                 'output': new_col_slope})

    #                 if 'atr' in X.columns:
    #                     new_col_dist_pct_atr = f'{col}_dist_pct_atr'
    #                     new_col_slope_pct_of_atr = f'{col}_slope_pct_of_atr'
    #                     X[new_col_dist_pct_atr] = _apply_pct_diff_transformation(col, 'atr').replace([np.inf, -np.inf], 0)#((X['close'] - X[col]) / X['atr']).replace([np.inf, -np.inf], 0)
    #                     X[new_col_slope_pct_of_atr] = X[f'{col}_slope'] / X['atr']
    #                     self.transformation_log.append({'transformation': 'pct_diff', 'features': [col],
    #                                                     'denominator': 'atr', 'output': new_col_dist_pct_atr})
    #                     self.transformation_log.append({'transformation': 'ratio', 'features': [f'{col}_slope', 'atr'],
    #                                                     'output': new_col_slope_pct_of_atr})

    #             X.drop(columns=col, inplace=True)

    #         for col in ['low_of_day', 'high_of_day']:
    #             if col in X.columns:
    #                 new_col_pct_diff_day = f'{col}_pct_diff'
    #                 X[new_col_pct_diff_day] = _apply_pct_diff_transformation(col, 'close')
    #                 self.transformation_log.append({'transformation': 'pct_diff', 'features': [col],
    #                                                 'denominator': 'close', 'output': new_col_pct_diff_day})
    #                 X.drop(columns=col, inplace=True)

    #         for col in ['atr', 'macd', 'macd_signal', 'macd_diff', 'body']:
    #             if col in X.columns:
    #                 new_col_pct = f'{col}_pct'
    #                 X[new_col_pct] = _apply_pct_transformation(col)
    #                 self.transformation_log.append({'transformation': 'pct', 'features': [col],
    #                                                 'denominator': 'close', 'output': new_col_pct})
    #                 X.drop(columns=col, inplace=True)

    #         # 🔹 Add SR and Pivot distance features
    #         for col in X.columns:
    #             if '_dist_to_next_' in col:
    #                 new_col_pct_dist_next = f'{col}_pct'
    #                 X[new_col_pct_dist_next] = _apply_pct_transformation(col)
    #                 self.transformation_log.append({'transformation': 'pct', 'features': [col],
    #                                                 'denominator': 'close', 'output': new_col_pct_dist_next})
    #                 X.drop(columns=col, inplace=True)
    #             # if col.startswith(('sr_', 'pivots_', 'levels_')) and not any(x in col for x in [
    #             #     '_pos_in_range', '_dist_to_next_up', '_dist_to_next_down', '_D_', '_M_']):
    #             #     X[f'{col}_dist_from_close'] = _apply_pct_diff_transformation(col, 'close')
    #             #     X.drop(columns=col, inplace=True)

    #     return X
