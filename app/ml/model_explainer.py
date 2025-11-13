import sys, os, numpy as np, pandas as pd, shap, matplotlib.pyplot as plt
from sklearn.tree import _tree
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor, plot_tree, _tree
from sklearn.metrics import accuracy_score
# from sklearn.utils.validation import check_is_fitted

parent_folder = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
sys.path.append(parent_folder)

from utils import helpers



class ModelExplainer:
    def __init__(self, model, trained, X_train, X_raw, X_transformed, y_train, feature_names, model_subtype='classification', shap_types=None, 
                 preprocessor=None, paths=None, depth_surrogate=3, show_figures=True, figures_folder=None):
        self.model = model
        self.trained = trained
        self.X_train = X_train
        self.y_train = y_train
        self.feature_names = feature_names
        self.model_subtype = model_subtype
        self.shap_types = shap_types or ['dot', 'bar'] # 'violin'
        self.preprocessor = preprocessor
        self.paths = paths or {}
        self.show_figures = show_figures
        self.figures_folder = figures_folder
        self.shap_values = None
        self.depth_surrogate = depth_surrogate
        self.surrogate_tree = None
        self.X_transformed = X_transformed
        self.X_raw = X_raw

    def compute_shap(self, X_sample=None):
        if X_sample is None:
            X_sample = self.X_train[self.feature_names]
        explainer = shap.Explainer(self.model, X_sample)
        self.shap_values = explainer(X_sample)

        return self.shap_values, X_sample
    
    def plot_shaps(self, strategy, timeframe, target, X_sample):
        # SHAP summary plot
        title = f"SHAP summary for strategy {strategy}, timeframe {timeframe}, target {target}"
        for shap_type in self.shap_types:
            self._plot_shap(self.shap_values, X_sample, plot_type=shap_type, title=title)

    def _plot_shap(self, shap_values, X_sample, plot_type, title, max_display=20, save_plot=True):
        shap.summary_plot(shap_values, X_sample, plot_type=plot_type, max_display=max_display, show=self.show_figures)
        plt.title(title)
        plt.tight_layout()
        plot_name = 'shap_' + plot_type
        if save_plot and plot_name in self.paths:
            os.makedirs(self.figures_folder, exist_ok=True)
            plt.savefig(self.paths[plot_name])
            print(f"📊 Saved SHAP {plot_type} graph to: {self.paths[plot_name]}")
        if self.show_figures:
            plt.show()
        plt.close()

    def plot_dependence(self, feature, save_plot=True):
        shap_values, _ = self.compute_shap(X_sample=self.X_train[self.feature_names])
        title = f"SHAP dependence for feature {feature}"

        if isinstance(feature, str):
            try:
                feature_index = shap_values.feature_names.index(feature)
            except ValueError:
                raise ValueError(f"Feature '{feature}' not found in SHAP feature names.")
        else:
            feature_index = feature
            feature = shap_values.feature_names[feature_index]

        shap.dependence_plot(feature, shap_values.values, self.X_train[self.feature_names], feature_names=self.feature_names, show=self.show_figures)

        plot_name = 'dependence_plot'
        if save_plot and plot_name in self.paths:
            os.makedirs(self.figures_folder, exist_ok=True)
            save_path_base_name, save_path_ext = os.path.splitext(self.paths[plot_name])
            save_path = f"{save_path_base_name}_{feature}{save_path_ext}"
            plt.title(title)
            plt.tight_layout()
            plt.savefig(save_path)
            print(f"📊 Saved SHAP dependence plot: {save_path}")
        if self.show_figures:
            plt.show()
        plt.close()

    def train_surrogate_tree(self, random_state=42, save_plot=True):
        """Train a surrogate decision tree to approximate the trained model."""
        if not self.trained:
            raise RuntimeError("Train model before fitting surrogate tree.")

        y_pred = self.model.predict(self.X_train[self.feature_names])
        if self.model_subtype != 'regression' and y_pred.ndim > 1:
            y_pred = np.argmax(y_pred, axis=1)

        if self.model_subtype == 'regression':
            surrogate = DecisionTreeRegressor(max_depth=self.depth_surrogate, random_state=random_state)
        else:
            surrogate = DecisionTreeClassifier(max_depth=self.depth_surrogate, random_state=random_state)

        surrogate.fit(self.X_train[self.feature_names], y_pred)

        # y_pred = y_pred.astype(np.int32)
        # surrogate.fit(self.X_train[self.feature_names], y_pred, sample_weight=None)

        print(f"Surrogate approximation R²: {surrogate.score(self.X_train[self.feature_names], y_pred):.4f}")
        if self.model_subtype != 'regression':
            print(f"Surrogate vs true labels accuracy: {accuracy_score(self.y_train, surrogate.predict(self.X_train[self.feature_names])):.4f}")

        plt.figure(figsize=(14, 6))
        plot_tree(surrogate, feature_names=self.feature_names, filled=True, rounded=True, max_depth=self.depth_surrogate, fontsize=10,
            class_names=None if self.model_subtype == 'regression' else ['Class 0', 'Class 1'])
        plt.title("Surrogate Decision Tree")

        plot_name = 'surrogate_tree'
        if save_plot and plot_name in self.paths:
            os.makedirs(self.figures_folder, exist_ok=True)
            plt.savefig(self.paths[plot_name])
            print(f"📊 Saved surrogate tree to: {self.paths[plot_name]}")
        if self.show_figures: plt.show()
        plt.close()

        self.surrogate_tree = surrogate

        return surrogate

    def extract_tree_rules(self, tree=None, feature_names=None, class_names=None, decimals=4, display_rules=True):
        tree = tree or self.surrogate_tree
        feature_names = feature_names or self.feature_names
        class_names = class_names or ['Class 0', 'Class 1'] if self.model_subtype != 'regression' else None

        tree_ = tree.tree_
        feature_name = [feature_names[i] if i != _tree.TREE_UNDEFINED else 'undefined!' for i in tree_.feature]

        rules = []

        def recurse(node, depth, rule_path, rule_path_original):
            if tree_.feature[node] != _tree.TREE_UNDEFINED:
                name = feature_name[node]
                threshold = tree_.threshold[node]
                rounded_threshold = round(threshold, decimals)

                # Reverse the threshold
                reverse_result = self.preprocessor.reverse_transform_feature(
                    threshold_value=threshold, feature_name=name,
                    transformed_X=self.X_transformed, original_X=self.X_raw)

                if reverse_result:
                    reversed_val = round(reverse_result['reversed_value'], decimals)
                    orig_name = reverse_result['feature']
                    approx = reverse_result['approx_from']
                    condition_orig_l = f"{orig_name} <= {reversed_val} ({approx})"
                    condition_orig_r = f"{orig_name} > {reversed_val} ({approx})"
                else:
                    # If reverse fails, fall back to transformed feature
                    condition_orig_l = f"{name} <= {rounded_threshold} (unreversed)"
                    condition_orig_r = f"{name} > {rounded_threshold} (unreversed)"

                # Left and right child rules
                recurse(
                    tree_.children_left[node],
                    depth + 1,
                    rule_path + [f"{name} <= {rounded_threshold}"],
                    rule_path_original + [condition_orig_l]
                )
                recurse(
                    tree_.children_right[node],
                    depth + 1,
                    rule_path + [f"{name} > {rounded_threshold}"],
                    rule_path_original + [condition_orig_r]
                )
            else:
                sample_count = int(tree_.n_node_samples[node])
                value = tree_.value[node][0] * sample_count

                if class_names is not None:
                    probs = value / value.sum()
                    prediction_idx = np.argmax(probs)
                    prediction = class_names[prediction_idx]
                    prob = round(probs[prediction_idx], decimals)
                    full_prob_dist = {class_names[i]: round(p, decimals) for i, p in enumerate(probs)}
                else:
                    prediction = int(np.argmax(value))
                    prob = round(np.max(value) / np.sum(value), decimals)
                    full_prob_dist = None

                rules.append({
                    "conditions": rule_path,
                    "conditions_original": rule_path_original,
                    "prediction": prediction,
                    "probability": prob,
                    "prob_distribution": full_prob_dist,
                    "samples": sample_count
                })

        recurse(0, 1, [], [])

        # if display_rules:
        #     for i, rule in enumerate(rules):
        #         cond_str = "IF " + " AND ".join(rule["conditions"])
        #         cond_orig_str = "IF " + " AND ".join(rule["conditions_original"])
        #         prob_str = f" with probability {rule['probability']}" if rule["probability"] is not None else ""
        #         print(f"Rule {i + 1}: {cond_str} THEN Predict: {rule['prediction']}{prob_str} [Samples: {rule['samples']}]")
        #         print(f"        🔄 Original-scale: {cond_orig_str}")

        if display_rules:

            # Determine max depth of any rule (i.e., number of conditions)
            max_depth = max(len(rule["conditions"]) for rule in rules)

            rows = []
            for idx, rule in enumerate(rules, start=1):
                row = {"rule#": idx}
                for i in range(max_depth):
                    if i < len(rule["conditions"]):
                        # Parse transformed condition
                        transformed_tokens = rule["conditions"][i].split(' ', 1)
                        feature = transformed_tokens[0]
                        condition_transformed = transformed_tokens[1] if len(transformed_tokens) > 1 else np.nan

                        # Parse original condition
                        original_condition = rule["conditions_original"][i]
                        if original_condition and ' ' in original_condition:
                            original_tokens = original_condition.split(' ', 1)
                            feature_original = original_tokens[0]
                            condition_original = original_tokens[1]
                        else:
                            feature_original = np.nan
                            condition_original = np.nan
                    else:
                        feature = np.nan
                        condition_transformed = np.nan
                        feature_original = np.nan
                        condition_original = np.nan

                    row[f"feature_transf_{i+1}"] = feature
                    row[f"condition_transf_{i+1}"] = condition_transformed
                    row[f"feature_orig_{i+1}"] = feature_original
                    row[f"condition_orig_{i+1}"] = condition_original

                row["prediction"] = rule["prediction"]
                row["probability"] = rule["probability"]
                row["samples"] = rule["samples"]
                rows.append(row)

            # Create the full DataFrame
            rules_df = pd.DataFrame(rows)

            # Generate Transformed Table
            transformed_cols = ["rule#"]
            for i in range(max_depth):
                transformed_cols.extend([f"feature_transf_{i+1}", f"condition_transf_{i+1}"])
            transformed_cols.extend(["prediction", "probability", "samples"])

            # Generate Original Table
            original_cols = ["rule#"]
            for i in range(max_depth):
                original_cols.extend([f"feature_orig_{i+1}", f"condition_orig_{i+1}"])
            original_cols.extend(["prediction", "probability", "samples"])

            print("🧮 Transformed Feature Rules")
            print(helpers.df_to_table(rules_df[transformed_cols]))

            print("\n🔄 Original Feature Rules")
            print(helpers.df_to_table(rules_df[original_cols]))

        return rules

    @staticmethod
    def generate_revised_logic_code(rules, class_label="Class 1", prob_threshold=0.7, min_samples=50, indent=4):
        logic_clauses = []
        indent_str = ' ' * indent
        indent_str_2x = ' ' * 2 * indent

        for rule in rules:
            if rule["prediction"] != class_label:
                continue
            if rule["probability"] < prob_threshold:
                continue
            if rule["samples"] < min_samples:
                continue

            clause_parts = []
            for cond in rule["conditions"]:
                try:
                    cond_clean = cond.split('(')[0].strip()
                    clause_parts.append(f"row['{cond_clean.split()[0]}'] {cond_clean.split()[1]} {cond_clean.split()[2]}")
                except IndexError:
                    continue

            full_clause = " and ".join(clause_parts)
            logic_clauses.append(f"({full_clause})")

        if not logic_clauses:
            return "# No rules matched criteria"

        trigger_code = f"revised_trigger = (\n{indent_str_2x}" + f"\n{indent_str_2x}".join([f" or\n{indent_str_2x}".join(logic_clauses)]) + f"\n{indent_str})"
        return trigger_code







# def extract_tree_rules(self, tree=None, class_names=None, decimals=4, display_rules=True):
#     tree = tree or self.surrogate_tree
#     feature_names = self.feature_names
#     class_names = class_names or (['Class 0', 'Class 1'] if self.model_subtype != 'regression' else None)

#     tree_ = tree.tree_
#     feature_name = [feature_names[i] if i != _tree.TREE_UNDEFINED else 'undefined!' for i in tree_.feature]

#     rules = []

#     def recurse(node, depth, rule_path, rule_path_original):
#         if tree_.feature[node] != _tree.TREE_UNDEFINED:
#             name = feature_name[node]
#             threshold = tree_.threshold[node]
#             rounded_threshold = round(threshold, decimals)

#             reverse_result = self.preprocessor.reverse_transform_feature(
#                 threshold_value=threshold, feature_name=name,
#                 transformed_X=self.X_transformed, original_X=self.X_raw) if self.preprocessor else None

#             if reverse_result:
#                 reversed_val = round(reverse_result['reversed_value'], decimals)
#                 orig_name = reverse_result['feature']
#                 approx = reverse_result['approx_from']
#                 condition_orig_l = f"{orig_name} <= {reversed_val} ({approx})"
#                 condition_orig_r = f"{orig_name} > {reversed_val} ({approx})"
#             else:
#                 condition_orig_l = f"{name} <= {rounded_threshold} (unreversed)"
#                 condition_orig_r = f"{name} > {rounded_threshold} (unreversed)"

#             recurse(tree_.children_left[node], depth + 1, rule_path + [f"{name} <= {rounded_threshold}"], rule_path_original + [condition_orig_l])
#             recurse(tree_.children_right[node], depth + 1, rule_path + [f"{name} > {rounded_threshold}"], rule_path_original + [condition_orig_r])
#         else:
#             sample_count = int(tree_.n_node_samples[node])
#             value = tree_.value[node][0] * sample_count
#             if class_names is not None:
#                 probs = value / value.sum()
#                 prediction_idx = np.argmax(probs)
#                 prediction = class_names[prediction_idx]
#                 prob = round(probs[prediction_idx], decimals)
#                 full_prob_dist = {class_names[i]: round(p, decimals) for i, p in enumerate(probs)}
#             else:
#                 prediction = int(np.argmax(value))
#                 prob = round(np.max(value) / np.sum(value), decimals)
#                 full_prob_dist = None

#             rules.append({
#                 "conditions": rule_path,
#                 "conditions_original": rule_path_original,
#                 "prediction": prediction,
#                 "probability": prob,
#                 "prob_distribution": full_prob_dist,
#                 "samples": sample_count
#             })

#     recurse(0, 1, [], [])

#     if display_rules:
#         print(f"🧮 Extracted {len(rules)} rules.")
#     return rules
