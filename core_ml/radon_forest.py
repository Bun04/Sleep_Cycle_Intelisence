import csv
import math
import random
import pickle
from collections import Counter, defaultdict


# =========================================================
# DATA PREPROCESSOR
# =========================================================
class DataPreprocessor:
    def __init__(self):
        self.label_encoders = {}
        self.feature_names = []
        self.target_encoder = {}
        self.target_decoder = {}

    def _is_number(self, value):
        try:
            float(value)
            return True
        except:
            return False

    def _parse_blood_pressure(self, bp):
        # "120/80" -> 100.0 (trung bình)
        if isinstance(bp, str) and "/" in bp:
            try:
                s, d = bp.split("/")
                return (float(s) + float(d)) / 2
            except:
                return 0.0
        return 0.0

    def fit_transform(self, rows, target_column):
        if len(rows) == 0:
            return [], []

        self.feature_names = [k for k in rows[0].keys() if k != target_column and k != "Person ID"]

        X = []
        y = []

        # Encode target
        unique_targets = sorted(list(set(row[target_column] for row in rows)))
        for idx, label in enumerate(unique_targets):
            self.target_encoder[label] = idx
            self.target_decoder[idx] = label

        # Build encoders for categorical features
        for feature in self.feature_names:
            values = []
            for row in rows:
                value = row[feature]
                if feature == "Blood Pressure":
                    continue
                if not self._is_number(value):
                    values.append(value)

            if values:
                unique_values = sorted(list(set(values)))
                self.label_encoders[feature] = {
                    val: idx for idx, val in enumerate(unique_values)
                }

        # Transform rows
        for row in rows:
            features = []

            for feature in self.feature_names:
                value = row[feature]

                if feature == "Blood Pressure":
                    features.append(self._parse_blood_pressure(value))
                elif feature in self.label_encoders:
                    features.append(self.label_encoders[feature].get(value, 0))
                else:
                    try:
                        features.append(float(value))
                    except:
                        features.append(0.0)

            X.append(features)
            y.append(self.target_encoder[row[target_column]])

        return X, y

    def transform_one(self, row):
        features = []

        for feature in self.feature_names:
            value = row.get(feature, 0)

            if feature == "Blood Pressure":
                features.append(self._parse_blood_pressure(value))
            elif feature in self.label_encoders:
                features.append(self.label_encoders[feature].get(value, 0))
            else:
                try:
                    features.append(float(value))
                except:
                    features.append(0.0)

        return features


# =========================================================
# TREE NODE
# =========================================================
class Node:
    def __init__(self, feature=None, threshold=None,
                 left=None, right=None, value=None):
        self.feature = feature
        self.threshold = threshold
        self.left = left
        self.right = right
        self.value = value

    def is_leaf(self):
        return self.value is not None


class DecisionTree:
    def _grow_tree(self, X, y, depth):
        n_samples = len(X)
        n_labels = len(set(y))

        # Stop conditions
        if (
            depth >= self.max_depth or
            n_labels == 1 or
            n_samples < self.min_samples_split
        ):
            leaf_value = self._most_common_label(y)
            return Node(value=leaf_value)

        feature, threshold = self._best_split(X, y)

        if feature is None:
            leaf_value = self._most_common_label(y)
            return Node(value=leaf_value)

        left_X, left_y, right_X, right_y = self._split(
            X, y, feature, threshold
        )

        if len(left_X) == 0 or len(right_X) == 0:
            leaf_value = self._most_common_label(y)
            return Node(value=leaf_value)

        left_node = self._grow_tree(left_X, left_y, depth + 1)
        right_node = self._grow_tree(right_X, right_y, depth + 1)

        return Node(
            feature=feature,
            threshold=threshold,
            left=left_node,
            right=right_node
        )

    def _traverse(self, x, node):
        if node.is_leaf():
            return node.value

        if x[node.feature] <= node.threshold:
            return self._traverse(x, node.left)
        return self._traverse(x, node.right)

    def predict_one(self, x):
        return self._traverse(x, self.root)

    def predict(self, X):
        return [self.predict_one(x) for x in X]
    


        print("Start training Random Forest...")
        self.model.fit(X_train, y_train)

        print("Evaluating model...")
        y_pred = self.model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)

        print(f"Accuracy: {acc:.4f}")
        return acc

    def predict_from_form(self, form_data):
        x = self.preprocessor.transform_one(form_data)

        pred = self.model.predict([x])[0]
        probs = self.model.predict_proba_one(x)

        label = self.preprocessor.target_decoder[pred]

        decoded_probs = {}
        for cls, prob in probs.items():
            decoded_probs[
                self.preprocessor.target_decoder[cls]
            ] = prob

        return {
            "prediction": label,
            "probabilities": decoded_probs
        }

    def save_model(self, filepath):
        with open(filepath, "wb") as f:
            pickle.dump(self, f)

    @staticmethod
    def load_model(filepath):
        with open(filepath, "rb") as f:
            return pickle.load(f)


# =========================================================
# EXAMPLE USAGE
# =========================================================
if __name__ == "__main__":
    system = SleepCycleSystem()

    # Train model
    accuracy = system.train_from_csv(
        "data/Sleep_health_and_lifestyle_dataset.csv"
    )

    # Save model
    system.save_model("models/random_forest_model.pkl")

    # Predict from form
    sample_user = {
        "Gender": "Male",
        "Age": 25,
        "Occupation": "Engineer",
        "Sleep Duration": 6.0,
        "Quality of Sleep": 6,
        "Physical Activity Level": 35,
        "Stress Level": 8,
        "BMI Category": "Overweight",
        "Blood Pressure": "130/85",
        "Heart Rate": 78,
        "Daily Steps": 4000
    }

    result = system.predict_from_form(sample_user)

    print("Prediction:", result["prediction"])
    print("Probabilities:")
    for label, prob in result["probabilities"].items():
        print(f"  {label}: {prob:.2%}")