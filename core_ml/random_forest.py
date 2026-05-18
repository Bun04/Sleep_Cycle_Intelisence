import csv
import random
import math
import pickle
import numpy as np
import matplotlib.pyplot as plt


# DATA PROCESSING


def safe_int(val, default=0):

    try:
        return int(float(val))

    except:
        return default


def safe_float(val, default=0.0):

    try:
        return float(val)

    except:
        return default


def load_and_process_csv(filename):

    dataset = []

    occ_set = set()
    bmi_set = set()

    with open(filename, 'r', encoding='utf-8') as file:

        reader = csv.DictReader(file)

        for row in reader:

            try:

                age = safe_float(row['Age'])

                stress = safe_float(
                    row['Stress Level']
                )

                sleep_quality = safe_float(
                    row['Quality of Sleep']
                )

                bp = row['Blood Pressure'].split('/')

                systolic = safe_int(bp[0], 120)
                diastolic = safe_int(bp[1], 80)

                smoker = safe_int(
                    row.get('Smoker', 0)
                )

                stroke = safe_int(
                    row.get('Stroke_History', 0)
                )

                deep_sleep = safe_float(
                    row.get(
                        'Deep_Sleep_Percentage',
                        0.2
                    )
                )

                heart_label = 1 if row.get(
                    'Heart_Disease',
                    'No'
                ) == 'Yes' else 0

                # Alzheimer heuristic
                if (
                    age > 50 and
                    (
                        deep_sleep < 0.15 or
                        stress > 7 or
                        sleep_quality < 5
                    )
                ):

                    alzheimer_label = 1

                else:

                    alzheimer_label = 0

                processed = {

                    'Age': age,

                    'Occupation':
                        row['Occupation'],

                    'Sleep Duration':
                        safe_float(
                            row['Sleep Duration']
                        ),

                    'Quality of Sleep':
                        sleep_quality,

                    'Physical Activity Level':
                        safe_float(
                            row[
                                'Physical Activity Level'
                            ]
                        ),

                    'Stress Level':
                        stress,

                    'Heart Rate':
                        safe_float(
                            row['Heart Rate']
                        ),

                    'BMI':
                        row['BMI Category'],

                    'Systolic':
                        systolic,

                    'Diastolic':
                        diastolic,

                    'Smoker':
                        smoker,

                    'Stroke':
                        stroke,

                    'Deep_Sleep':
                        deep_sleep,

                    'Sleep Disorder':
                        row['Sleep Disorder'],

                    'Heart_Disease':
                        heart_label,

                    'Alzheimer_Risk':
                        alzheimer_label
                }

                dataset.append(processed)

                occ_set.add(
                    row['Occupation']
                )

                bmi_set.add(
                    row['BMI Category']
                )

            except Exception as e:

                print("ROW ERROR:", e)
                print(row)

    occ_map = {

        name: i
        for i, name in enumerate(occ_set)

    }

    bmi_map = {

        name: i
        for i, name in enumerate(bmi_set)

    }

    for row in dataset:

        row['Occ_Num'] = occ_map[
            row['Occupation']
        ]

        row['BMI_Num'] = bmi_map[
            row['BMI']
        ]

    return dataset, occ_map, bmi_map


# TRAIN TEST SPLIT


def train_test_split(
    dataset,
    test_ratio=0.2,
    seed=42
):

    random.seed(seed)

    dataset = list(dataset)

    random.shuffle(dataset)

    split = int(
        len(dataset) * (1 - test_ratio)
    )

    return dataset[:split], dataset[split:]


# TREE UTILITIES


def test_split(feature, value, dataset):

    left = []
    right = []

    for row in dataset:

        if row[feature] < value:
            left.append(row)

        else:
            right.append(row)

    return left, right


# CLASSIFICATION GINI


def calculate_gini(groups, classes, target):

    total = float(
        sum(len(g) for g in groups)
    )

    gini = 0.0

    for group in groups:

        size = float(len(group))

        if size == 0:
            continue

        score = 0.0

        for class_val in classes:

            p = [

                r[target]
                for r in group

            ].count(class_val) / size

            score += p * p

        gini += (
            1.0 - score
        ) * (size / total)

    return gini



# REGRESSION VARIANCE


def calculate_variance(groups, target):

    total_size = sum(
        len(g) for g in groups
    )

    weighted_variance = 0

    for group in groups:

        size = len(group)

        if size == 0:
            continue

        values = [
            r[target]
            for r in group
        ]

        mean = sum(values) / size

        variance = sum(

            (x - mean) ** 2
            for x in values

        ) / size

        weighted_variance += (
            variance * (size / total_size)
        )

    return weighted_variance



# BEST SPLIT


def get_best_split(
    dataset,
    features,
    target,
    mode
):

    best_feature = None
    best_value = None
    best_score = 999999
    best_groups = None

    # FIX: compute classes once outside inner loop to avoid variable shadowing
    if mode == 'classification':
        classes = list(
            set(r[target] for r in dataset)
        )

    for feature in features:

        for row in dataset:

            groups = test_split(
                feature,
                row[feature],
                dataset
            )

            if (
                len(groups[0]) == 0 or
                len(groups[1]) == 0
            ):

                continue

            if mode == 'classification':

                score = calculate_gini(
                    groups,
                    classes,
                    target
                )

            else:

                score = calculate_variance(
                    groups,
                    target
                )

            if score < best_score:

                best_feature = feature
                best_value = row[feature]
                best_score = score
                best_groups = groups

    # FIX: if no valid split found, return a terminal signal
    if best_groups is None:
        return None

    return {

        'index': best_feature,
        'value': best_value,
        'groups': best_groups

    }



# TERMINAL NODE


def to_terminal(group, target, mode):

    outcomes = [

        row[target]
        for row in group

    ]

    if mode == 'classification':

        return max(
            set(outcomes),
            key=outcomes.count
        )

    return sum(outcomes) / len(outcomes)


# SPLIT TREE


def split_node(
    node,
    max_depth,
    min_size,
    depth,
    features,
    target,
    mode
):

    left, right = node['groups']

    del(node['groups'])

    if not left or not right:

        node['left'] = node['right'] = (
            to_terminal(
                left + right,
                target,
                mode
            )
        )

        return

    if depth >= max_depth:

        node['left'] = to_terminal(
            left,
            target,
            mode
        )

        node['right'] = to_terminal(
            right,
            target,
            mode
        )

        return

    # LEFT

    if len(left) <= min_size:

        node['left'] = to_terminal(
            left,
            target,
            mode
        )

    else:

        random_features = random.sample(

            features,

            k=max(
                1,
                int(math.sqrt(len(features)))
            )
        )

        # FIX: handle case where no valid split exists for this sub-node
        left_split = get_best_split(
            left,
            random_features,
            target,
            mode
        )

        if left_split is None:

            node['left'] = to_terminal(
                left,
                target,
                mode
            )

        else:

            node['left'] = left_split

            split_node(

                node['left'],
                max_depth,
                min_size,
                depth + 1,
                features,
                target,
                mode
            )

    # RIGHT

    if len(right) <= min_size:

        node['right'] = to_terminal(
            right,
            target,
            mode
        )

    else:

        random_features = random.sample(

            features,

            k=max(
                1,
                int(math.sqrt(len(features)))
            )
        )

        #handle case where no valid split exists for this sub-node
        right_split = get_best_split(
            right,
            random_features,
            target,
            mode
        )

        if right_split is None:

            node['right'] = to_terminal(
                right,
                target,
                mode
            )

        else:

            node['right'] = right_split

            split_node(

                node['right'],
                max_depth,
                min_size,
                depth + 1,
                features,
                target,
                mode
            )



# BUILD TREE


def build_tree(
    train,
    max_depth,
    min_size,
    features,
    target,
    mode
):

    random_features = random.sample(

        features,

        k=max(
            1,
            int(math.sqrt(len(features)))
        )
    )

    root = get_best_split(

        train,
        random_features,
        target,
        mode
    )

    # FIX: if root has no valid split, return a terminal node directly
    if root is None:
        return to_terminal(train, target, mode)

    split_node(

        root,
        max_depth,
        min_size,
        1,
        features,
        target,
        mode
    )

    return root



# PREDICT TREE


def predict_tree(node, row):

    # if node is a terminal value (not a dict), return it directly
    if not isinstance(node, dict):
        return node

    #  if node dict has no 'index' key or index is None, treat as terminal
    if node.get('index') is None:
        return node.get('value')

    if row[node['index']] < node['value']:

        if isinstance(node['left'], dict):

            return predict_tree(
                node['left'],
                row
            )

        return node['left']

    else:

        if isinstance(node['right'], dict):

            return predict_tree(
                node['right'],
                row
            )

        return node['right']


# RANDOM FOREST

class RandomForest:

    def __init__(
        self,
        n_trees=10,
        max_depth=4,
        min_size=2,
        sample_size=0.8,
        mode='classification'
    ):

        self.n_trees = n_trees
        self.max_depth = max_depth
        self.min_size = min_size
        self.sample_size = sample_size
        self.mode = mode

        self.trees = []

    def subsample(self, dataset):

        sample = []

        n_sample = round(
            len(dataset) *
            self.sample_size
        )

        while len(sample) < n_sample:

            index = random.randrange(
                len(dataset)
            )

            sample.append(
                dataset[index]
            )

        return sample

    def fit(
        self,
        train,
        features,
        target
    ):

        self.features = features
        self.target = target

        self.trees = []

        for i in range(self.n_trees):

            sample = self.subsample(train)

            tree = build_tree(

                sample,

                self.max_depth,

                self.min_size,

                features,

                target,

                self.mode
            )

            self.trees.append(tree)

            print(
                f"Tree {i+1}/{self.n_trees} trained"
            )

    def predict_row(self, row):

        predictions = [

            predict_tree(tree, row)

            for tree in self.trees
        ]

        if self.mode == 'classification':

            return max(
                set(predictions),
                key=predictions.count
            )

        return sum(predictions) / len(predictions)

    def predict(self, dataset):

        return [

            self.predict_row(row)

            for row in dataset
        ]

    def predict_probability(
        self,
        row,
        positive_class=1
    ):

        votes = [

            predict_tree(tree, row)

            for tree in self.trees
        ]

        positive_votes = votes.count(
            positive_class
        )

        return positive_votes / len(votes)



# METRICS


def classification_metrics(actual, predicted):

    accuracy = sum(

        1
        for a, p in zip(actual, predicted)
        if a == p

    ) / len(actual)

    labels = sorted(set(actual))

    print("\nCONFUSION INFO")

    # Dict to collect per-label metrics for charting
    metrics_by_label = {}

    for label in labels:

        tp = sum(

            1
            for a, p in zip(actual, predicted)

            if a == label and p == label
        )

        fp = sum(

            1
            for a, p in zip(actual, predicted)

            if a != label and p == label
        )

        fn = sum(

            1
            for a, p in zip(actual, predicted)

            if a == label and p != label
        )

        tn = sum(

            1
            for a, p in zip(actual, predicted)

            if a != label and p != label
        )

        precision = (
            tp / (tp + fp)
            if (tp + fp)
            else 0
        )

        recall = (
            tp / (tp + fn)
            if (tp + fn)
            else 0
        )

        f1 = (

            2 * precision * recall /
            (precision + recall)

            if (precision + recall)

            else 0
        )

        metrics_by_label[label] = {
            'Precision': round(precision, 4),
            'Recall':    round(recall,    4),
            'F1 Score':  round(f1,        4),
        }

        print("\nLabel:", label)

        print("TP:", tp)
        print("FP:", fp)
        print("FN:", fn)
        print("TN:", tn)

        print(
            "Precision:",
            round(precision, 4)
        )

        print(
            "Recall:",
            round(recall, 4)
        )

        print(
            "F1:",
            round(f1, 4)
        )

    print(
        "\nAccuracy:",
        round(accuracy, 4)
    )

    return accuracy, metrics_by_label


# METRICS SUMMARY CHART


def plot_metrics_summary(
    metrics_by_label,
    accuracy,
    title="Metrics Summary"
):

    labels = list(metrics_by_label.keys())
    metric_names = ['Precision', 'Recall', 'F1 Score']

    n_labels = len(labels)
    n_metrics = len(metric_names)

    # Bar positions
    x = np.arange(n_labels)
    bar_width = 0.2

    fig, ax = plt.subplots(
        figsize=(max(6, n_labels * 2.5), 5)
    )

    colors = ['#4C72B0', '#55A868', '#C44E52']

    for i, metric in enumerate(metric_names):

        values = [
            metrics_by_label[label][metric]
            for label in labels
        ]

        bars = ax.bar(
            x + i * bar_width,
            values,
            width=bar_width,
            label=metric,
            color=colors[i],
            alpha=0.85,
            edgecolor='white'
        )

        # Annotate value on top of each bar
        for bar in bars:

            height = bar.get_height()

            ax.text(
                bar.get_x() + bar.get_width() / 2,
                height + 0.01,
                f'{height:.2f}',
                ha='center',
                va='bottom',
                fontsize=8
            )

    # Draw accuracy as horizontal dashed line
    ax.axhline(
        y=accuracy,
        color='orange',
        linestyle='--',
        linewidth=1.5,
        label=f'Accuracy = {accuracy:.4f}'
    )

    ax.set_xticks(x + bar_width * (n_metrics - 1) / 2)
    ax.set_xticklabels(
        [str(l) for l in labels],
        fontsize=10
    )

    ax.set_ylim(0, 1.15)
    ax.set_ylabel("Score")
    ax.set_title(title)
    ax.legend(loc='lower right')
    ax.grid(axis='y', linestyle='--', alpha=0.5)

    plt.tight_layout()
    plt.show()


def regression_metrics(actual, predicted):

    mae = sum(

        abs(a - p)

        for a, p in zip(actual, predicted)

    ) / len(actual)

    mse = sum(

        (a - p) ** 2

        for a, p in zip(actual, predicted)

    ) / len(actual)

    rmse = math.sqrt(mse)

    print("\nMAE:", round(mae, 4))
    print("MSE:", round(mse, 4))
    print("RMSE:", round(rmse, 4))

    return rmse


# ROC CURVE


def plot_roc_curve(
    actual,
    probabilities,
    positive_class=1
):

    thresholds = np.linspace(0, 1, 100)

    tpr_list = []
    fpr_list = []

    for threshold in thresholds:

        predicted = [

            1 if p >= threshold else 0

            for p in probabilities
        ]

        tp = sum(

            1

            for a, p in zip(actual, predicted)

            if a == positive_class and p == 1
        )

        tn = sum(

            1

            for a, p in zip(actual, predicted)

            if a != positive_class and p == 0
        )

        fp = sum(

            1

            for a, p in zip(actual, predicted)

            if a != positive_class and p == 1
        )

        fn = sum(

            1

            for a, p in zip(actual, predicted)

            if a == positive_class and p == 0
        )

        tpr = (
            tp / (tp + fn)
            if (tp + fn)
            else 0
        )

        fpr = (
            fp / (fp + tn)
            if (fp + tn)
            else 0
        )

        tpr_list.append(tpr)
        fpr_list.append(fpr)

    auc = np.trapezoid(
        tpr_list,
        fpr_list
    )

    plt.figure(figsize=(7, 7))

    plt.plot(

        fpr_list,
        tpr_list,

        label=f"AUC={abs(auc):.3f}"
    )

    plt.plot(
        [0, 1],
        [0, 1],
        linestyle='--'
    )

    plt.xlabel("FPR")
    plt.ylabel("TPR")

    plt.title("ROC Curve")

    plt.legend()

    plt.grid(True)

    plt.show()



# CONFUSION MATRIX


def plot_confusion_matrix(actual, predicted, title="Confusion Matrix"):

    labels = sorted(set(actual) | set(predicted))

    n = len(labels)

    # Build matrix: rows = actual, cols = predicted
    matrix = [
        [
            sum(
                1
                for a, p in zip(actual, predicted)
                if a == labels[i] and p == labels[j]
            )
            for j in range(n)
        ]
        for i in range(n)
    ]

    fig, ax = plt.subplots(figsize=(max(5, n * 1.5), max(4, n * 1.2)))

    im = ax.imshow(matrix, interpolation='nearest', cmap=plt.cm.Blues)

    plt.colorbar(im, ax=ax)

    ax.set_xticks(range(n))
    ax.set_yticks(range(n))

    ax.set_xticklabels(labels, rotation=45, ha='right')
    ax.set_yticklabels(labels)

    ax.set_xlabel("Predicted Label")
    ax.set_ylabel("Actual Label")
    ax.set_title(title)

    # Annotate each cell with count
    thresh = max(matrix[i][j] for i in range(n) for j in range(n)) / 2.0

    for i in range(n):

        for j in range(n):

            ax.text(
                j, i,
                str(matrix[i][j]),
                ha='center',
                va='center',
                color='white' if matrix[i][j] > thresh else 'black'
            )

    plt.tight_layout()

    plt.show()


#save

def save_model(model, filename):

    with open(filename, 'wb') as f:

        pickle.dump(model, f)

    print("MODEL SAVED")


def load_model(filename):

    with open(filename, 'rb') as f:

        model = pickle.load(f)

    print("MODEL LOADED")

    return model



if __name__ == "__main__":

    import os
    csv_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "Sleep_health_and_lifestyle_dataset.csv",
    )
    dataset, occ_map, bmi_map = load_and_process_csv(csv_path)

    print(
        "Dataset size:",
        len(dataset)
    )

    train_set, test_set = train_test_split(
        dataset
    )

    # CLASSIFICATION


    features_disorder = [

        'Age',
        'Occ_Num',
        'BMI_Num',
        'Systolic',
        'Diastolic',
        'Stress Level'

    ]

    rf_disorder = RandomForest(

        n_trees=10,
        max_depth=6,
        min_size=2,
        sample_size=0.8,
        mode='classification'
    )

    rf_disorder.fit(

        train_set,

        features_disorder,

        'Sleep Disorder'
    )

    actual = [

        row['Sleep Disorder']

        for row in test_set
    ]

    predicted = rf_disorder.predict(
        test_set
    )

    acc_disorder, metrics_disorder = classification_metrics(
        actual,
        predicted
    )

    plot_metrics_summary(
        metrics_disorder,
        acc_disorder,
        title="Metrics Summary - Sleep Disorder"
    )

    plot_confusion_matrix(
        actual,
        predicted,
        title="Confusion Matrix - Sleep Disorder"
    )


    # REGRESSION
 

    features_quality = [

        'Physical Activity Level',
        'Stress Level',
        'Heart Rate',
        'Age'

    ]

    rf_quality = RandomForest(

        n_trees=10,
        max_depth=6,
        min_size=2,
        sample_size=0.8,
        mode='regression'
    )

    rf_quality.fit(

        train_set,

        features_quality,

        'Quality of Sleep'
    )

    actual_reg = [

        row['Quality of Sleep']

        for row in test_set
    ]

    predicted_reg = rf_quality.predict(
        test_set
    )

    regression_metrics(
        actual_reg,
        predicted_reg
    )

    # HEART DISEASE ROC


    heart_features = [

        'Age',
        'Systolic',
        'Diastolic',
        'Heart Rate',
        'Smoker',
        'Stroke',
        'BMI_Num'
    ]

    rf_heart = RandomForest(

        n_trees=15,
        max_depth=6,
        min_size=2,
        sample_size=0.8,
        mode='classification'
    )

    rf_heart.fit(

        train_set,

        heart_features,

        'Heart_Disease'
    )

    actual_heart = [

        row['Heart_Disease']

        for row in test_set
    ]

    probs = [

        rf_heart.predict_probability(
            row,
            positive_class=1
        )

        for row in test_set
    ]

    preds = [

        1 if p >= 0.5 else 0

        for p in probs
    ]

    acc_heart, metrics_heart = classification_metrics(
        actual_heart,
        preds
    )

    plot_metrics_summary(
        metrics_heart,
        acc_heart,
        title="Metrics Summary - Heart Disease"
    )

    plot_roc_curve(
        actual_heart,
        probs,
        positive_class=1
    )

   
    # ALZHEIMER RISK
    

    features_alzheimer = [
        'Age',
        'Physical Activity Level',
        'Stress Level',
        'Heart Rate',
        'Deep_Sleep'
    ]

    rf_alzheimer = RandomForest(
        n_trees=10,
        max_depth=6,
        min_size=2,
        sample_size=0.8,
        mode='classification'
    )

    rf_alzheimer.fit(
        train_set,
        features_alzheimer,
        'Alzheimer_Risk'
    )

    actual_alzheimer = [
        row['Alzheimer_Risk']
        for row in test_set
    ]

    predicted_alzheimer = rf_alzheimer.predict(test_set)

    acc_alzheimer, metrics_alzheimer = classification_metrics(
        actual_alzheimer,
        predicted_alzheimer
    )

    plot_metrics_summary(
        metrics_alzheimer,
        acc_alzheimer,
        title="Metrics Summary - Alzheimer Risk"
    )

   #save model
    models = {

        'rf_disorder': rf_disorder,
        'rf_quality': rf_quality,
        'rf_heart': rf_heart,
        'rf_alzheimer': rf_alzheimer,

        'occ_map': occ_map,
        'bmi_map': bmi_map
    }

    save_model(
        models,
        "sleep_model.pkl"
    )