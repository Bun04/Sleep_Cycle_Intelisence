import math
import heapq

class KNN:
    def __init__(self, k=5):
        self.k = k
        self._X = []
        self._y = []
        self.features = []

    def _vec_from_row(self, row, features):
        vec = []
        for f in features:
            # missing values -> 0.0
            vec.append(float(row.get(f, 0.0)))
        return vec

    def fit(self, dataset, features, target_field):
        self.features = list(features)
        self._X = [self._vec_from_row(r, self.features) for r in dataset]
        self._y = [r.get(target_field) for r in dataset]

    def _distance(self, a, b):
        s = 0.0
        for x, y in zip(a, b):
            s += (x - y) ** 2
        return math.sqrt(s)

    def predict_row(self, mapped_row):
        if not self._X:
            return None
        vec = self._vec_from_row(mapped_row, self.features)
        # find k smallest distances
        heap = []
        for i, train_vec in enumerate(self._X):
            d = self._distance(vec, train_vec)
            heap.append((d, i))

        heap.sort(key=lambda x: x[0])
        k_neigh = heap[:min(self.k, len(heap))]
        # majority vote for classification; if numeric -> average
        votes = [self._y[i] for (_d, i) in k_neigh]
        # detect numeric
        try:
            nums = [float(v) for v in votes]
            return sum(nums) / len(nums)
        except Exception:
            # categorical majority
            from collections import Counter
            c = Counter(votes)
            return c.most_common(1)[0][0]

    def predict(self, rows):
        return [self.predict_row(r) for r in rows]
