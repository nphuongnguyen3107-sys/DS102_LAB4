import numpy as np

class DecisionTreeClassifier:
    def __init__(self, max_depth=None, min_samples_split=2, min_samples_leaf=1):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.tree = None

    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y)
        self.classes_, y_enc = np.unique(y, return_inverse=True)
        self.n_classes_ = len(self.classes_)
        self.tree = self._build_tree(X, y_enc, depth=0)

    def predict(self, X):
        X = np.asarray(X, dtype=float)
        preds = np.array([self._predict_sample(x, self.tree) for x in X])
        return self.classes_[preds]

    def _predict_sample(self, x, node):
        while node["type"] != "leaf":
            if x[node["feature"]] < node["threshold"]:
                node = node["left"]
            else:
                node = node["right"]
        return node["class"]

    def _build_tree(self, X, y, depth):
        m = X.shape[0]
        counts = np.bincount(y, minlength=self.n_classes_)
        predicted = int(np.argmax(counts))
        node = {"type": "leaf", "class": predicted}

        max_d = self.max_depth if self.max_depth is not None else np.inf
        if (depth < max_d
            and m >= self.min_samples_split
            and np.count_nonzero(counts) > 1):

            best_feat, best_thresh = self._best_split(X, y)
            if best_feat is not None:
                left = X[:, best_feat] < best_thresh
                right = ~left
                if left.sum() >= self.min_samples_leaf and right.sum() >= self.min_samples_leaf:
                    node = {
                        "type": "node",
                        "feature": best_feat,
                        "threshold": best_thresh,
                        "left":  self._build_tree(X[left],  y[left],  depth + 1),
                        "right": self._build_tree(X[right], y[right], depth + 1),
                    }
        return node

    @staticmethod
    def _gini_from_counts(counts, m):
        if m == 0:
            return 0.0
        p = counts / m
        return 1.0 - np.sum(p * p)

    def _best_split(self, X, y):
        m, n = X.shape
        if m <= 1:
            return None, None

        best_gini = float("inf")
        best_feat, best_thresh = None, None
        total_counts = np.bincount(y, minlength=self.n_classes_)

        for feat in range(n):
            order = np.argsort(X[:, feat], kind="mergesort")
            xs = X[order, feat]
            ys = y[order]

            left_counts  = np.zeros(self.n_classes_, dtype=int)
            right_counts = total_counts.copy()

            for i in range(1, m):
                c = ys[i - 1]
                left_counts[c]  += 1
                right_counts[c] -= 1

                if i < self.min_samples_leaf or (m - i) < self.min_samples_leaf:
                    continue
                if xs[i] == xs[i - 1]:
                    continue

                gL = self._gini_from_counts(left_counts, i)
                gR = self._gini_from_counts(right_counts, m - i)
                gini = (i * gL + (m - i) * gR) / m

                if gini < best_gini:
                    best_gini = gini
                    best_feat = feat
                    best_thresh = (xs[i] + xs[i - 1]) / 2.0

        return best_feat, best_thresh
