import numpy as np
from collections import Counter
from .decision_tree import DecisionTreeClassifier

class RandomForestClassifier:
    def __init__(
        self,
        n_estimators=100,
        max_depth=None,
        min_samples_split=2,
        min_samples_leaf=1,
        max_features='sqrt',
        random_state = None
    ):
        self.random_state = random_state
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.max_features = max_features
        self.trees = []
        self.tree_features = [] 
    
    def bootstrap(self, X, y):
        n_samples = X.shape[0]
        indices = np.random.choice(n_samples, size=n_samples, replace=True)
        return X[indices], y[indices]
    
    def fit(self, X, y):
        if self.random_state is not None:
            np.random.seed(self.random_state)
        self.trees = []
        self.tree_features = []
        
        n_features_rf = X.shape[1]
        if self.max_features == 'sqrt':
            n_features_rf = int(np.sqrt(X.shape[1]))
        elif isinstance(self.max_features, int):
            n_features_rf = self.max_features

        for _ in range(self.n_estimators):
            # 1. Bốc thăm dòng dữ liệu
            X_bootstrap, y_bootstrap = self.bootstrap(X, y)
            
            # 2. Bốc thăm cột ngẫu nhiên CHO RIÊNG CÂY NÀY
            features_idx = np.random.choice(X.shape[1], n_features_rf, replace=False)
            self.tree_features.append(features_idx)
            
            # 3. Lọc lại data, ép cây chỉ học những cột đã bốc thăm
            X_subset = X_bootstrap[:, features_idx]
            
            tree = DecisionTreeClassifier(
                max_depth=self.max_depth,
                min_samples_split=self.min_samples_split,
                min_samples_leaf=self.min_samples_leaf
            )
            tree.fit(X_subset, y_bootstrap)
            self.trees.append(tree)
    
    def _most_common_label(self, y):
        counter = Counter(y)
        return counter.most_common(1)[0][0]
       
    def predict(self, X):
        tree_preds = []
        
        # 4. Dự đoán: Cây nào lúc train học cột nào thì lúc test phải mớm cho nó đúng mấy cột đó
        for tree, features_idx in zip(self.trees, self.tree_features):
            X_subset = X[:, features_idx]
            pred = tree.predict(X_subset)
            tree_preds.append(pred)
            
        tree_preds = np.array(tree_preds)
        tree_preds = np.swapaxes(tree_preds, 0, 1)
        
        return np.array([self._most_common_label(pred) for pred in tree_preds])