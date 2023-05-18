import numpy as np
from collections import Counter

class Node:
    def __init__(self, feature=None, threshold=None, left=None, right=None, *, value=None):
        self.feature = feature # best feature to split to left & right nodes
        self.threshold = threshold # thrshold to split to left & right nodes
        self.left = left # left child node
        self.right = right # right child node
        self.value = value # representative label

    def is_leaf_node(self):
        return self.value is not None

class DecisionTree:
    def __init__(self, min_samples_split = 2, max_depth=100, n_features=None):
        self.min_samples_split=min_samples_split # mininum number of samples to split
        self.max_depth=max_depth # max number of depth to split
        self.n_features = n_features # num of features of this tree(mostly the )
        self.root = None
        
    def fit(self, X, y):

        # if the number of features is None, use the number of features of the input. If not None, use the given num of features
        self.n_features = X.shape[1] if not self.n_features else min(X.shape[1], self.n_features)
        
        # _grow_tree returns the current Node, and the initial node is the root.
        self.root = self._grow_tree(X,y)

    def _grow_tree(self, X, y, depth=0):

        # number samples and input features
        n_samples, n_feats = X.shape

        # this assumes that the output has a finite number of classes.
        n_labels = len(np.unique(y))
        
        # check the stopping criteria
        # if met, create a leaf node, where the node has the value (the representative label)
        # leaf node if 1)the depth meat the max limit, 2) single class, or 3) the number of samples are small
        if(depth>=self.max_depth or n_labels == 1 or n_samples<self.min_samples_split):
            leaf_value = self._most_common_label(y)
            return Node(value=leaf_value)

        # create a vector of size (self.n_feasures) and each element is in the range of [0, n_feats]. all random numbers of unique(replace=False)
        feat_idxs = np.random.choice(n_feats, self.n_features, replace=False)

        # find the best split feature index and the threshold for the feature
        best_feature_idx, best_thresh = self._best_split(X, y, feat_idxs)

        # create child nodes
        left_idxs, right_idxs = self._split(X[:, best_feature_idx], best_thresh)
        left = self._grow_tree(X[left_idxs, :], y[left_idxs], depth+1)
        right = self._grow_tree(X[right_idxs, :], y[right_idxs], depth+1)
        
        return Node(best_feature_idx, best_thresh, left, right)
 
    def _best_split(self, X, y, feat_idxs):
        best_gain = -1
        split_feature_idx, split_threshould = None, None

        for feat_idx in feat_idxs:

            # input vector of feat_idx
            X_column = X[:, feat_idx]
            
            # this assumes the feature values are finite classes. it probably still works for real numbers, but
            # it may take a long time if the input data is really long. Also, the threshold will be always one in the input data set.
            thresholds = np.unique(X_column) 

            # try all thresholds, and find the best feature index and the threshold
            # that produces the most difference in the information reduction, i.e. the largest information reduction from parent to child.
            # this is like finding the best split line into two classes within the feature space.
            for thr in thresholds:
                # calculate teh information gain
                gain = self._information_gain(y, X_column, thr)
                if gain>best_gain:
                    best_gain = gain
                    split_feature_idx = feat_idx
                    split_threshould = thr
        
        return split_feature_idx, split_threshould

    def _information_gain(self, y, X_column, threshold):
        #parent entropy
        parent_entropy = self._entropy(y)

        #create children
        left_idxs, right_idxs = self._split(X_column, threshold)

        if len(left_idxs) == 0 or len(right_idxs) ==0:
            return 0
        

        #calculate the weighted avg entropy of children
        n = len(y)
        n_l, n_r = len(left_idxs), len(right_idxs)
        e_l, e_r = self._entropy(y[left_idxs]), self._entropy(y[right_idxs])
        child_entropy = (n_l/n)*e_l + (n_r/n)*e_r
        
        # calculate IG
        information_gain = parent_entropy - child_entropy

        return information_gain

    def _split(self, X_column, threshold):
        left_idxs = np.argwhere(X_column <= threshold).flatten()
        right_idxs = np.argwhere(X_column > threshold).flatten()
        return left_idxs, right_idxs


    def _entropy(self,y):
        hist = np.bincount(y)
        ps = hist/len(y)
        return -np.sum([p * np.log(p) for p in ps if p>0])

    def _most_common_label(self, y):
        counter = Counter(y)
        value = counter.most_common(1)[0][0]
        return value
        

    def predict(self, X):
        return np.array([self._traverse_tree(x, self.root) for x in X])
    
    def _traverse_tree(self, x, node):
        if node.is_leaf_node():
            return node.value
        
        if x[node.feature] <= node.threshold:
            return self._traverse_tree(x, node.left)
        else: 
            return self._traverse_tree(x, node.right)

        



    