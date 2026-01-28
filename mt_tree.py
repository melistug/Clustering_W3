# Based on the work by baydoganm/mtTrees

import os
import pandas as pd
import numpy as np
import random
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
from matplotlib import pyplot as plt

from dtaidistance import dtw


from scipy.stats import rankdata
from sklearn.decomposition import PCA
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split
from sklearn import preprocessing

class Node:
    """
    Node class for a Decision Tree.

    Attributes:
        right (Node): Right child node.
        left (Node): Left child node.
        column (int): Index of the feature used for splitting.
        column_name (str): Name of the feature.
        threshold (float): Threshold for the feature split.
        id (int): Identifier for the node.
        depth (int): Depth of the node in the tree.
        is_terminal (bool): Indicates if the node is a terminal node.
        prediction (numpy.ndarray): Predicted values for the node.
        count (int): Number of samples in the node.

    Methods:
        No specific methods are defined in this class.

        It will be mainly used in the construction of the tree.
    """
    def __init__(self):
        self.right = None
        self.left = None
        self.column = None
        self.column_name = None
        self.threshold = None
        self.id = None
        self.depth = None
        self.is_terminal = False
        self.prediction = None
        self.count = None
def find_medoid_given_distances(distances_all, instances):
    leaf_distances = distances_all.loc[
        distances_all.index.isin(instances),
        distances_all.columns.isin(instances)
    ]
    return leaf_distances.index[np.argmin(leaf_distances.mean())]

def std_agg(cnt, s1, s2):
    """
    Calculate the standard deviation of aggregated values.

    Args:
        cnt (int): Number of values.
        s1 (float): Sum of values.
        s2 (float): Sum of squared values.

    Returns:
        float: Standard deviation of the aggregated values.
    """

    return np.sqrt((s2/cnt) - (s1/cnt)**2)

def find_dist_to_medoids(yi):
    """
    Find the index of medoid and the sum of other instances to medoid in a matrix.

    Args:
        yi (numpy.ndarray): Matrix containing data.

    Returns:
        tuple: Index of the medoid, sum of values for the medoid.
    """

    yi_idx = np.argmin(yi.mean(axis=1))

    return yi_idx, sum(yi[yi_idx, :])

def get_dist_of_sample_to_leaf_instances(y_test_df_sample, y_train_df):
    """
    Calculate the distance of a sample to leaf instances.

    Args:
        y_test_df_sample (pandas.DataFrame): DataFrame containing the sample.
        y_train_df (pandas.DataFrame): DataFrame containing training output data.

    Returns:
        pandas.Series: Minimum, mean, and maximum distance to leaf instances.
    """
    sample_leaf_id = y_test_df_sample[['leaf_id']].values[0]
    y_train_df_leaf_instances = y_train_df[y_train_df['leaf_id'] == sample_leaf_id].drop(columns=['instance_id', 'leaf_id'])
    sample_leaf_dists = (y_train_df_leaf_instances - y_test_df_sample[range(201)]) ** 2
    sample_leaf_dists_agg = sample_leaf_dists.sum(axis=1)
    sample_leaf_dists_agg_res = sample_leaf_dists_agg.agg(['min', 'mean', 'max'])

    return sample_leaf_dists_agg_res

def get_dtw_dist_of_sample_to_leaf_instances(y_test_df_sample, dtw_distance):
    """
    Calculate DTW distances of a sample to leaf instances.

    Args:
        y_test_df_sample (pandas.DataFrame): DataFrame containing the sample.
        dtw_distance (numpy.ndarray): DTW distances.

    Returns:
        list: List of DTW distances to leaf instances.
    """
    dtw_dists = [dtw_distance[y_test_df_sample['instance_id'], train_idx] for train_idx in y_test_df_sample['instances']]

    return dtw_dists

def find_mean_dtw_dists_in_leaf(dtw_dist_df, instances):
    """
    Calculate the mean DTW distances in a leaf node.

    Args:
        dtw_dist_df (pandas.DataFrame): DataFrame of DTW distances.
        instances (list): List of instance IDs.

    Returns:
        float: Mean DTW distance in the leaf node.
    """
    dtw_dist_vals = dtw_dist_df[dtw_dist_df.index.isin(instances)][instances].values.ravel()

    return dtw_dist_vals[dtw_dist_vals != 0].mean()

class TreeForecast:
    """
    Predictive Clustering Tree.

    Args:
        target_type (str): Type of target variable ('multi', 'single', 'pca', etc.).
        max_features (int): Maximum number of features to consider.
        max_target (int): Maximum number of target variables.
        max_depth (int): Maximum depth of the tree.
        min_samples_leaf (int): Minimum number of samples in a leaf node.
        min_samples_split (int): Minimum number of samples to split a node.
        split_style (str): Splitting style.
        target_diff (bool): Whether to use target differences.
        lambda_decay (float): Lambda decay parameter.
        obj_weights (numpy.ndarray): Object weights.
        verbose (bool): Whether to print verbose information.

    Attributes:
        max_features (int): Maximum number of features to consider.
        max_target (int): Maximum number of target variables.
        max_depth (int): Maximum depth of the tree.
        min_samples_leaf (int): Minimum number of samples in a leaf node.
        min_samples_split (int): Minimum number of samples to split a node.
        target_type (str): Type of target variable ('multi', 'single', 'pca', etc.).
        split_style (str): Splitting style (e.g. 'custom')
        target_diff (bool): Whether to use target differences.
        lambda_decay (float): Lambda decay parameter.
        is_weighted (bool): Indicates if weights are used.
        obj_weights (numpy.ndarray): Object weights.
        verbose (bool): Whether to print verbose information.
        Tree (Node): Root node of the predictive clustering tree.

    Methods:
        buildDT(features, labels, node):
            Build the predictive clustering tree.

        fit(features, labels):
            Fit the predictive clustering tree to the data.

        nodePredictions(y):
            Calculate predictions for a node.

        selectTarget(target_type, labels):
            Select the target variable.

        applySample(features, depth, node):
            Passes one object through the decision tree and returns the prediction.

        get_rule(features, depth, node, var_list = []):
            Passes one object through the decision tree and returns the prediction rules.

        apply(features, depth):
            Returns the node id for each X.

        splits(node, split_list = []):
            Returns a list of node splits.

        printRule(features, depth, node, leaf_info_list):
            Passes one object through the decision tree and returns the probability of it belonging to each class.

        get_rules_for_selection(features, depth, node, rules):
            Returns the decision rules for feature selection.

        calcBestSplit(features, labels, current_label):
            Calculates the best split based on features and labels.

        calcBestSplitCustom(features, labels):
            Calculates the best custom split for features and labels.

        find_purity_of_leaves_for_sampling(X_train, X_test, dtw_distance_train_df):
            Calculates the purity of leaves for sampling.

        get_test_dtw_performance(y_train_df, y_test_df, dtw_distance, aggregate=True):
            Retrieves test DTW performance based on train and test data.
    """
    def __init__(self,
                 target_type = 'multi',
                 max_features = None,
                 max_target = 1,
                 max_depth = 5,
                 min_samples_leaf = 1,
                 min_samples_split = 2,
                 split_style = None,
                 target_diff = False,
                 lambda_decay = None,
                 obj_weights = None,
                 verbose = False
                 ):
        self.max_features = max_features
        self.max_target = max_target
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.min_samples_split = min_samples_split
        self.target_type = target_type
        self.split_style = split_style
        self.target_diff = target_diff
        self.lambda_decay = lambda_decay
        self.is_weighted = False
        if lambda_decay is not None:
            self.is_weighted = True
        self.obj_weights = obj_weights
        self.verbose = verbose
        self.Tree = None

    def buildDT(self, features, labels, node):
        """
        Build the predictive clustering tree.

        Args:
            features (pandas.DataFrame): The input features used for building the tree.
            labels (pandas.DataFrame): The labels or target variables corresponding to the features.
            node (Node): The current node in the tree being built.
        """
        node.prediction = self.nodePredictions(labels)
        node.count = labels.shape[0]
        if node.depth >= self.max_depth:
            node.is_terminal = True
            return

        if features.shape[0] < self.min_samples_split:
            node.is_terminal = True
            return

        if self.target_type == "multi":
            current_label = range(labels.shape[1])
            target = labels
        elif self.target_type == "single":
            current_label = 0
            target = labels
        else:
            current_label = self.selectTarget(self.target_type, labels)

            if self.target_type == "pca":
                target = pd.DataFrame(current_label)
                current_label = 0
            elif self.target_type == "pca-random":
                target = pd.DataFrame(current_label)
                current_label = random.randint(0, target.shape[1]-1)
            elif self.target_type == "mean":
                target = pd.DataFrame(current_label)
                current_label = 0
            else:
                target = labels

        (split_info, split_gain, n_cuts) = self.calcBestSplitCustom(features, target)

        split_info = split_info[~np.isnan(split_gain).any(axis=1),:]
        split_gain = split_gain[~np.isnan(split_gain).any(axis=1),:]

        split_info = split_info[~np.isinf(split_gain).any(axis=1), :]
        split_gain = split_gain[~np.isinf(split_gain).any(axis=1), :]
        # print(n_cuts)
        if n_cuts == 0:
            node.is_terminal = True
            return

        min_max_scaler = preprocessing.MinMaxScaler()
        split_gain_scaled = min_max_scaler.fit_transform(split_gain) + split_gain.min(axis=0)
        # split_gain_scaled_df = pd.DataFrame(split_gain_scaled)
        split_gain_scaled_total = np.dot(split_gain_scaled, self.obj_weights)
        mean_rank_sort = np.argsort(split_gain_scaled_total)

        # ranked_gain = rankdata(split_gain_scaled_total, method='average', axis = 0)
        # if len(ranked_gain.shape) > 1:
        #     mean_rank_sort = np.argsort(ranked_gain.mean(axis=1))
        # else:
        #     mean_rank_sort = np.argsort(ranked_gain)

        # print(split_info[mean_rank_sort[0],:])
        # if self.verbose is True:
        #     print(ranked_gain[mean_rank_sort[0]])

        splitCol = int(split_info[mean_rank_sort[0],0])
        thresh = split_info[mean_rank_sort[0],1]
        split_feature = features.columns[splitCol]

        medoid_parent, intra_cluster_dist_parent = find_dist_to_medoids(labels.to_numpy())
        self.feature_importances[split_feature] += intra_cluster_dist_parent - split_gain_scaled[mean_rank_sort[0], 0]

        node.column = splitCol
        node.column_name = features.columns[splitCol]
        node.threshold = thresh

        labels_left = labels.loc[features.iloc[:,splitCol] <= thresh, :]
        labels_left = labels_left.loc[:, features.iloc[:, splitCol] <= thresh]
        labels_right = labels.loc[features.iloc[:,splitCol] > thresh, :]
        labels_right = labels_right.loc[:, features.iloc[:, splitCol] > thresh]

        features_left = features.loc[features.iloc[:,splitCol] <= thresh]
        features_right = features.loc[features.iloc[:,splitCol] > thresh]

        # creating left and right child nodes
        node.left = Node()
        node.left.depth = node.depth + 1
        node.left.id = 2 * node.id

        node.right = Node()
        node.right.depth = node.depth + 1
        node.right.id = 2 * node.id + 1

        # splitting recursively
        self.buildDT(features_left, labels_left, node.left)
        self.buildDT(features_right, labels_right, node.right)


    def fit(self, features, labels):
        """
        Fit the predictive clustering tree to the data.

        Args:
            features (pandas.DataFrame): The input features used for building the tree.
            labels (pandas.DataFrame): The labels or target variables corresponding to the features.
        """
        self.Tree = Node()
        self.feature_importances = {col: 0 for col in features.columns}
        self.Tree.depth = 0
        self.Tree.id = 1
        self.buildDT(features, labels, self.Tree)
        self.feature_importances_df = pd.DataFrame(self.feature_importances, index=[0])

    def nodePredictions(self, y):
        """
        Calculate predictions for a node as the mean.

        Args:
            y (numpy.ndarray): The labels or target variables for a node.

        Returns:
            predictions (numpy.ndarray): Predictions for the node, which represent the mean of target variables.
        """
        predictions = np.asarray(y.mean(axis=0))

        return predictions

    def selectTarget(self, target_type, labels):
        """
        Select the target variable.

        Args:
            target_type (str): Type of target variable ('random', 'max_var', 'max_cor', 'max_inv_cor', 'pca', 'pca-random', 'mean').
            labels (pandas.DataFrame): The labels or target variables corresponding to the features.

        Returns:
            int or numpy.ndarray: The selected target variable or index.
        """
        if target_type == 'random':
            return(random.randint(0, labels.shape[1]-1))
        elif target_type == 'max_var':
            target_var = labels.var(axis=0)
            max_index = np.where(target_var.index == target_var.idxmax())
            return max_index[0]
        elif target_type == 'max_cor':
            cor_mat = labels.corr()
            avg = cor_mat.mean(axis=1)
            max_index = np.where(avg.index == avg.idxmax())
            return max_index[0]
        elif target_type == 'max_inv_cor':
            cor_mat = labels.corr().to_numpy()
            inv_cor = np.abs(np.linalg.inv(cor_mat))
            avg = inv_cor.mean(axis=1)
            max_index = np.where(avg == max(avg))
            return max_index[0]
        elif target_type == 'pca':
            pca = PCA(n_components=1).fit_transform(labels)
            return pca
        elif target_type == 'pca-random':
            if labels.shape[1] < labels.shape[0]:
                pca = PCA(n_components=labels.shape[1]).fit_transform(labels)
            else:
                pca = PCA(n_components=labels.shape[0]).fit_transform(labels)
            return pca
        elif target_type == 'mean':
            return labels.mean(axis=1)


    def calculate_inertia(self, features, labels, depth):
        train_ids = self.apply(features, depth)
        y_train_df = pd.DataFrame.from_dict({'instance_id': features.index.values, 'leaf_id': train_ids})
        y_train_leaves = y_train_df.groupby(['leaf_id'])['instance_id'].apply(
            list).reset_index().rename(columns={'instance_id': 'instances'})
        y_train_leaves['medoid'] = y_train_leaves.apply(lambda x: find_medoid_given_distances(
            labels, x['instances']), axis=1)
        self.inertia_ = y_train_leaves.apply(lambda x: labels.loc[x['medoid']][x['instances']].sum(), axis=1).sum()

        return self.inertia_

    def applySample(self, features, depth, node):
        """
        Passes one object through the predictive clustering tree and returns the leaf ID.

        Args:
            features (pandas.Series): The input features for a single object.
            depth (int): The depth at which to stop traversing the tree.
            node (Node): The current node in the tree being traversed.

        Returns:
            predicted (int): The predicted node ID.
        """

        # if we have reached the terminal node of the tree
        if node.is_terminal:
            return node.id

        # if we have reached the provided depth
        if node.depth == depth:
            return node.id

        if features[node.column] > node.threshold:
            predicted = self.applySample(features, depth, node.right)
        else:
            predicted = self.applySample(features, depth, node.left)

        return predicted

    def get_rule(self, features, depth, node, var_list = []):
        """
        Passes one object through the predictive clustering tree and returns the prediction rules.

        Args:
            features (pandas.Series): The input features for a single object.
            depth (int): The depth at which to stop traversing the tree.
            node (Node): The current node in the tree being traversed.
            var_list (list): A list to store the prediction rules.

        Returns:
            var_list (list): The list of prediction rules.
        """

        # if we have reached the terminal node of the tree
        if node.is_terminal:
            return var_list

        var_list.append(node.column_name)

        # if we have reached the provided depth
        if node.depth == depth:
            return var_list

        if features[node.column] > node.threshold:
            var_list = self.get_rule(features, depth, node.right, var_list)
        else:
            var_list = self.get_rule(features, depth, node.left, var_list)

        return var_list

    def apply(self, features, depth):
        """
        Returns the node ID for each input object.

        Args:
            features (pandas.DataFrame): The input features for multiple objects.
            depth (int): The depth at which to stop traversing the tree.

        Returns:
            predicted_ids (numpy.ndarray): The predicted node IDs for each input object.
        """
        predicted_ids = [self.applySample(features.loc[i], depth, self.Tree) for i in features.index]
        predicted_ids = np.asarray(predicted_ids)
        return predicted_ids

    def splits(self, node, split_list = []):
        """
        Returns a list of node splits.

        Args:
            node (Node): The root node of the tree.
            split_list (list): A list to store the node splits.

        Returns:
            split_rules (numpy.ndarray): An array containing node splits and related information.
        """
        # if we have reached the terminal node of the tree
        if node.is_terminal is False:
            #print([node.id, node.depth, node.column])
            #print(node.column_name)
            split_list.append([node.id, node.depth, node.column, node.threshold])
        else:
            #print(split_list)
            return split_list

        self.splits(node.left, split_list)
        self.splits(node.right, split_list)

        split_rules = np.array(split_list)

        return split_rules

    def printRule(self, features, depth, node, leaf_info_list):
        """
        Passes one object through the decision tree and returns the rules of assigning to the leaf node.

        Args:
            features (pandas.DataFrame): The input features for a single object.
            depth (int): The depth at which to stop traversing the tree.
            node (Node): The current node in the tree being traversed.
            leaf_info_list (list): A list to store the leaf information.

        Returns:
            leaf_info_list (list): The list of leaf rules.
        """
        # if we have reached the terminal node of the tree
        if node.is_terminal:
            msg = f'Ended at terminal node with ID: {node.id}'
            print(msg)
            leaf_info_list.append(msg)
            return leaf_info_list

        # if we have reached the provided depth
        if node.depth == depth:
            msg = f'Ended at depth' + str(node.depth)
            print(msg)
            leaf_info_list.append(msg)
            return leaf_info_list

        if features.iloc[:,node.column].values[0] > node.threshold:
            msg = f'Going right: Node ID: {node.id}, Rule: {features.columns[node.column]} > {node.threshold}'
            print(msg)
            leaf_info_list.append(msg)
            leaf_info_list = self.printRule(features, depth, node.right, leaf_info_list)
        else:
            msg = f'Going left: Node ID: {node.id}, Rule: {features.columns[node.column]} <= {node.threshold}'
            print(msg)
            leaf_info_list.append(msg)
            leaf_info_list = self.printRule(features, depth, node.left, leaf_info_list)

        return leaf_info_list

    def get_rules_for_selection(self, features, depth, node, rules, verbose=False):
        """
        Returns the decision rules for leaf node assignment.

        Args:
            features (pandas.Series): The input features for a single object.
            depth (int): The depth at which to stop traversing the tree.
            node (Node): The current node in the tree being traversed.
            rules (list): A list to store the decision rules.

        Returns:
            rules (list): The updated list of decision rules.
        """
        # if we have reached the terminal node of the tree
        if node.is_terminal:
            msg = f'Ended at terminal node with ID: {node.id}'
            if verbose:
                print(msg)
            return rules

        # if we have reached the provided depth
        if node.depth == depth:
            msg = f'Ended at depth' + str(node.depth)
            if verbose:
                print(msg)
            return rules

        if features.iloc[:,node.column].values[0] > node.threshold:
            msg = f'Going right: Node ID: {node.id}, Rule: {features.columns[node.column]} > {node.threshold}'
            if verbose:
                print(msg)
            rules.append({features.columns[node.column]: {'min': node.threshold}})
            rules = self.get_rules_for_selection(features, depth, node.right, rules)
        else:
            msg = f'Going left: Node ID: {node.id}, Rule: {features.columns[node.column]} <= {node.threshold}'
            if verbose:
                print(msg)
            rules.append({features.columns[node.column]: {'max': node.threshold}})
            rules = self.get_rules_for_selection(features, depth, node.left, rules)

        return rules


    def calcBestSplit(self, features, labels, current_label):
        """
        Calculates the best split based on features and labels.

        Args:
            features (pandas.DataFrame): The input features for a single object.
            labels (pandas.DataFrame): The labels or target variables corresponding to the features.
            current_label (int): The index of the current target variable.

        Returns:
            split_col (int): The column index for the best split.
            threshold (float): The threshold for the best split.
        """
        bdc = DecisionTreeRegressor(
            random_state=0,
            criterion="squared_error",
            max_features=self.max_features,
            max_depth=1,
            min_samples_split=self.min_samples_split,
            min_samples_leaf=self.min_samples_leaf
        )
        bdc.fit(features, labels.iloc[:, current_label])

        threshold = bdc.tree_.threshold[0]
        split_col = bdc.tree_.feature[0]

        return split_col, threshold

    def calcBestSplitCustom(self, features, labels):
        """
        Calculates the best custom split for features and labels.

        This is where we mainly apply the DTW-based ruling.

        We use the weights and values of the sub-objectives to obtain the best split.

        Args:
            features (pandas.DataFrame): The input features used for building the tree.
            labels (pandas.DataFrame): The labels or target variables corresponding to the features.

        Returns:
            split_info (numpy.ndarray): Information about the best split.
            split_gain (numpy.ndarray): Information about the gain from the best split.
            n_cuts (int): The number of possible splits.
        """
        # python style cart split - not efficient - multiple targets
        n = features.shape[0]
        dist_mat = labels.copy()
        current_score = float('inf')
        cut_id = 0
        n_obj = 3
        split_perf = np.zeros((n * features.shape[1], n_obj))
        split_info = np.zeros((n * features.shape[1], 2))
        for k in range(features.shape[1]):
            if self.verbose:
                print(f'Feature Index: {k}')
            x = features.iloc[:, k].to_numpy()
            y = dist_mat.to_numpy()
            sort_idx = np.argsort(x)
            sort_x = x[sort_idx]
            sort_y = y[sort_idx, :]
            sort_y = sort_y[:, sort_idx]

            for i in range(self.min_samples_leaf, n - self.min_samples_leaf - 1):
                xi = sort_x[i]
                left_yi = sort_y[:i, :]
                left_yi = left_yi[:, :i]
                right_yi = sort_y[i:, :]
                right_yi = right_yi[:, i:]

                between_yi = sort_y[:i, :]
                between_yi = between_yi[:, i:]
                avg_dist_btw_separated_instances = between_yi.mean()

                left_medoid_idx, left_perf = find_dist_to_medoids(left_yi)
                right_medoid_idx, right_perf = find_dist_to_medoids(right_yi)
                dist_btw_medoids = sort_y[left_medoid_idx, left_yi.shape[0] + right_medoid_idx]

                curr_score = left_perf + right_perf
                split_perf[cut_id, 0] = curr_score
                split_perf[cut_id, 1] = 1/dist_btw_medoids
                split_perf[cut_id, 2] = 1/avg_dist_btw_separated_instances
                split_info[cut_id, 0] = k;
                split_info[cut_id, 1] = xi;

                if i < self.min_samples_leaf or xi == sort_x[i + 1]:
                    continue

                cut_id += 1

        split_info = split_info[range(cut_id), :]
        split_gain = split_perf[range(cut_id), :]
        n_cuts = cut_id

        return split_info, split_gain, n_cuts

    def find_purity_of_leaves_for_sampling(self, X_train, X_test, dtw_distance_train_df):
        """
        Calculates the purity of leaves for sampling.

        Args:
            X_train (pandas.DataFrame): The training data features.
            X_test (pandas.DataFrame): The test data features.
            dtw_distance_train_df (pandas.DataFrame): DTW distances for training data.

        Returns:
            test_leaves_df (pandas.DataFrame): Purity information for test samples in leaves.
        """
        train_ids = self.apply(X_train, self.max_depth)
        y_train_df = pd.DataFrame.from_dict({'instance_id': X_train.index.values,
         'leaf_id': train_ids})

        y_train_leaves = y_train_df.groupby(['leaf_id'])['instance_id'].apply(
            list).reset_index().rename(columns={'instance_id': 'instances'})

        y_train_leaves['mean_dtw'] = y_train_leaves.apply(
            lambda x: find_mean_dtw_dists_in_leaf(dtw_distance_train_df, x['instances']), axis=1)
        test_ids = self.apply(X_test, self.max_depth)
        test_leaves_df = pd.DataFrame.from_dict({'sample_id': X_test.index.values,
                                             'leaf_id': test_ids})
        test_leaves_df = pd.merge(test_leaves_df, y_train_leaves[['leaf_id', 'mean_dtw']],
                                  on='leaf_id', how='left').rename(columns={
                                'mean_dtw': 'sum_var'}).drop(columns=['leaf_id'])

        return test_leaves_df

    @staticmethod
    def get_test_dtw_performance(y_train_df, y_test_df, dtw_distance, aggregate=True):
        """
        Calculate Dynamic Time Warping (DTW) clustering performance for test samples.

        Args:
            y_train_df (pd.DataFrame): DataFrame containing training data with 'instance_id' and 'leaf_id' columns.
            y_test_df (pd.DataFrame): DataFrame containing test data with 'instance_id' and 'leaf_id' columns.
            dtw_distance (numpy.ndarray): DTW distance matrix between training and test samples.
            aggregate (bool, optional): If True, aggregate DTW performance metrics for all test samples.
                If False, return metrics individually for each sample. Defaults to True.

        Returns:
            y_test_leaves (dict or pd.DataFrame): If 'aggregate' is True, returns a dictionary with aggregated DTW performance metrics
            including 'min', 'avg', and 'max'. If 'aggregate' is False, returns a DataFrame with metrics for each test sample.
        """
        y_train_leaves = y_train_df.groupby(['leaf_id'])['instance_id'].apply(
            list).reset_index().rename(columns={'instance_id': 'instances'})
        y_test_leaves = pd.merge(y_test_df[['instance_id', 'leaf_id']],
                                 y_train_leaves, on=['leaf_id'])
        y_test_leaves['dtw_distances'] = y_test_leaves.apply(
            lambda x: get_dtw_dist_of_sample_to_leaf_instances(x, dtw_distance), axis=1)
        y_test_leaves = pd.DataFrame(y_test_leaves['dtw_distances'].map(
            lambda x: [min(x), np.mean(x), max(x)]).to_list(),
                     columns=['min', 'avg', 'max'])
        if aggregate:
            y_test_leaves = y_test_leaves.mean().to_dict()

        return y_test_leaves

if __name__ == '__main__':
    base_folder = os.path.dirname(os.getcwd())
    input_folder = f'{base_folder}/data/antenna/inputs'
    output_folder = f'{base_folder}/data/antenna/outputs/mt_tree/automated_diff/ddtw'

    # Read frequency values
    features_df = pd.read_excel(f'{input_folder}/input_parameters.xlsx')
    real_df = pd.read_excel(f'{input_folder}/reel.xlsx')
    imaginary_df = pd.read_excel(f'{input_folder}/imaginary.xlsx')

    use_magnitudes = True
    tree_related_plots = False
    dtw_differences_str_list = ['_diff', ''] # could be '_diff' or ''
    dtw_window_list = [20, None]
    output_dist_metric = 'dtw'
    m_depth_list = [15]
    m_samples_split = 40
    m_samples_leaf_list = [20]
    max_ahead = 3
    m_features = None
    split_type = 'custom'
    target_setting = 'multi'
    target_diff_setting = False
    test_set_ratio = 0.8
    number_of_replicates = 5
    lambda_decay_setting = 0.5
    obj_weights_list = [[1., 0.1, 0.1], [1., 0.1, 0.], [1., 0., 0.1], [1.0, 0., 0.]]
    target_name = 'utilization'

    perf_df = pd.DataFrame()

    for replication in range(number_of_replicates):
        print(f"============== Replication: {replication} ==============")
        # output_array = dp.handle_antenna_output_data(real_df, imaginary_df, use_magnitudes)
        output_array = pd.DataFrame(np.swapaxes(output_array, 1, 2)[:, :, 0])

        X_train, X_test, y_train, y_test = train_test_split(features_df, output_array,
                                                            test_size=test_set_ratio, shuffle=True)
        train_inds = X_train.index
        for dtw_window in dtw_window_list:
            print(f"============== DTW Window: {dtw_window} ==============")
            for dtw_differences_str in dtw_differences_str_list:
                print(f"============== DTW Difference: {dtw_differences_str} ==============")
                dtw_distances_original_path = f"{input_folder}/dtw_{dtw_window}.csv"
                dtw_distances_path = f"{input_folder}/dtw{dtw_differences_str}_{dtw_window}.csv"
                if output_dist_metric == 'dtw':
                    if not os.path.exists(dtw_distances_path):
                        if 'diff' in dtw_differences_str:
                            timeseries = output_array.to_numpy()
                            d_0 = timeseries[:,:-2]
                            d_1 = timeseries[:,1:-1]
                            d_2 = timeseries[:,2:]
                            timeseries = ((d_1 - d_0) + (d_2 - d_0) / 2) / 2
                        else:
                            timeseries = output_array.to_numpy()
                        dtw_distance = dtw.distance_matrix_fast(timeseries, window=dtw_window)
                        np.savetxt(dtw_distances_path, dtw_distance, delimiter=",")
                    else:
                        dtw_distance = np.genfromtxt(dtw_distances_path, delimiter=',')
                    print(dtw_distance)
                    if 'diff' in dtw_differences_str:
                        if os.path.exists(dtw_distances_path):
                            dtw_distance_original = np.genfromtxt(dtw_distances_original_path, delimiter=',')
                        else:
                            timeseries = output_array.to_numpy()
                            dtw_distance_original = dtw.distance_matrix_fast(timeseries, window=dtw_window)
                    else:
                        dtw_distance_original = dtw_distance.copy()
                    dtw_distance_df = pd.DataFrame(dtw_distance)

                for m_depth in m_depth_list:
                    print(f"============== Tree Depth: {m_depth} ==============")
                    for m_samples_leaf in m_samples_leaf_list:
                        print(f"============== Min Samples Leaf: {m_samples_leaf} ==============")
                        for obj_weights in obj_weights_list:
                            print(f"============== Obj Weights: {obj_weights} ==============")
                            weights_str = "_".join(map(str, obj_weights))
                            output_dir = f'{output_folder}/dtw_window_{dtw_window}_dtw_diff{dtw_differences_str}_max_depth_{m_depth}_min_samples_leaf_{m_samples_leaf}_obj_weights_{weights_str}'
                            Path(output_dir).mkdir(parents=True, exist_ok=True)

                            dtw_distance_train_df = dtw_distance_df.iloc[train_inds, :]
                            dtw_distance_train_df = dtw_distance_train_df.iloc[:, train_inds]
                            tree = TreeForecast(target_type=target_setting, max_depth=m_depth, max_features=m_features,
                                                min_samples_leaf=m_samples_leaf, min_samples_split=m_samples_split,
                                                split_style=split_type, target_diff=target_diff_setting,
                                                lambda_decay=lambda_decay_setting, obj_weights = obj_weights, verbose=True)

                            tree.fit(X_train, dtw_distance_train_df)

                            train_ids = tree.apply(X_train, tree.max_depth)
                            y_train_df = pd.DataFrame(y_train)
                            y_train_df['leaf_id'] = train_ids
                            y_train_df = y_train_df.reset_index().rename(columns={'index': 'instance_id'})
                            y_train_df.to_csv(f'{output_dir}/train_df_indices.csv', index=False)

                            leaf_id_list = y_train_df['leaf_id'].unique()
                            train_leaf_counts = y_train_df['leaf_id'].value_counts().sort_values()
                            leaf_ids_plot = train_leaf_counts[:5].index.tolist() + train_leaf_counts[-5:].index.tolist()

                            print_statement_list = []
                            for leaf_id in leaf_ids_plot:
                                print_statements = []
                                y_train_df_filtered = y_train_df[y_train_df['leaf_id'] == leaf_id]
                                leaf_sample_instance = y_train_df_filtered.iloc[0]['instance_id']
                                y_train_df_filtered.drop(columns=['instance_id'], inplace=True)
                                print_statements = tree.printRule(X_train[X_train.index == leaf_sample_instance],
                                                                  tree.max_depth, tree.Tree, print_statements)
                                print_statement_list.append(print_statements)
                                leaf_count = train_leaf_counts[train_leaf_counts.index == leaf_id].values[0]
                                save_path = f'{output_dir}/leaf_id_{leaf_id}_instance_count_{leaf_count}.png'
                                plt.figure(figsize=(12, 10), dpi=200)
                                for _, row in y_train_df_filtered.iterrows():
                                    plt.plot(range(len(row.drop(['leaf_id']))), row.drop(['leaf_id']), linewidth=1.0)
                                plt.title(f'Leaf ID: {leaf_id}, Instances Count: {leaf_count}')
                                plt.savefig(save_path)
                                # plt.show()

                            print_statements_str = list(map('\n'.join, print_statement_list))
                            with open(f"{output_dir}/leaf_plots_rules.txt", "w") as text_file:
                                text_file.write('\n\n\n'.join(print_statements_str))

                            split_rules = tree.splits(tree.Tree)
                            split_rules_df = pd.DataFrame(split_rules, columns=['node_id', 'depth', 'feature', 'threshold'])
                            split_rules_df.to_csv(f'{output_dir}/split_rules.csv', index=False)
                            split_rules_df = pd.read_csv(f'{output_dir}/split_rules.csv')

                            test_ids = tree.apply(X_test, tree.max_depth)
                            y_test_df = pd.DataFrame(y_test)
                            y_test_df['leaf_id'] = test_ids
                            y_test_df = y_test_df.reset_index().rename(columns={'index': 'instance_id'})
                            y_test_df.to_csv(f'{output_dir}/test_df_indices.csv', index=False)

                            y_test_leaves_agg = tree.get_test_dtw_performance(y_train_df, y_test_df, dtw_distance_original)
                            row_details = {'dtw_window': dtw_window, 'replication': replication, 'dtw_diff': dtw_differences_str,
                                           'tree_depth': m_depth, 'min_samples_leaf': m_samples_leaf,
                                           'obj_weights': obj_weights, **y_test_leaves_agg}
                            perf_df_temp = pd.DataFrame.from_dict(row_details, 'index').T
                            perf_df = pd.concat([perf_df, perf_df_temp], axis=0)
                            perf_df.to_csv(f'{output_folder}/perf_df.csv', index=False)

                            # y_test_df_w_dists = y_test_df.apply(lambda x: get_dist_of_sample_to_leaf_instances(x, y_train_df), axis=1)

                            for idx in range(5):
                                leaf_id = y_test_df.iloc[idx]['leaf_id']
                                instance_id = y_test_df.iloc[idx]['instance_id']
                                y_train_df_filtered = y_train_df[y_train_df['leaf_id'] == leaf_id]
                                leaf_count = train_leaf_counts[train_leaf_counts.index == leaf_id].values[0]
                                save_path = f'{output_dir}/instance_idx_{idx}_leaf_id_{leaf_id}_instance_count_{leaf_count}.png'
                                drop_cols = ['leaf_id', 'instance_id']
                                plt.figure(figsize=(12, 10), dpi=200)
                                for _, row in y_train_df_filtered.iterrows():
                                    plt.plot(range(len(row.drop(drop_cols))), row.drop(drop_cols), linewidth=1.0)
                                plt.plot(range(len(row.drop(drop_cols))), y_test_df.iloc[idx][range(0,201)], linewidth=3.0)
                                plt.title(f'Test Instance: {instance_id}, Leaf ID: {leaf_id}, Instances Count: {leaf_count}')
                                plt.savefig(save_path)
