import pandas as pd
import numpy as np
from queue import Queue
from sklearn import datasets

class Node(object):
    '''
    representation of a node in a tree
    '''
    def __init__(self, node_args):
        self.index = node_args['index']
        self.depth = node_args['depth']
        self.parent = node_args['parent']
        self.direction = node_args['direction']
        self.is_leaf = node_args['is_leaf']
        self.left_child_ix = node_args['left_child_ix']
        self.right_child_ix = node_args['right_child_ix']
        self.split_feature = node_args['split_feature']
        self.split_value = node_args['split_value']
        self.metric_score = node_args['metric_score']
        self.class_counts = node_args['class_counts']
        self.pruned = False


class Gini(object):
    '''
    Class to define Gini impurity. Gini = 0 means split
    obtains node with pure classes
    '''
    metric_name = 'Gini'

    def __call__(self, le_grp, gt_grp):
        '''
        calculate gini impurity of left and right split, and average
        https://en.wikipedia.org/wiki/Decision_tree_learning#Gini_impurity
        '''
        prop_le = np.unique(le_grp, return_counts=True)[1] / np.shape(le_grp)
        prop_gt = np.unique(gt_grp, return_counts=True)[1] / np.shape(gt_grp)

        gini_lt = np.sum(prop_le - prop_le **2)
        gini_gt = np.sum(prop_gt - prop_gt **2)

        return np.mean((gini_lt, gini_gt))

    @staticmethod
    def score_improved(old_val, new_val):
        '''
        static method to determine if the Gini index
        has imporoved following a split.
        Static method is useful, as not all metrics are better
        when smaller!
        '''
        return new_val < old_val
    
    @staticmethod
    def should_prune(parent_val, children_val):
        '''
        When considering for pruning, return true
        if the parent val is less than or equal to the mean
        of the children value
        '''
        return parent_val <= children_val


class Tree(object):
    '''
    Store tree as list of nodes. We grow the tree using in-order
    traversal, which is a fact we use to store the tree and recover
    paths when we score samples
    '''

    def __init__(self, max_depth, metric):
        '''
        initialise the tree object
        '''
        self.max_depth = max_depth
        self.metric = metric
        self.nodes = []
        self.node_index_counter = 0
        self.is_grown = False

    def grow_tree(self, df, features, target, prune = True):
        '''
        assumes classes are labeled 0:(n classes-1)
        '''
        # how many unique classes in the outcome?
        self.n_cls = df[target].value_counts().shape[0]
        # recursive partition
        self.__rpart(df, features, target, depth=0, parent=None)

        # add child node indices by inspecting parents
        for i in range(len(self.nodes)):
            tmp = [x for x in self.nodes if x.parent == i]
            if tmp == []:
                self.nodes[i].left_child_ix = None
                self.nodes[i].right_child_ix = None
            else:
                LC = tmp[0].index
                RC = tmp[1].index
                self.nodes[i].left_child_ix = LC
                self.nodes[i].right_child_ix = RC

        # We can prune the tree to prevent overfitting
        # A very simple scheme has been implemented
        if prune:
            self.__prune()

        self.is_grown = True

    def __rpart(self, df, features, target, depth=0,
                parent = None, direction = None):
        '''
        Grow the tree using a recursive partitioning approach.
        Child indices will be added after recursion is finished
        '''
        
        # population proportions at each node
        class_counts = np.bincount(df[target].values, minlength = self.n_cls)
        # what is the purity of the current node?
        grp = df.loc[:, target].values
        metric_score = self.metric(grp, grp)
        
        # hold onto dict of Node parameters
        node_args = {'index' : self.node_index_counter,
                     'depth' : depth,
                     'metric_score' : metric_score,
                     'split_feature' : None,
                     'split_value' : None,
                     'is_leaf' : True,
                     'parent' : parent,
                     'direction' : direction,
                     'class_counts' : class_counts,
                     'left_child_ix' : None,
                     'right_child_ix' : None
             }

        if depth > self.max_depth:
            # we have hit a leaf node
            self.nodes.append(Node(node_args))

            self.node_index_counter +=1
            return

        # we only want to split if we improve purity following split
        is_leaf = True
        split_feature = None
        split_value = None
        split = False

        for f in features:
            # order the candidate slit values
            # (np.unique sorts and removes duplicates)
            splits = np.unique(df.loc[:, f].values)

            # if only one value, cannot split
            if splits.shape[0] == 1:
                continue

            # loop over split values, averaging the distance
            # between points to give some interpolation
            for i in range((len(splits) - 1)):
                s_val = np.mean(splits[i:i+2])
                le_grp = df.loc[df[f] <= s_val, target].values 
                gt_grp = df.loc[df[f] > s_val, target].values

                # what is purity with this split?
                tmp_metric_score = self.metric(le_grp, gt_grp)

                # did we improve on the best split?
                if self.metric.score_improved(metric_score, tmp_metric_score):
                    split = True
                    is_leaf = False
                    metric_score = tmp_metric_score
                    split_feature = f
                    split_value = s_val

        node_args.update({'is_leaf' : is_leaf, 
                          'metric_score' : metric_score,
                          'split_feature' : split_feature,
                          'split_value' : split_value})
        
        self.nodes.append(Node(node_args))

        if not split:
            # if no split was made, we have found a leaf
            self.node_index_counter += 1
            return

        # update parent, depth and counter
        parent = self.node_index_counter
        depth += 1
        self.node_index_counter += 1

        # Left child
        direction = 'L'
        self.__rpart(df[df[split_feature] <= split_value], features, target, depth, parent, direction)
        # Right child
        direction = 'R'
        self.__rpart(df[df[split_feature] > split_value], features, target, depth, parent, direction)

    def __prune(self):
        '''
        Very simple pruning scheme
        We look at parents of each leaf, and consider if the overall
        purity of the tree is improved. Use a queue instead of recursive
        scheme as I think its a bit easier to follow
        '''
        q = Queue()
        leaf_list = []
        # insert leaves into list
        for node in self.nodes:
            if node.is_leaf:
                leaf_list.append(node)

        # sort by depth
        leaf_list.sort(key = lambda x: x.depth, reverse = True)

        # insert into queue
        for node in leaf_list:
            q.put(self.nodes[node.parent].index)

        # while queue not empty
        while q.qsize():

            # process in order
            curr_parent_ix = q.get()
            metric_score_parent = self.nodes[curr_parent_ix].metric_score
            l_child = self.nodes[curr_parent_ix].left_child_ix
            r_child = self.nodes[curr_parent_ix].right_child_ix
            metric_score_children = 0.

            if self.nodes[l_child].pruned:
                continue

            if l_child:
                metric_score_children += self.nodes[l_child].metric_score
            if r_child:
                metric_score_children += self.nodes[r_child].metric_score

            # if the metric score of the parent is equal or better than the children
            # then prune
            if self.metric.should_prune(metric_score_parent, metric_score_children / 2.):
                self.nodes[curr_parent_ix].is_leaf = True
                self.nodes[l_child].pruned = True
                self.nodes[r_child].pruned = True
                q.put(self.nodes[curr_parent_ix].parent)


    def predict(self, feature_dict, predict_proba = False):
        '''
        Given dict holding features as key-value pairs, will
        predict the class of the observation
        '''
        if not self.is_grown:
            raise Exception('tree not grown, cannot predict')

        curr_node = 0

        while True:
            if self.nodes[curr_node].is_leaf:
                cnts = self.nodes[curr_node].class_counts
                if predict_proba:
                    return cnts / np.sum(cnts)
                else:
                    return cnts.argmax()

            curr_split_feature = self.nodes[curr_node].split_feature
            curr_split_value = self.nodes[curr_node].split_value

            if feature_dict[curr_split_feature] <= curr_split_value:
                curr_node = self.nodes[curr_node].left_child_ix
            else:
                curr_node = self.nodes[curr_node].right_child_ix


                
class RandomForest(object):

    def __init__(self, ntrees, max_depth, mtry, metric):
        self.ntrees = ntrees
        self.max_depth = max_depth
        self.mtry = mtry
        self.metric = metric
        self.trees = []


    def grow_forest(self, df, features, target, seed = 1234):
        '''
        as per brieman's specification, we take bootstrap sample of
        data, and randomly select features to ensure trees arent correlated
        dont prune!
        As an obvious performance enhancement, consider growing trees in parralel.
        df: input pandas dataframe
        features: list of feature names (string)
        target: target name (string)
        '''
        self.n_cls = df[target].value_counts().shape[0]
        np.random.seed(seed)

        # Growing forest is an obvious candidate for parallel processing
        for i in range(self.ntrees):
            
            # bootstrap sample of data
            df_ix = df.index.values
            bag_ix = np.random.choice(df_ix, size = df_ix.shape[0], replace = True)
            # not actually using this atm, could in principle keep hold 
            # of out of bag performance
            oob_ix = np.setdiff1d(df_ix, np.unique(bag_ix))

            # randomly select mtry features
            feature_ix = np.random.choice(np.arange(len(features)), self.mtry, replace = False)
            curr_features = [features[x] for x in feature_ix]

            # initialise tree
            tree = Tree(max_depth = self.max_depth,
                        metric = self.metric
                        )
            # grow unpruned tree CHECK THIS- SHOULD THIS BE ILOC OR LOC
            tree.grow_tree(df.loc[bag_ix, :], curr_features, target, prune=False)

            # add tree to forest
            self.trees.append(tree)


    def predict(self, feature_dict, predict_proba = False, no_majority = False):
        '''
        predict outcome for new sample
        Note: this isnt vectorised, takes one sample at a time
        this would be an obvious performance enhancement
        '''
        votes = np.zeros(self.n_cls)
        for tree in self.trees:
            curr_node = 0

            while True:
                if tree.nodes[curr_node].is_leaf:
                    tmp = tree.nodes[curr_node].class_counts
                    votes += tmp / np.sum(tmp)
                    break

                curr_split_feature = tree.nodes[curr_node].split_feature
                curr_split_value = tree.nodes[curr_node].split_value

                if feature_dict[curr_split_feature] <= curr_split_value:
                    curr_node = tree.nodes[curr_node].left_child_ix
                else:
                    curr_node = tree.nodes[curr_node].right_child_ix

        if predict_proba:
            return votes / self.ntrees
        # if no class has > 0.5 certainty
        elif no_majority:
            if (votes[votes.argmax()] / self.ntrees) > 0.5:
                return votes.argmax()
            else:
                return self.n_cls 
        else:
            return votes.argmax()
            