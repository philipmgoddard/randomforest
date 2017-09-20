import pandas as pd
import numpy as np
from sklearn import datasets


class Node(object):

    def __init__(self, index, depth, parent, direction,
                 is_leaf, left_child_ix, right_child_ix,
                 split_feature, split_value, metric_score,
                 class_counts):
        self.index = index
        self.depth = depth
        self.parent = parent
        self.direction = direction
        self.is_leaf = is_leaf
        self.left_child_ix = left_child_ix
        self.right_child_ix = right_child_ix
        self.split_feature = split_feature
        self.split_value = split_value
        self.metric_score = metric_score
        self.class_counts = class_counts


class Gini(object):
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

    # Gini = 0 means pure
    @staticmethod
    def score_improved(old_val, new_val):
        if old_val is None:
            return True
        else:
            return old_val > new_val


class Tree(object):
    '''
    We take advantage of how tree is grown to represent as a list of nodes.
    The child nodes (l_child_ix and r_child_ix) correspond to the index in the list.

    '''

    def __init__(self, max_depth, min_grp_size, metric):
        '''

        '''
        self.max_depth = max_depth
        self.min_grp_size = min_grp_size
        self.metric = metric
        self.nodes = []
        self.node_index_counter = 0
        self.is_grown = False


    def grow_tree(self, df, features, target, prune = True):
        '''
        assumes classes are labeled 0:(n classes-1)
        '''
        # recursive partition
        self.n_cls = df[target].value_counts().shape[0]
        self.__rpart(df, features, target, depth=0, parent=None)

        # add in child nodes by inspecting parents
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

        if prune:
            self.__prune()

        self.is_grown = True


    def __rpart(self, df, features, target, depth=0, parent = None, direction = None):
        '''
        Child indices will be added after recursion is finished
        '''

        # population proportions at each node
        class_counts = np.bincount(df[target].values, minlength = self.n_cls)

        if depth > self.max_depth:
            # LEAF
            grp = df.loc[:, target].values
            metric_score = self.metric(grp, grp)
            self.nodes.append(Node(index = self.node_index_counter, depth = depth,
                                   metric_score = None, split_feature = None,
                                   split_value = None, is_leaf = True,
                                   parent=parent, direction = direction,
                                   class_counts = class_counts,
                                   left_child_ix = None, right_child_ix = None))
            self.node_index_counter +=1
            return

        # interested in the split optimises metric
        metric_score = None
        is_leaf = True
        split_feature = None
        split_value = None
        split = False

        for f in features:
            # order the candidates (np.unique sorts and removes duplicates)
            splits = np.unique(df.loc[:, f].values)

            # if only one value, cannot split
            if splits.shape[0] == 1:
                continue

            for s in splits[:-1]: # all but last, as left split is <=
                le_grp = df[df[f] <= s].loc[:, target].values
                gt_grp = df[df[f] > s].loc[:, target].values

                # if split leads to a too small sample
                if ((le_grp.shape[0] < self.min_grp_size) or (gt_grp.shape[0] < self.min_grp_size)):
                    continue

                tmp_metric_score = self.metric(le_grp, gt_grp)

                # did we improve on best?
                if self.metric.score_improved(metric_score, tmp_metric_score):
                    split = True
                    is_leaf = False
                    metric_score = tmp_metric_score
                    split_feature = f
                    split_value = s

        if not split:
            # its a leaf, whats the purity of the leaf?
            grp = df.loc[:, target].values
            metric_score = self.metric(grp, grp)

        self.nodes.append(Node(index = self.node_index_counter, depth = depth,
                               metric_score = metric_score, split_feature = split_feature,
                               split_value = split_value,is_leaf = is_leaf,
                               parent = parent, direction = direction,
                               class_counts = class_counts,
                               left_child_ix = None, right_child_ix = None))

        if not split:
            # its a leaf, pop stack
            self.node_index_counter += 1
            return

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
        # VERY simple pruning scheme- only criteria is that purity is not improved
        # between child and parent
        # build queue of leaves
        # process
        # if prune, add parent to back of queue
        # once queue empty, break
        pass


    def predict(self, feature_dict, predict_proba = False):
        '''
        Useful docstring

        Given new user, will output recommendations
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




    def get_rules(self):
        # can we add direction as a node property? i.e. to denote if a left or
        # a right child
        # Then a list of strings that represent each rule
        # TAKE OUT OF __repr__ and call get_rules(self) - check to see if initialised then away we go
        #

        if not self.is_grown:
            raise Exception('tree not grown, cannot predict')

        rules = []
        out = ''
        for node in self.nodes:
            print()
            if node.is_leaf:
                rule = []
                rule.append(['class:' + str(node.class_counts.argmax())])
                print('LEAF', node.class_counts)
                tmp_node = node
                while True:

                    tmp_parent = tmp_node.parent
                    if tmp_parent is None:
                        break
                    direction = (lambda x: '<=' if x == 'L' else '>')(tmp_node.direction)
                    split_feature = self.nodes[tmp_parent].split_feature
                    split_value = self.nodes[tmp_parent].split_value
                    rule.append([ split_feature + direction + str(split_value)])

                    tmp_node = self.nodes[tmp_parent]
                rule = rule[::-1]
                rules.append(rule)


        # what we want is a nicely formatted set of rules to be displayed!!
        return rules


# We could create a TreeRF class by inheritance. This could have extra
# beaviour and attributes, eg store performance on oob samples

class RandomForest(object):

    def __init__(self, ntrees, max_depth, mtry, min_grp_size, metric):
        self.ntrees = ntrees
        self.max_depth = max_depth
        self.mtry = mtry
        self.min_grp_size = min_grp_size
        self.metric = metric
        self.trees = []


    def grow_trees(self, df, features, target):
        '''
        as per brieman's specification, we take bootstrap sample of
        data, and randomly select features to ensure trees arent correlated
        dont prune!
        df: input pandas dataframe
        features: list of feature names (string)
        target: target name (string\)
        '''
        self.n_cls = df[target].value_counts().shape[0]

        # Growing forest is an obvious candidate for parallel processing
        for i in range(self.ntrees):
            print('growing tree', i)
            tree = Tree(self.max_depth, self.min_grp_size, self.metric, prune = False)

            # bootstrap sample of data
            df_ix = df.index.values
            bag_ix = np.random.choice(df_ix, size = df_ix.shape[0], replace = True)
            oob_ix = np.setdiff1d(df_ix, np.unique(bag_ix)) # not actually using this atm

            # randomly select mtry features
            feature_ix = np.random.choice(np.arange(len(features)), self.mtry, replace = False)
            curr_features = [features[x] for x in feature_ix]

            # initialise tree
            tree = Tree(max_depth = self.max_depth,
                        min_grp_size = self.min_grp_size,
                        metric = self.metric
                        )
            # grow tree
            tree.grow_tree(df.loc[bag_ix, :], curr_features, target)

            # add tree to forest
            self.trees.append(tree)


    def predict(self, feature_dict, predict_proba = False):
        '''
        predict outcome for new sample
        This isnt vectorised, takes one sample at a time
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
        else:
            return votes.argmax()


iris = datasets.load_iris()
X = iris.data
y = iris.target
iris_df = pd.DataFrame(X, columns = ['pl','pw','sl','sw'])
iris_df['species'] = y

rpart = Tree(max_depth = 4, min_grp_size = 10, metric = Gini())
rpart.grow_tree(df = iris_df, features = ['pl','pw','sl','sw'], target = 'species')

tst = iris_df.loc[1,: ][:4].to_dict()
rpart.predict(tst)

# rf = RandomForest(ntrees= 10, max_depth= 4, mtry = 2, min_grp_size= 10, metric = Gini())
# rf.grow_trees(df = iris_df, features = ['pl','pw','sl','sw'], target = 'species')


# TODO: plot decision boundaries, make sure is sensible
