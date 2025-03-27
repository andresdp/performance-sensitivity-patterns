import math
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from scipy import stats

from ema_workbench import RealParameter, IntegerParameter, ScalarOutcome, Constant, Model
from ema_workbench import MultiprocessingEvaluator, ema_logging, perform_experiments
from ema_workbench.analysis import pairs_plotting
from ema_workbench.analysis import prim
from ema_workbench.analysis import feature_scoring
from ema_workbench.em_framework.evaluators import Samplers
from ema_workbench.em_framework.salib_samplers import get_SALib_problem
from ema_workbench import save_results
from ema_workbench import load_results
from ema_workbench.analysis import cart
from ema_workbench.analysis.scenario_discovery_util import RuleInductionType
from sklearn.tree._tree import TREE_LEAF, TREE_UNDEFINED
from sklearn.tree import _tree
from sklearn.model_selection import train_test_split
from sklearn import manifold

from SALib.analyze import sobol
from tqdm import tqdm

from collections import Counter
from natsort import natsorted
import itertools

from collections import Counter

from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import OrdinalEncoder

import prim as rhodium_prim

from operator import itemgetter

import umap
import umap.plot
from umap import UMAP


class ArchSpace:

    RESPONSE_TIME_LABELS = ['fast', 'average', 'slow'] #['very-fast', 'fast', 'average', 'low', 'very-low'] #['fast', 'average', 'low']
    UTILIZATION_LABELS = ['low', 'average', 'high']
    # THROUGHPUT_LABELS = ['low', 'average', 'high'] 
    # AVAILABILITY_LABELS = ['high', 'average', 'low']

    ALL_TRADEOFF_LABELS = natsorted([','.join(t) for t in itertools.product(*[RESPONSE_TIME_LABELS, UTILIZATION_LABELS])])

    ALL_LABELS = dict()
    ALL_LABELS['response_time'] = RESPONSE_TIME_LABELS
    ALL_LABELS['utilization'] = UTILIZATION_LABELS
    # ALL_LABELS['throughput'] = THROUGHPUT_LABELS
    # ALL_LABELS['availability'] = AVAILABILITY_LABELS
   
    def create_model(self, name):
        self.name_ = name
        self.model_ = None #Model(name=name, function=ArchSpace.modelFunction)
        
        return self.model_
    
    def get_model(self):
        return self.model_
    
    def load_results(self, path):
        pass
        return None, None
    
    def run_experiments(self, n_scenarios=100, n_policies=1, path=None, remove_initial_architecture=False):

        # self.experiments_df_, outcomes = self.results_
        self.outcomes_df_ = None #pd.DataFrame.from_dict(outcomes)
        self.experiments_df = None

        return self.experiments_df_, self.outcomes_df_

    def get_experiment(self, configuration=0): 

        exp_filter = self.experiments_df_
        policy = exp_filter['policy'].unique()
        # print('policies:', policy)
        exp_out_df = pd.DataFrame()
        if len(policy) > 0:
            # exp_filter = exp_filter[exp_filter['policy'] == policy[0]]
            exp_filter = exp_filter[exp_filter['policy'] == configuration]
            # print(exp_filter)
            # print(self.outcomes_df)
            exp_out_df = self.outcomes_df_.loc[exp_filter.index]
            discrete_exp_out_df = None
            if hasattr(self, 'discrete_outcomes_df_'):
                discrete_exp_out_df = self.discrete_outcomes_df_.loc[exp_filter.index]
    
        return exp_filter, exp_out_df, discrete_exp_out_df
    
    def get_experiments(self, configuration_list=[0]): 
        exp_filter_list = []
        exp_out_df_list = []
        discrete_out_df_list = []
        for c in configuration_list:
            exp_filter, exp_out_df, discrete_exp_out_df = self.get_experiment(c)
            if len(exp_filter) > 0:
                exp_filter_list.append(exp_filter)
                exp_out_df_list.append(exp_out_df)
                if discrete_exp_out_df is not None:
                    discrete_out_df_list.append(discrete_exp_out_df)

        exp_filter = pd.concat(exp_filter_list)
        exp_out_df = pd.concat(exp_out_df_list)
        discrete_exp_out_df = None
        if len(discrete_out_df_list) > 0:
            discrete_exp_out_df = pd.concat(discrete_out_df_list)
    
        return exp_filter, exp_out_df, discrete_exp_out_df
    
    def get_configurations(self):
        all_configs = [0]
        return all_configs
    
    def get_config_parameters(policy):
        return []

    def get_nearest_configurations(self, configuration=0, k=5, metric='euclidean'):
        # TODO: Does this return a void dataframe or a list? Real behavior needs to be provided
        # return pd.DataFrame(columns=self.INPUT_PARAMETERS)
        return [configuration]
    
    def get_nearest_tradeoffs(self, qa_tradeoff, k=5, metric='euclidean', separator=','):
        reference_tradeoff = qa_tradeoff.split(separator)
        tradeoff_list = [qat.split(separator) for qat in self.available_tradeoffs_.keys()]
        df = pd.DataFrame(tradeoff_list, columns=self.OUTPUTS) 
        
        # Create mapper object
        mapper = dict()
        for c in self.ALL_LABELS.keys():
            mapper[c] = dict()
            value = -1
            for label in self.ALL_LABELS[c]:
                mapper[c][label] = value
                value = value + 1
        #print(mapper)

        for c in df.columns:
            df[c] = df[c].replace(mapper[c]) # Apply on each column
        # print(df)
        
        nbrs = NearestNeighbors(n_neighbors=k, metric=metric)
        nbrs.fit(df.values)
        #print("Reference tradeoff:", reference_tradeoff)
        df = pd.DataFrame(np.array(reference_tradeoff).reshape(1,len(self.OUTPUTS)), columns=self.OUTPUTS) 
        # print(df)
        for c in df.columns:
            df[c] = df[c].replace(mapper[c])
        # print(df)
        y = df.loc[0, :].values.tolist() # Convert the first/unique row to a list
        # print(type(y), y)
        distances, indices = nbrs.kneighbors([y])
        print("Nearest neighbors for", qa_tradeoff, ":", indices[0], distances)

        return [separator.join(tradeoff_list[i]) for i in indices[0]]
    
    # Split the values of the column into n buckets, according to the distribution of values
    @staticmethod
    def get_distribution_bins(col, n, min_max=(None,None)): 
        unique_values = set(col.values)
        if min_max == (None,None):
            min_x = np.min(list(unique_values))
            max_x = np.max(list(unique_values))
        else:
            print("Using predefined limits",min_max)
            min_x = min_max[0]
            max_x = min_max[1]
        #print(min_x, max_x)
        unique_values.update([min_x, max_x])
        unique_values = sorted(unique_values)
    
        return sorted(set(pd.qcut(unique_values, n).map(lambda x: x.left, na_action="ignore"))) + [max(unique_values)]

    @staticmethod
    def get_bins(col, n, min_max=(None,None)):
        unique_values = sorted((set(col.values)))
        if min_max == (None,None):
            min_x = np.min(unique_values)
            max_x = np.max(unique_values)
        else:
            #print("Using predefined limits",min_max)
            min_x = min_max[0]
            max_x = min_max[1]
        min_x = min_x - 0.1
        max_x = max_x + 0.1
        delta = (max_x - min_x)/n
        #print(min_x, max_x, delta)
        return ([min_x+i*delta for i in range(0,n)] + [max_x])
    
    @staticmethod
    def get_tradeoffs(df, separator=','):
        temp = df.to_string(header=False,index=False,index_names=False).split('\n')
        tradeoffs = [separator.join(ele.split()) for ele in temp]
        return Counter(tradeoffs)
    
    @staticmethod
    def discretize(df, n_bins=3, mins_maxs=(None,None), all_labels=None):
        discrete_df = df.copy()
        for idx, c in enumerate(df.columns):
            qa = df[c]
            min_max = (None,None)
            if mins_maxs != (None,None):
                min_max = mins_maxs[idx]
            qa_bins = ArchSpace.get_bins(qa, n_bins, min_max=min_max)
            # print(qa_bins, c, all_labels[c])
            qa_labels = pd.cut(qa, bins=qa_bins, labels=all_labels[c])
            discrete_df[c] = qa_labels

        available_tradeoffs = ArchSpace.get_tradeoffs(discrete_df) #{k:v for (k,v) in natsorted(Counter(tradeoffs).items())}
        return discrete_df, available_tradeoffs

    def _get_cluster_labels(self, df=None, separator=',', encode=False):
        if df is None:
            df = self.discrete_outcomes_df_
        temp = df.to_string(header=False,index=False,index_names=False).split('\n')
        tradeoffs = [separator.join(ele.split()) for ele in temp]
        if encode:
            encoder = {x:idx for idx,x in enumerate(self.available_tradeoffs_.keys())}
            tradeoffs = [encoder[t] for t in tradeoffs]
        return tradeoffs
    
    def discretize_outcomes(self, n_bins=3, mins_maxs=(None,None)):
        self.discrete_outcomes_df_, self.available_tradeoffs_ = ArchSpace.discretize(
                self.outcomes_df_, n_bins=n_bins, mins_maxs=mins_maxs, all_labels=self.ALL_LABELS)  
        
        self.label_values_df_ = self.describe_labels(n_bins=n_bins, mins_maxs=mins_maxs)  
        
        return self.discrete_outcomes_df_

    def compute_robustness(self, configurations, qa_tradeoff=None, metric='r1'):

        if isinstance(qa_tradeoff, list):
            acum_r = 0
            qat_list = []
            for qat in qa_tradeoff:
                r, reference_tradeoff = self.compute_robustness(configurations, qat)
                acum_r = acum_r + r
                qat_list.append(reference_tradeoff)
                #print("-->", qat, r, configurations)
            return acum_r, qat_list

        #if all(isinstance(elem, list) for elem in configurations): # Several configurations in a list
        if isinstance(configurations, list): # Several configurations in a list
            #print("This is a list of configurations:", len(configurations))
            _, _, discrete_exp_out_df = self.get_experiments(configurations)
        else: # Single configuration
           _, _, discrete_exp_out_df = self.get_experiment(configurations)

        if discrete_exp_out_df is None:
            print("Warning: A discretization of the outcomes is firstly required!")
            return None
        
        # Compute local tradeoffs (for the selected configuration)
        local_tradeoffs = ArchSpace.get_tradeoffs(discrete_exp_out_df)

        if qa_tradeoff is None:
            qa_tradeoff = local_tradeoffs.most_common(1)[0]
            reference_tradeoff = qa_tradeoff[0]
            print("Selecting most common (local) tradeoff:", reference_tradeoff)
        else:
            reference_tradeoff = qa_tradeoff
            qa_tradeoff = (qa_tradeoff, local_tradeoffs[qa_tradeoff])
        
        r = qa_tradeoff[1]/(local_tradeoffs.total())
        #print("Robustness of single solution w.r.t tradeoff:", reference_tradeoff, "-->", round(r*100, 2), "%")
    
        return r, reference_tradeoff

    @staticmethod
    def _count_tradeoffs(tradeoffs, normalize=True, all_tradeoffs=True, all_labels=None):
        counter_frequency_labels = dict()
        for t in all_labels:
            if all_tradeoffs:
                counter_frequency_labels[t] = 0
            if t in tradeoffs:
                counter_frequency_labels[t] = tradeoffs[t]
        
        if normalize:
            total = sum(counter_frequency_labels.values(), 0.0)
            for k in counter_frequency_labels:
                counter_frequency_labels[k] /= total

        return counter_frequency_labels

    def describe_labels(self, all_labels=None, n_bins=3, show_used=False, mins_maxs=(None,None)):
        if all_labels is None:
            all_labels = self.ALL_LABELS
        list_items = []
        for idx, c in enumerate(self.outcomes_df_.columns):
            qa = self.outcomes_df_[c]
            min_max = (None,None)
            if mins_maxs != (None,None):
                min_max = mins_maxs[idx]
            qa_bins = ArchSpace.get_bins(qa, n_bins, min_max=min_max)
            # print(c,qa_bins)
            for i in range (0, n_bins):
                dict_i = dict()
                dict_i['objective'] = c
                dict_i['min'] = qa_bins[i]
                dict_i['max'] = qa_bins[i+1]
                dict_i['label'] = all_labels[c][i]
                list_items.append(dict_i)
    
        labels_df = pd.DataFrame(list_items)
    
        if show_used:
            #_, tradeoffs = discretize_outcomes(df_outcomes, all_labels, n_bins=n_bins)
            tradeoffs = ArchSpace.ALL_TRADEOFF_LABELS #tradeoffs.keys()
            all_indices = []
            for idx, c in enumerate(self.outcomes_df_.columns):
                used_labels = {t.split(',')[idx] for t in tradeoffs}
                #print(c, used_labels)
                indices = labels_df[(labels_df['objective'] == c) & (labels_df['label'].isin(used_labels))].index.tolist()
                #print(c, indices)
                all_indices.extend(indices)
            # print("Used rows:", all_indices)
            # df_labels_styled = apply_colors(df_labels, all_indices, colors=my_colors, qa_values=False)
        
        return labels_df
    
    def get_property_box(self, label, df=None):
        if df is None:
            df = self.label_values_df_ 
        label_list = label.split(',')
        result = []
        for idx, l in enumerate(label_list):
            row = df[df['label'] == l][['min','max']].values.tolist()
            #print(row)
            if len(row) > 1:
                row = [row[idx]]
            result.extend(row)
        result = [tuple(l) for l in result] # Convert to tuple
        result_dict = dict()
        for k,v in zip(self.OUTPUTS, result):
            # print(k, v)
            result_dict[k] = v
        return result_dict #result
    
    def adjust_parameters(self, parameters_dict):
        # print(self.experiments_df_.shape)
        for p in parameters_dict.keys():
            p_min = parameters_dict[p]['min']
            p_max = parameters_dict[p]['max']
            indices_p = (self.experiments_df_[p]>=p_min)&(self.experiments_df_[p]<=p_max)
            self.experiments_df_ = self.experiments_df_[indices_p]
            self.outcomes_df_ = self.outcomes_df_[indices_p]
            # self.discrete_outcomes_df_ = self.discrete_outcomes_df_[indices_p]
        self.experiments_df_.reset_index(drop=True, inplace=True)
        self.outcomes_df_.reset_index(drop=True, inplace=True)
        # self.discrete_outcomes_df_.reset_index(drop=True, inplace=True)
        # print("Instances after adjusting parameters:", self.experiments_df_.shape)

    def run_prim(self, property, threshold=0.8, method='rhodium', key_parameters=None, policy=None, 
                 n_boxes=False, verbose=True):
        i = self.OUTPUTS[0]
        j = self.OUTPUTS[1]

        exps_df = self.experiments_df_
        outs_df = self.outcomes_df_
        if policy is not None:
            p_indices = (self.experiments_df_['policy'] == policy)
            exps_df = self.experiments_df_[p_indices].copy().reset_index(drop=True)
            outs_df = self.outcomes_df_[p_indices].copy().reset_index(drop=True)
        rt = outs_df[i]
        u = outs_df[j]

        # The property to ensure is that values are within the box (tradeoff)
        y = (rt >= property[i][0]) & (rt <= property[i][1]) & (u >= property[j][0]) & (u <= property[j][1])
        x = exps_df.drop(columns=['policy', 'model', 'scenario'])
        if key_parameters is not None:
            x = x[key_parameters]
        if verbose:
            print('Instances satisfying property:', y.value_counts())
            print("Total instances:", x.shape)

        if verbose:
            print("Running PRIM ...", method)
        if method == 'rhodium':
            prim_alg = rhodium_prim.Prim(x, y, threshold=threshold)
            all_boxes = prim_alg.find_all()
        else:
            prim_alg = prim.Prim(x, y, threshold=threshold)
            box = prim_alg.find_box()
            all_boxes = [box]

        if verbose:
            print(len(all_boxes), 'possible boxes')
        if n_boxes:
            return all_boxes
        else:
            box1 = all_boxes[0]
            if method == 'rhodium':
                df = ArchSpace.get_box_limits(box1)[['min', 'max']]
                return box1, df.to_dict(orient='index'), prim_alg
            else:
                # TODO: Alternatively, try prim_alg.boxes_to_dataframe()
                df = box1.inspect(style='data')[0][1].to_dict(orient='index')
                for key in df.keys():
                    df[key] = {k2:v for (k1,k2),v in df[key].items() if k2 in ['min', 'max']}
                return box1, df, prim_alg
    
    @staticmethod
    def get_box_limits(box):
        stats = box.peeling_trajectory.iloc[box._cur_box].to_dict()
        stats['restricted_dim'] = stats['res dim']

        qp_values = box._calculate_quasi_p(box._cur_box)
        
        uncs = [(key, value) for key, value in qp_values.items()]
        uncs.sort(key=itemgetter(1))
        uncs = [uncs[0] for uncs in uncs]
        
        box_lim = pd.DataFrame(np.zeros((len(uncs), 3)), 
                               index=uncs, 
                               columns=['min', 'max', 'qp values'])
        
        for unc in uncs:
            values = box._box_lims[box._cur_box][unc][:]
            box_lim.loc[unc] = [values[0], values[1], qp_values[unc]]
             
        return box_lim
    
    @staticmethod
    def _is_leaf(inner_tree, index):
    # Check whether node is leaf node
        return (inner_tree.children_left[index] == TREE_LEAF and 
            inner_tree.children_right[index] == TREE_LEAF)

    @staticmethod
    def _prune_index(inner_tree, decisions, index=0):
        # # Start pruning from the bottom - if we start from the top, we might miss
        # # nodes that become leaves during pruning.
        # # Do not use this directly - use prune_duplicate_leaves instead.
        # if not ArchSpace._is_leaf(inner_tree, inner_tree.children_left[index]):
        #     ArchSpace._prune_index(inner_tree, decisions, inner_tree.children_left[index])
        # if not ArchSpace._is_leaf(inner_tree, inner_tree.children_right[index]):
        #     ArchSpace._prune_index(inner_tree, decisions, inner_tree.children_right[index])

        # # Prune children if both children are leaves now and make the same decision:     
        # if (ArchSpace._is_leaf(inner_tree, inner_tree.children_left[index]) and
        #     ArchSpace._is_leaf(inner_tree, inner_tree.children_right[index]) and
        #     (decisions[index] == decisions[inner_tree.children_left[index]]) and 
        #     (decisions[index] == decisions[inner_tree.children_right[index]])):
        #     # turn node into a leaf by "unlinking" its children
        #     inner_tree.children_left[index] = TREE_LEAF
        #     inner_tree.children_right[index] = TREE_LEAF
        #     inner_tree.feature[index] = TREE_UNDEFINED
        #     ##print("Pruned {}".format(index))
        
        # Start pruning from the bottom - if we start from the top, we might miss
        # nodes that become leaves during pruning.
        # Do not use this directly - use prune_duplicate_leaves instead.
        if not ArchSpace._is_leaf(inner_tree, inner_tree.children_left[index]):
            ArchSpace._prune_index(inner_tree, decisions, inner_tree.children_left[index])
        if not ArchSpace._is_leaf(inner_tree, inner_tree.children_right[index]):
            ArchSpace._prune_index(inner_tree, decisions, inner_tree.children_right[index])

        # Prune children if both children are leaves now and make the same decision:     
        if (ArchSpace._is_leaf(inner_tree, inner_tree.children_left[index]) and
            ArchSpace._is_leaf(inner_tree, inner_tree.children_right[index]) and
            (decisions[index] == decisions[inner_tree.children_left[index]]) and 
            (decisions[index] == decisions[inner_tree.children_right[index]])):
            # turn node into a leaf by "unlinking" its children
            inner_tree.children_left[index] = TREE_LEAF
            inner_tree.children_right[index] = TREE_LEAF
            inner_tree.feature[index] = TREE_UNDEFINED
            ##print("Pruned {}".format(index))

    @staticmethod
    # https://stackoverflow.com/questions/51397109/prune-unnecessary-leaves-in-sklearn-decisiontreeclassifier
    # https://github.com/scikit-learn/scikit-learn/issues/10810
    def prune_duplicate_leaves(mdl):
        # Remove leaves if both 
        decisions = mdl.tree_.value.argmax(axis=2).flatten().tolist() # Decision for each node
        ArchSpace._prune_index(mdl.tree_, decisions)
        return mdl
    
    # https://stackoverflow.com/questions/20224526/how-to-extract-the-decision-rules-from-scikit-learn-decision-tree#%20decision-rules-from-scikit-learn-decision-tree
    @staticmethod
    def _get_rules(tree, feature_names, class_names):
        tree_ = tree.tree_
        feature_name = [
            feature_names[i] if i != _tree.TREE_UNDEFINED else "undefined!"
            for i in tree_.feature
        ]

        paths = []
        path = []
        
        def recurse(node, path, paths):
            
            if tree_.feature[node] != _tree.TREE_UNDEFINED:
                name = feature_name[node]
                threshold = tree_.threshold[node]
                p1, p2 = list(path), list(path)
                p1 += [('<=', name, np.round(threshold, 3))] # [f"({name} <= {np.round(threshold, 3)})"]
                recurse(tree_.children_left[node], p1, paths)
                p2 += [('>', name, np.round(threshold, 3))] # [f"({name} > {np.round(threshold, 3)})"]
                recurse(tree_.children_right[node], p2, paths)
            else:
                path += [(tree_.value[node], tree_.n_node_samples[node])]
                paths += [path]
                
        recurse(0, path, paths)

        # sort by samples count
        samples_count = [p[-1][1] for p in paths]
        ii = list(np.argsort(samples_count))
        paths = [paths[i] for i in reversed(ii)]
        
        rules = []
        triples = []
        for path in paths:
            rule = "if "
            tuple_rule = []
            for p in path[:-1]:
                # print(p)
                tuple_rule.append(p)
                if rule != "if ":
                    rule += " and "
                rule += p[1] + " " + p[0] + " " + str(p[2]) #str(p)
            rule += " then "
            if class_names is None:
                rule += "response: "+str(np.round(path[-1][0][0][0],3))
            else:
                classes = path[-1][0][0]
                l = np.argmax(classes)
                # print(classes, l)
                prob_class = np.round(classes[l]/np.sum(classes),2)
                rule += f"class: {class_names[l]} (proba: {100.0*prob_class}%)"
                tuple_rule.append(('class', class_names[l], prob_class))
            rule += f" | based on {path[-1][1]:,} samples"
            rules += [rule]
            triples.append(tuple_rule)
            
        return rules, triples
    
    def run_cart_raw(self, key_parameters=None, policy=None, threshold=0.05, prune_tree=False, mass_min=None):

        exps_df = self.experiments_df_
        outs_df = self.discrete_outcomes_df_
        if policy is not None:
            p_indices = (self.experiments_df_['policy'] == policy)
            # print(p_indices)
            exps_df = self.experiments_df_[p_indices].copy().reset_index(drop=True)
            outs_df = self.self.discrete_outcomes_df_[p_indices].copy().reset_index(drop=True)
        
        if key_parameters is not None:
            x = exps_df[key_parameters]
        else:
            x = exps_df.drop(columns=['policy', 'model', 'scenario'])
        y = self._get_cluster_labels(outs_df, encode=True)

        print(x.shape, x.columns)
        if mass_min is None:
            mass_min =  2.0 / x.shape[0]
        print("Running CART ...", mass_min)
        cart_alg = cart.CART(x, y, mass_min, mode=RuleInductionType.CLASSIFICATION)
        cart_alg.build_tree()

        if prune_tree:
            cart_alg.clf = ArchSpace.prune_duplicate_leaves(cart_alg.clf)

        class_names = [*Counter(self._get_cluster_labels())]
        string_rules, triple_rules = ArchSpace._get_rules(cart_alg.clf, feature_names=key_parameters, class_names=class_names) #self._get_cluster_labels())
        #print(triple_rules)

        return cart_alg, string_rules, triple_rules 
    
    @staticmethod 
    def intersect_intervals_from_paths(paths, variable_name, min_bound=None, max_bound=None):
        """
        Computes the intersection of all intervals related to a given variable.

        Args:
        paths (list of list of tuples): Each path is a list of tuples in the format (operator, variable_name, value).
        variable_name (str): The name of the variable for which the intersection is computed.

        Returns:
        tuple: The intersection interval for the specified variable or None if no intersection exists.
        """
        # Initialize min and max bounds to infinity and negative infinity
        if min_bound is None:
            min_bound = float('-inf')
        if max_bound is None:
            max_bound = float('inf')

        # Iterate over all paths to find constraints for the specified variable
        for path in paths:
            for (operator, var_name, value) in path:
                if var_name == variable_name:
                    if operator == '<=':
                        max_bound = min(max_bound, value)
                    elif operator == '<':
                        max_bound = min(max_bound, value - 1e-9)  # Slightly less for strict less-than
                    elif operator == '>=':
                        min_bound = max(min_bound, value)
                    elif operator == '>':
                        min_bound = max(min_bound, value + 1e-9)  # Slightly more for strict greater-than

        # Check if there is a valid intersection
        if min_bound > max_bound:
            return None

        return (min_bound, max_bound)
    
    def run_cart(self, key_parameters=None, policy=None, threshold=0.05, prune_tree=False, mass_min=None):
        cart_alg, string_rules, triple_rules = self.run_cart_raw(key_parameters=key_parameters,
                                            policy=policy, prune_tree=prune_tree, mass_min=mass_min)
        cart_boxes = dict()
        for t in triple_rules:
            qa_label = t[-1][1]
            if qa_label not in cart_boxes.keys():
                cart_boxes[qa_label] = []
            vars = set([x[1] for x in t[0:-1]])
            # print(qa_label, vars)
            vars_ranges = dict()
            for v in vars:
                # print(t)
                vrange = ArchSpace.intersect_intervals_from_paths([t[0:-1]], v, min_bound=0, max_bound=None)
                # print("\t",v,vrange)
                if vrange is not None:
                    vars_ranges[v] = {'min': vrange[0], 'max': vrange[1]}
                else:
                    print("WARNING: Skipping rule", t)
            if len(vars_ranges.keys()) > 0:
                cart_boxes[qa_label].append(vars_ranges)
            # else:
            #     print("WARNING: No variables for:", qa_label)

        to_be_removed = []
        for qa_label in cart_boxes.keys(): # First box (there could be more boxes for the same label)
            # print(qa_label, cart_boxes[qa_label])
            if len(cart_boxes[qa_label]) > 0:
                cart_boxes[qa_label] = cart_boxes[qa_label][0]
            else:
                to_be_removed.append(qa_label)
        for qa in to_be_removed:
            cart_boxes.pop(qa, None)
        
        return triple_rules, cart_boxes, cart_alg
    
    def split_dataset(self, test_size=0.25, random_state=42, remove_outliers=None):
        l = [self.experiments_df_, self.outcomes_df_]
        concat_df = pd.concat(l, axis=1).reset_index(drop=True)
        labels = self._get_cluster_labels()
        print(concat_df.shape, len(labels))

        instances_to_remove = set()
        # https://stackoverflow.com/questions/23199796/detect-and-exclude-outliers-in-a-pandas-dataframe
        if remove_outliers is not None:
            print("Removing outliers ...", remove_outliers)
            for qa in self.ALL_LABELS.keys():
                outliers = concat_df[np.abs(stats.zscore(concat_df[qa])) > remove_outliers]
                instances_to_remove.update(list(outliers.index))
            print("Instances to remove (outliers):", instances_to_remove)
        
        for qat, n in self.available_tradeoffs_.items():
            if n < 2:
                idx = labels.index(qat)
                print("Warning:", qat, "not enough instances for splitting", idx)
                instances_to_remove.add(idx)
        print("Instances to remove (not enough tradeoffs):", instances_to_remove)
        # labels = list(filter(lambda x: labels.index(x) not in instances_to_remove, labels))
        labels = [labels[x] for x in pd.Series(labels).index if x not in instances_to_remove]
        print(set(labels))
        concat_df.drop(index=list(instances_to_remove), inplace=True)
        print(concat_df.shape, len(labels))

        # concat_df.reset_index(drop=True, inplace=True)
        X_train, X_test, y_train, y_test = train_test_split(concat_df, labels, stratify=labels, test_size=test_size, random_state=random_state)
        y_train = pd.DataFrame([x.split(',') for x in y_train], columns=list(self.outcomes_df_.columns))
        y_test = pd.DataFrame([x.split(',') for x in y_test], columns=list(self.outcomes_df_.columns))
       
        return X_train.reset_index(drop=True), X_test.reset_index(drop=True), y_train, y_test
    
    def import_dataset(self, X, y):
        # It substitutes the current dataset with the new one
        self.experiments_df_ = X[self.experiments_df_.columns]
        self.outcomes_df_ = X[self.outcomes_df_.columns]
        self.discrete_outcomes_df_ = y
        self.available_tradeoffs_ = ArchSpace.get_tradeoffs(y)
    
    def compute_robustness_matrix(self):
        main_dict = dict()
        for exp in tqdm(self.get_configurations()):
            # print(exp)
            main_dict[str(exp)] = dict()
            for qa in self.available_tradeoffs_.keys():
                main_dict[str(exp)][qa], _ = self.compute_robustness(exp, qa)
        
        df = pd.DataFrame.from_dict(main_dict, orient='columns').sort_index()
        return df

# --------------------------------------------------------------


def show_pairplot(space, kind='sns', title='', group='scenario', legend=False, filename=None):
    
        experiments = space.experiments_df_
        outcomes = dict()
        for col in space.OUTPUTS:
            outcomes[col] = np.array(space.outcomes_df_[col].tolist())

        if kind == 'sns':
            data = pd.DataFrame(outcomes)
            data['scenario'] = experiments['scenario'].astype(str)
            data['policy'] = experiments['policy'].astype(str)
            g = sns.pairplot(data, hue=group, vars=list(outcomes.keys()), corner=True, 
                             plot_kws={'alpha':0.55}, diag_kws={"linewidth": 0, "fill": False})
            for i, y_var in enumerate(g.y_vars):
                for j, x_var in enumerate(g.x_vars):
                    if x_var == y_var:
                        g.axes[i, j].set_visible(False)
            if not legend:
                g._legend.remove()
            g.fig.suptitle(title)
            if filename is not None:
                plt.savefig(filename)
            plt.show()
        if kind == 'matplotlib':
            data = experiments.copy()
            data['scenario'] = experiments['scenario'].astype(str)
            data['policy'] = experiments['policy'].astype(str)
            fig, axes = pairs_plotting.pairs_scatter(data, outcomes, group_by=group,legend=legend)
            fig.set_size_inches(8,8)
            fig.suptitle(title)
            if filename is not None:
                plt.savefig(filename)
            plt.show()
    
def show_feature_scores(space, title='', filename=None, policy=None, size=(6,6), parameters=None):
        indices = None
        if policy is not None:
            indices = space.experiments_df_[space.experiments_df_['policy'] == policy].index
        if indices is None:
            experiments = space.experiments_df_
            outcomes = space.outcomes_df_
        else:
            experiments = space.experiments_df_.loc[indices][space.CONFIGURATION_PARAMETERS[policy]]
            outcomes = space.outcomes_df_.loc[indices]
            print("Instances for configuration:", experiments.shape)
        
        if parameters is not None:
            experiments = experiments[parameters]
        
        outcomes_dict = dict()
        for col in space.OUTPUTS:
            outcomes_dict[col] = np.array(outcomes[col].tolist())
        
        fs = feature_scoring.get_feature_scores_all(experiments, outcomes_dict) 

        fig, ax = plt.subplots(figsize=size)
        g = sns.heatmap(fs, cmap='viridis', annot=True, ax=ax)
        g.set(ylabel=None)
        if policy is None:
            plt.title(title)
        else:
            plt.title(title + " policy= " + str(policy))
        if filename is not None:
            plt.savefig(filename)
        plt.show()

        return fs
    
def show_outcome_distributions(space, n_bins=100, labels=3, filename=None, mins_maxs=(None,None), 
                                   size=(8,8), policy=None, colors = ['b', 'r', 'g', 'y']):
        qas = list(space.outcomes_df_.columns)

        indices = None
        if policy is not None:
            indices = space.experiments_df_[space.experiments_df_['policy'] == policy].index
            # print("Indices for policy", policy, ":", indices)
            if indices.shape[0] == 0:
                print("Warning: No experiments found for policy", policy)
            else:
                print("Experiments for policy", policy, ":", indices.shape[0])
    
        qa_range = dict()
        fig, axs = plt.subplots(len(qas), 1, sharey=True, tight_layout=True, figsize=size)
        for idx, qa in enumerate(qas):
            # Filter the outcomes for a particular policy (indices)
            if indices is not None:
                col = space.outcomes_df_.loc[indices, qa]
            else:
                col = space.outcomes_df_[qa]
            min_max = (None,None)
            if mins_maxs != (None,None):
                min_max = mins_maxs[idx]
                #print(min_max)
            #col_range = get_distribution_bins(col, labels, min_max)
            col_range = ArchSpace.get_bins(col, labels, min_max)
            print(qa, col_range)
            qa_range[qa] = col_range
            if min_max != (None,None):
                col = col.tolist()
                col.append(min_max[0])
                col.append(min_max[1])
            axs[idx].hist(col, bins=n_bins, color=colors[idx])
            axs[idx].set_title(qa)
            for xc in col_range:
                axs[idx].axvline(x=xc, color='gray', linestyle='dashed')

        if filename is not None:
            plt.savefig(filename)
        plt.show()
        return qa_range

def show_tradeoff_distribution_2d(counter_frequency_labels, exp_tradeoffs=None, size=(6,12), axes=None):        
        x = set()
        y = set()
        for lb in counter_frequency_labels.keys():
            pair = lb.split(',')
            x.add(pair[0])
            y.add(pair[1])
        freq_dict = dict()
        highlights = []
        for idx, i in enumerate(x):
            freq_dict[i] = dict()
            for idj, j in enumerate(y):
                k = i+','+j
                freq_dict[i][j] = counter_frequency_labels[k]
                if (exp_tradeoffs is not None) and (k in exp_tradeoffs.keys()):
                    # print((idx, idj), k, exp_tradeoffs[k])
                    highlights.append((idx, idj))
        
        df = pd.DataFrame(freq_dict)
        
        fig, ax = plt.subplots(figsize=size)
        if exp_tradeoffs is None:
            g = sns.heatmap(df, cmap='magma_r', annot=True, ax=ax, vmin=0, vmax=1)
        else:
            g = sns.heatmap(df, cmap='magma_r', annot=True, ax=ax, vmin=0, vmax=1)
            for h in highlights:
                ax.add_patch(Rectangle(h, 1, 1, fill=False, edgecolor='blue', lw=2))

        if axes is not None:
            g.set(xlabel=axes[0])
            g.set(ylabel=axes[1])

        plt.title("Distribution (frequency) of tradeoffs")
        plt.show()

        return df

def show_tradeoff_distribution_histogram(counter_frequency_labels, exp_tradeoffs=None, size=(6,12)):
        
        # print('Categories:', len(counter_frequency_labels))
        fig = plt.figure(figsize=size)
        vals = np.array(list(counter_frequency_labels.values()))
        if exp_tradeoffs is None:
            max_vals = np.array(vals.max())
        else:
            max_vals = np.array([v for qat, v in counter_frequency_labels.items() if (qat in exp_tradeoffs.keys())])
            #max_vals = np.array(list(exp_counter_frequency_labels.values()))
            # print("max_vals:", max_vals)
        plt.barh(list(counter_frequency_labels.keys()), counter_frequency_labels.values(), 
            color=np.where(np.isin(vals,max_vals), '#ff9999','cadetblue'))
        #plt.xticks(rotation=90)
        plt.title("Distribution (frequency) of tradeoffs")
        plt.show()

def show_tradeoff_distribution(space, configuration=None, in2D=False, size=(6,12)):

        #print(self.ALL_TRADEOFF_LABELS)
        counter_frequency_labels = ArchSpace._count_tradeoffs(space.available_tradeoffs_, all_labels=space.ALL_TRADEOFF_LABELS)
        # print(len(available_tradeoffs), available_tradeoffs)
        # print(len(counter_frequency_labels), counter_frequency_labels)
        
        exp_tradeoffs = None
        if configuration is not None:
            _, _, discrete_exp_out_df = space.get_experiment(configuration)
            exp_tradeoffs = ArchSpace.get_tradeoffs(discrete_exp_out_df)
            print(configuration, exp_tradeoffs)
            print("Instances for configuration:", discrete_exp_out_df.shape)
            #exp_counter_frequency_labels = ArchSpace._count_tradeoffs(exp_tradeoffs, all_tradeoffs=False)
            #print(exp_counter_frequency_labels)

        # Show using the different layouts
        if in2D:
            _ = show_tradeoff_distribution_2d(counter_frequency_labels, exp_tradeoffs, size, axes=space.OUTPUTS)
        else:
            show_tradeoff_distribution_histogram(counter_frequency_labels, exp_tradeoffs, size)

def show_configuration_space2D(df, model2D=None, n_components=2, random_state=42, alpha=1.0,
                               size=(6,6), no_axis=True, color_key=None, marker='s', marker_size=50):
    if model2D is None:
        embeddings = df.drop(['scenario', 'policy', 'model'], axis=1)
        #le = LabelEncoder()
        #embeddings['label'] = le.fit_transform(df['policy'].values)
        model2D = umap.UMAP(n_components=n_components, random_state=random_state).fit(embeddings)
    
    plt.figure(figsize=size)
    xy_df = pd.DataFrame(model2D.embedding_, columns=['x', 'y'])
    xy_df['decision'] = df['policy'].astype('category')
    if color_key is None:
        color_key = 'tab10'
    g = sns.scatterplot(data=xy_df, x='x', y='y', alpha=alpha, hue='decision', palette=color_key, marker=marker, s=marker_size)
    if no_axis:
        g.set(xticklabels=[], yticklabels=[])  
        g.set(xlabel=None, ylabel=None)
    plt.show()
    return xy_df

def show_quality_attribute_space2D(df, min_max_ranges=None, size=(6,6), show_grid=True, alpha=1.0, no_axis=False, dpi=None,
                                        color_key=None, marker='o', marker_size=70, qualities=['response_time', 'utilization']):
    xy_df = pd.DataFrame(df.values, columns= qualities + ['decision'])
    # xy_df['decision'] = df['policy'].astype('category')
    
    if color_key is None:
        color_key = 'tab10'
    plt.figure(figsize=size)
    g = sns.scatterplot(data=xy_df, x=qualities[0], y=qualities[1], alpha=alpha, hue='decision', 
                            palette=color_key, s=marker_size, marker=marker)
    if show_grid:
        vertical = ArchSpace.get_bins(df[qualities[0]],n=3, min_max=min_max_ranges[0])
        for v in vertical:
            plt.axvline(v, color='lightgray', linestyle='--')
        horizontal = ArchSpace.get_bins(df[qualities[1]],n=3, min_max=min_max_ranges[1])
        for h in horizontal:
            plt.axhline(h, color='lightgray', linestyle='--')
    if no_axis:
        g.set(xticklabels=[], yticklabels=[])  
        g.set(xlabel=None, ylabel=None)
    plt.legend(loc='upper right')
    if dpi is not None:
        print("Generating PDF image...", dpi)
        plt.savefig('quality_attribute_space.pdf', dpi=dpi, format='pdf')
    plt.show()


class GatewayAggregation(ArchSpace):

    INPUT_PARAMETERS = ['N_A', 'N_B', 'r_Z_A', 'r_Z_B', 'r_A_gw', 'r_B_gw', 'r_A_s1', 'r_B_s1', 'r_A_s2', 'r_B_s2', 'r_A_s3', 'r_B_s3']
    OUTPUTS = ['response_time', 'utilization'] #, 'utilization_s1', 'utilization_s2', 'utilization_s3']
    N = 25

    def load_results(self, path):
        # Parsing simulation file and renaming some columns
        df = pd.read_csv(path)
        print(df.shape) 
        # print(df.columns)
        # filter_cols = ['N_A', 'N_B', 'r_Z_A', 'r_Z_B', 'r_A_gw', 'r_B_gw', 'r_A_s1', 'r_B_s1', 'r_A_s2', 'r_B_s2', 'r_A_s3', 'r_B_s3', 
        #       'sim_time_sec', 'R0', 'X0', 'Ugw', 'Us1', 'Us2', 'Us3']
        filter_cols = self.INPUT_PARAMETERS + ['R0', 'Ugw']
        df = df[filter_cols]
        df.rename(columns={'R0': 'response_time', 'Ugw': 'utilization'}, inplace=True)
        df.sort_values(by='N_A', inplace=True)
        df.reset_index(inplace=True, drop=True)

        # df['N_A/N'] = df['N_A'] / self.N

        self.experiments_df_ = df[self.INPUT_PARAMETERS].copy()
        self.experiments_df_.index.name = 'scenario'
        self.experiments_df_.reset_index(inplace=True)
        self.experiments_df_['model'] = 'gateway_aggregation'
        self.experiments_df_['policy'] = 0 #np.nan #float('nan')
        # The only policy (i.e. design decision) is the selection of 3 services being aggregated by the gateway
        self.experiments_df_['policy'] = self.experiments_df_['policy'].astype('category')

        self.outcomes_df_ = df[self.OUTPUTS].copy()

        return self.experiments_df_, self.outcomes_df_

    def get_configurations(self):
        all_configs = set(self.experiments_df_['policy'])
        return list(all_configs) #[[x] for x in all_configs]


class GatewayOffloading(ArchSpace):

    INPUT_PARAMETERS = ['N_A', 'N_B', 'r_Z_A', 'r_Z_B', 'r_gw', 'r_A_s1', 'r_B_s2', 'r_B_s3']
    INPUT_PARAMETERS1 = ['Z_A', 'Z_B', 'S_gw', 's_A_s1', 's_B_s2', 's_B_s3']
    OUTPUTS = ['response_time', 'utilization'] #['sim_time_sec', 'response_time', 'utilization_gw']
    S_GW_DECISIONS = [0, 5, 10]

    CONFIGURATIONS = {
        0: 'no-offloading',
        5: 'short-services-offloaded',
        10: 'long-services-offloaded'
    }

    CONFIGURATION_PARAMETERS = {
        'no-offloading': INPUT_PARAMETERS,
        'short-services-offloaded': INPUT_PARAMETERS,
        'long-services-offloaded': INPUT_PARAMETERS
    }
    
    def load_results(self, path, reverse=False):
        # Parsing simulation file and renaming some columns
        df = pd.read_csv(path)
        print(df.shape) 
        # print(df.columns)
        # filter_cols = ['N_A', 'N_B', 'r_Z_A', 'r_Z_B', 'r_gw', 'r_A_s1', 'r_B_s2', 'r_B_s3', 
        #                'sim_time_sec', 'R0', 'X0', 'Ugw', 'Us1', 'Us2', 'Us3']
        filter_cols = self.INPUT_PARAMETERS + ['R0', 'Ugw']
        df = df[filter_cols]
        df.rename(columns={'R0': 'response_time', 'Ugw': 'utilization'}, inplace=True)
        df['S_gw'] = 1 / df['r_gw']
        if reverse:
            df['Z_A'] = 1 / df['r_Z_A']
            df['Z_B'] = 1 / df['r_Z_B']
            df['s_A_s1'] = (1 / df['r_A_s1']) + df['S_gw']
            df['s_B_s2'] = (1 / df['r_B_s2']) + df['S_gw']
            df['s_B_s3'] = (1 / df['r_B_s3']) + df['S_gw']
        df.sort_values(by=['N_A', 'S_gw'], inplace=True)
        # df ['N_A/N'] = df['N_A'] / (df['N_A'] + df['N_B'])
        # df.reset_index(inplace=True, drop=True)

        # print("Values for S_gw:", df['S_gw'].unique())
        list_dfs = []
        for s in df['S_gw'].unique():
            # print(s,int(s))
            if int(s) in self.S_GW_DECISIONS:
                if reverse:
                    temp = df[df['S_gw'] == s][self.INPUT_PARAMETERS + self.OUTPUTS + self.INPUT_PARAMETERS1].copy()
                else:
                    temp = df[df['S_gw'] == s][self.INPUT_PARAMETERS + self.OUTPUTS].copy()                
                temp['policy'] = self.CONFIGURATIONS[int(s)] #int(s) #np.nan #float('nan')
                temp['policy'] = temp['policy'].astype('category')
                # print("S_gw =", s, ":", temp.shape)
                # temp = temp.drop(temp[temp['r_gw'] > 100].index)
                list_dfs.append(temp)

        if reverse:
            self.experiments_df_ = pd.concat(list_dfs)[['policy', 'N_A', 'N_B'] + self.INPUT_PARAMETERS1]
        else:
            self.experiments_df_ = pd.concat(list_dfs)[['policy']]
        # self.experiments_df_ = # df[self.INPUT_PARAMETERS].copy()
        self.experiments_df_.index.name = 'scenario'
        self.experiments_df_.reset_index(inplace=True)
        self.experiments_df_['model'] = 'gateway_offloading'
        # self.experiments_df_.drop('S_gw', axis=1, inplace=True)
        print("filtered size:", self.experiments_df_ .shape)
        if reverse:
            self.INPUT_PARAMETERS = self.INPUT_PARAMETERS1
            for k in self.CONFIGURATION_PARAMETERS.keys():
                self.CONFIGURATION_PARAMETERS[k] = self.INPUT_PARAMETERS1
        
        # self.experiments_df_['policy'] = 0 #np.nan #float('nan')
        # There are 3 possible policies (i.e, design decisions) here, depending on the value of S_gw 
        # S_gw can be 0, 5 or 10ms, which means: no offloading, short services offloaded, or long services ofloaded
        # self.experiments_df_['policy'] = self.experiments_df_['policy'].astype('category')
        
        # self.outcomes_df_ = df[self.OUTPUTS].copy()
        self.outcomes_df_ = pd.concat(list_dfs)[self.OUTPUTS]
        self.outcomes_df_.reset_index(inplace=True, drop=True)

        return self.experiments_df_, self.outcomes_df_
    
    def get_configurations(self):
        all_configs = set(self.experiments_df_['policy'])
        return list(all_configs) #[[x] for x in all_configs]


class CQRS(ArchSpace):

    # Constants
    Nread = 90
    Nwrite = 10
    Zread = 10

    COMMON_PARAMETERS = ['N_read', 'N_write', 'r_Z_read', 'r_Z_write']
    INPUT_PARAMETERS_SW = COMMON_PARAMETERS + ['C_DB' ,'r_DB_read', 'r_DB_write'] #, 'U_DBwrite', 'U_DBread']
    INPUT_PARAMETERS_HW = COMMON_PARAMETERS + ['C_DB_read', 'r_DB_read' , 'C_DB_write', 'r_DB_write'] #,'U_DBwrite', 'U_DBread']
    # OUTPUTS = ['response_time', 'read_response_time', 'write_response_time', 'utilization'] 
    OUTPUTS = ['response_time', 'utilization'] 
    # S_GW_DECISIONS = [0, 5, 10]

    CONFIGURATION_PARAMETERS = {
        'sw': INPUT_PARAMETERS_SW,
        'hw': INPUT_PARAMETERS_HW
    }

    # ALL_TRADEOFF_LABELS = natsorted([','.join(t) for t in itertools.product(*[ArchSpace.RESPONSE_TIME_LABELS, 
    #                                     ArchSpace.RESPONSE_TIME_LABELS, ArchSpace.RESPONSE_TIME_LABELS, ArchSpace.UTILIZATION_LABELS])])

    # ALL_LABELS = dict()
    # ALL_LABELS['response_time'] = ArchSpace.RESPONSE_TIME_LABELS
    # ALL_LABELS['read_response_time'] = ArchSpace.RESPONSE_TIME_LABELS
    # ALL_LABELS['write_response_time'] = ArchSpace.RESPONSE_TIME_LABELS
    # ALL_LABELS['utilization'] = ArchSpace.UTILIZATION_LABELS

    def __init__(self):
        # super().__init__()
        self.experiments_df_ = None
        self.outcomes_df_ = None
    
    def load_results(self, path, kind='sw'):
        # Parsing simulation file and renaming some columns
        df = pd.read_csv(path)
        print(df.shape) 
        # print(df.columns)
        #filter_cols = ['N_read', 'N_write', 'r_Z_read' ,'r_Z_write','r_DB_read', 'r_DB_write', 'U_DBwrite', 'U_DBread',
        #               'sim_time_sec', 'R0', 'R0_write', 'R0_read']
        if kind == 'sw':
            filter_cols = self.INPUT_PARAMETERS_SW + ['U_DBwrite', 'U_DBread', 'R0']
        if kind == 'hw':
            filter_cols = self.INPUT_PARAMETERS_HW + ['U_DBwrite', 'U_DBread', 'R0'] 

        df = df[filter_cols]
        # df.rename(columns={'R0': 'response_time', 'R0_write': 'write_response_time', 'R0_read': 'read_response_time'}, inplace=True)
        df.rename(columns={'R0': 'response_time'}, inplace=True)
        df['utilization'] = (df['U_DBwrite'] + df['U_DBread'])/2
        # print(df.columns)
        df['Z_read'] = 1 / df['r_Z_read']
        df['Z_write'] = 1 / df['r_Z_write']
        # df = df[(df['N_read']==CQRS.Nread) & (df['N_write']==CQRS.Nwrite) & (df['Z_read']==CQRS.Zread)]
        print(df.shape) 
        #df.sort_values(by='Z_write', inplace=True)

        if self.experiments_df_ is None: # Nothing loaded yet
            if kind == 'sw': # This is only for compatiblity of parameters between the 2 options
                self.experiments_df_ = df[self.INPUT_PARAMETERS_SW].copy()
                self.experiments_df_['C_DB_read'] = df['C_DB']
                self.experiments_df_['C_DB_write'] = df['C_DB']
            elif kind == 'hw': # This is only for compatiblity of parameters between the 2 options
                self.experiments_df_ = df[self.INPUT_PARAMETERS_HW].copy()
                self.experiments_df_['C_DB'] = df['C_DB_read'] + df['C_DB_write']
            else:
                print("Incorrect design option:", kind)
                return None, None
            
            self.experiments_df_['policy'] = kind
            self.experiments_df_['policy'] = self.experiments_df_['policy'].astype('category')
            self.experiments_df_.index.name = 'scenario'
            self.experiments_df_.reset_index(inplace=True)
            self.experiments_df_['model'] = 'cqrs'
            print("initial size:", self.experiments_df_ .shape)
            
            self.outcomes_df_ = df[self.OUTPUTS].copy()
            self.outcomes_df_.reset_index(inplace=True, drop=True)
        else:
            if kind == 'sw': # This is only for compatiblity of parameters between the 2 options
                temp = df[self.INPUT_PARAMETERS_SW].copy()
                temp['C_DB_read'] = df['C_DB']
                temp['C_DB_write'] = df['C_DB']
            elif kind == 'hw': # This is only for compatiblity of parameters between the 2 options
                temp = df[self.INPUT_PARAMETERS_HW].copy()
                temp['C_DB'] = df['C_DB_read'] + ['C_DB_write']
            else:
                print("Incorrect design option:", kind)
                return None, None
            
            temp['policy'] = kind
            temp['policy'] = temp['policy'].astype('category')
            temp.index.name = 'scenario'
            temp['model'] = 'cqrs'
            temp.reset_index(inplace=True)
            print("appended size:", temp.shape)
            
            self.experiments_df_ = pd.concat([self.experiments_df_, temp])
            self.experiments_df_.reset_index(inplace=True, drop=True)
            print("final size:", self.experiments_df_.shape)
            
            out_df = df[self.OUTPUTS].copy()
            self.outcomes_df_ = pd.concat([self.outcomes_df_, out_df])
            self.outcomes_df_.reset_index(inplace=True, drop=True)

        return self.experiments_df_, self.outcomes_df_
    
    def get_configurations(self):
        all_configs = set(self.experiments_df_['policy'])
        return list(all_configs) #[[x] for x in all_configs]


class PipesAndFilters(ArchSpace):

    COMMON_PARAMETERS = ['N_A', 'N_B', 'r_Z_A', 'r_Z_B']

    INPUT_PARAMETERS1 = COMMON_PARAMETERS + ['c_A_T1','c_A_T2','c_A_T3','c_B_T1','c_B_T2','c_B_T4',
                         'r_A_T1','r_A_T2','r_A_T3','r_B_T1','r_B_T2','r_B_T4']
    INPUT_PARAMETERS2 = COMMON_PARAMETERS + ['c_T1','c_T2','c_T3','c_T4',
                         'r_A_T1','r_B_T1','r_A_T2','r_B_T2','r_A_T3','r_B_T4']

    OUTPUTS = ['response_time', 'throughput'] #['sim_time_sec', 'response_time', 'utilization_gw

    ALL_LABELS = dict()
    ALL_LABELS['response_time'] = ArchSpace.RESPONSE_TIME_LABELS
    ALL_LABELS['throughput'] = ArchSpace.UTILIZATION_LABELS

    N = 50

    CONFIGURATION_PARAMETERS = {
        'separated': INPUT_PARAMETERS1,
        'joint1': INPUT_PARAMETERS2,
        'joint2': INPUT_PARAMETERS2
    }

    def __init__(self):
        # super().__init__()
        self.experiments_df_ = None
        self.outcomes_df_ = None
    
    def load_results(self, path, kind='separated'):
        # Parsing simulation file and renaming some columns
        df = pd.read_csv(path)
        print(df.shape) 
        # print(df.columns)
        #filter_cols = ['N_A','N_B','r_Z_A','r_Z_B','c_T1','c_T2','c_T3','c_T4','r_A_T1','r_B_T1','r_A_T2','r_B_T2','r_A_T3','r_B_T4'
        #               'sim_time_sec', 'R0', 'X0']

        df.rename(columns={'R0': 'response_time', 'X0': 'throughput'}, inplace=True)

        # df = df[df['N_A']+df['N_B']==self.N]
        df.sort_values(by='N_A', inplace=True)
        # df['N_A/N'] = df['N_A'] / self.N

        if self.experiments_df_ is None: # Nothing loaded yet
            if (kind == 'separated'):
                self.experiments_df_ = df[self.INPUT_PARAMETERS1].copy()
                # self.experiments_df_['N_A/N'] = df['N_A/N']
            elif (kind == 'joint1') or (kind == 'joint2'):
                self.experiments_df_ = df[self.INPUT_PARAMETERS2].copy()
                # self.experiments_df_['N_A/N'] = df['N_A/N']
            else:
                print("Incorrect design option:", kind)
                return None, None
            
            self.experiments_df_['policy'] = kind
            self.experiments_df_['policy'] = self.experiments_df_['policy'].astype('category')
            self.experiments_df_.index.name = 'scenario'
            self.experiments_df_.reset_index(inplace=True)
            self.experiments_df_['model'] = 'pipes_and_filters'
            print("initial size:", self.experiments_df_ .shape)
            
            self.outcomes_df_ = df[self.OUTPUTS].copy()
            self.outcomes_df_.reset_index(inplace=True, drop=True)
        else:
            if (kind == 'separated'):
                self.experiments_df_ = df[self.INPUT_PARAMETERS1].copy()
                # self.experiments_df_['N_A/N'] = df['N_A/N']
            elif (kind == 'joint1') or (kind == 'joint2'):
                temp = df[self.INPUT_PARAMETERS2].copy()
                # temp['N_A/N'] = df['N_A/N']
            else:
                print("Incorrect design option:", kind)
                return None, None
            
            temp['policy'] = kind
            temp['policy'] = temp['policy'].astype('category')
            temp.index.name = 'scenario'
            temp['model'] = 'pipes_and_filters'
            temp.reset_index(inplace=True)
            print("appended size:", temp.shape)
            
            self.experiments_df_ = pd.concat([self.experiments_df_, temp])
            self.experiments_df_.fillna(0.0, inplace=True)
            self.experiments_df_.reset_index(inplace=True, drop=True)
            print("final size:", self.experiments_df_.shape)
            
            out_df = df[self.OUTPUTS].copy()
            self.outcomes_df_ = pd.concat([self.outcomes_df_, out_df])
            self.outcomes_df_.reset_index(inplace=True, drop=True)

        return self.experiments_df_, self.outcomes_df_
    
    def get_configurations(self):
        all_configs = set(self.experiments_df_['policy'])
        return list(all_configs) #[[x] for x in all_configs]


class AnticorruptionLayer(ArchSpace):

    INPUT_PARAMETERS = ['N_A', 'N_B', 'r_Z_A', 'r_Z_B', 'C_SS1', 'r_A_SS1', 'r_B_SS1', 'p_SS1toACL_A', 'p_SS1toRef_A',
                        'p_SS1toACL_B', 'p_SS1toRef_B', 'r_A_ACL', 'r_B_ACL', 'r_A_SS2', 'r_B_SS2']
    OUTPUTS = ['response_time', 'utilization'] # Utilization if for the ACL, although it could also be for SS1 and SS2

    N = 24
    # C = 1

    def load_results(self, path):
        # Parsing simulation file and renaming some columns
        df = pd.read_csv(path)
        print(df.shape) 
        # print(df.columns)
        # filter_cols = ['N_A', 'N_B', 'r_Z_A', 'r_Z_B', 'r_A_gw', 'r_B_gw', 'r_A_s1', 'r_B_s1', 'r_A_s2', 'r_B_s2', 'r_A_s3', 'r_B_s3', 
        #       'sim_time_sec', 'R0', 'X0', 'Ugw', 'Us1', 'Us2', 'Us3']
        filter_cols = self.INPUT_PARAMETERS + ['R0', 'Uacl']
        df = df[filter_cols]
        df.rename(columns={'R0': 'response_time', 'Uacl': 'utilization'}, inplace=True)
        # df = df[df['N_A']+df['N_B']==self.N]
        df.sort_values(by='N_A', inplace=True)
        df.reset_index(inplace=True, drop=True)

        self.experiments_df_ = df[self.INPUT_PARAMETERS].copy()
        self.experiments_df_.index.name = 'scenario'
        self.experiments_df_.reset_index(inplace=True)
        self.experiments_df_['model'] = 'anticorruption_layer'
        self.experiments_df_['policy'] = 0 #np.nan #float('nan')
        # The only policy (i.e. design decision) is the selection of 3 services being aggregated by the gateway
        self.experiments_df_['policy'] = self.experiments_df_['policy'].astype('category')

        self.outcomes_df_ = df[self.OUTPUTS].copy()

        return self.experiments_df_, self.outcomes_df_

    def get_configurations(self):
        all_configs = set(self.experiments_df_['policy'])
        return list(all_configs) #[[x] for x in all_configs]

