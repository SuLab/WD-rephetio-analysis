import os
import sys

import hashlib
import argparse

import numpy as np
import pandas as pd

from tqdm import tqdm
from glmnet import LogitNet
from itertools import product
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, average_precision_score

import hetnet_ml.src.graph_tools as gt
from hetnet_ml.src.extractor import MatrixFormattedGraph
from hetnet_ml.src.processing import DegreeTransform, DWPCTransform

## Set arguments to run the script
parser = argparse.ArgumentParser(description='Run Machine Learning on Time-Based Wikidata Network')
parser.add_argument('data_dir', help="The directory of the source files for machine learning", type=str)
parser.add_argument('-g', '--gs_treat', help='Replace the TREATS edges in the network with those from the Gold Standard',
                    action='store_true')
parser.add_argument('-a', '--alpha', help="Set the alpha value for the ElasticNet Regression", type=float, default=0.1)
parser.add_argument('-w', '--weight', help="Set the damping exponent for DWPC extraction", type=float, default=0.4)
parser.add_argument('-m', '--multiplier', help="Multiplier for selecting the number of negatives for training.  Will"+
                        " use this factor of the number of positives", type=int, default=10)
parser.add_argument('-s', '--scoring', help='Scoring metric to use for ElasticNet regression', type=str,
                    default='recall')
parser.add_argument('-n', '--num_folds', help='The number of folds for the Cross Validation', type=int, default=5)
parser.add_argument('-e', '--seed', help='Seed for random split between folds', type=int, default=0)
parser.add_argument('-d', '--degree_features', help='Use Degree Features in the model', action='store_true')
parser.add_argument('-c', '--comp_xval', help='Use a Cross-Validation method where holdouts are based on drugs rather'+
                        ' than indication', action='store_true')
parser.add_argument('-l', '--max_length', help='The maximum Lenth for a metapath to be extracted', type=int, default=4)
parser.add_argument('-r', '--remove_similarity_mps', help='Remove Metapaths with CxXxCtD and CtDxXxD pattern.',
                    action='store_true')
parser.add_argument('-f', '--fold_number', help="Run only 1 fold of a 5-fold CV", type=int, default=None)
args = parser.parse_args()

## Define variables that will govern the network analysis
#remove_mps = ['CrCRrCtD', 'CtDmsMSmsD']
remove_mps = []
#remove_edges = ['CtD']
remove_edges = []
# Test params will be prepended to any output files if they differ from defaults
test_params = '' #'remove_all_sim'

## Unpack command line variables
target_edge_abv = 'CtD'
data_dir = args.data_dir
gs_treat = args.gs_treat
alpha = args.alpha
w = args.weight
negative_multiplier = args.multiplier
scoring = args.scoring
n_folds = args.num_folds
seed = args.seed
include_degree_features = args.degree_features
comp_xval = args.comp_xval
max_length = args.max_length
remove_similarity_mps = args.remove_similarity_mps
fold_number = args.fold_number
if scoring.lower() == 'none':
    scoring = None

# Convert load dir into an integer for a consistent random seed
ini_seed = int(hashlib.sha1(data_dir.encode()).hexdigest(), 16) % 2**16

# Out dir is based on this filename
out_dir = os.path.join('../2_pipeline', sys.argv[0].split('.')[0], 'out')

# Convert non-default parameters to output directory
for k in sorted(list(vars(args).keys())):
    # piecewise folds don't need to be considered a param
    if k == 'fold_number':
        continue
    if k == 'data_dir':
        dirname = os.path.split(vars(args)[k])
        dirname = dirname[-1] if dirname[-1] else dirname[-2]
        test_params += dirname+'.'
        continue
    v = vars(args)[k]
    if v != parser.get_default(k):
        if type(v) == bool:
            test_params += '{}.'.format(k)
        else:
            test_params += '{}-{}.'.format(k, v)
test_params = test_params.rstrip('.')
print('Non-default testing params: {}'.format(test_params))

n_jobs = 32

# Make sure the save directory exists, if not, make it
try:
    os.stat(os.path.join(out_dir, test_params))
except:
    os.makedirs(os.path.join(out_dir, test_params))

def remove_edges_from_gold_standard(to_remove, gs_edges):
    """
    Remove edges from the gold standard
    """
    remove_pairs = set([(tup.c_id, tup.d_id) for tup in to_remove.itertuples()])
    gs_tups = set([(tup.start_id, tup.end_id) for tup in gs_edges.itertuples()])

    remaining_edges = gs_tups - remove_pairs

    return pd.DataFrame({'start_id': [tup[0] for tup in remaining_edges],
                         'end_id': [tup[1] for tup in remaining_edges],
                         'type': 'TREATS_CtD'})

def add_percentile_column(in_df, group_col, new_col, cdst_col='prediction'):

    grpd = in_df.groupby(group_col)
    predict_dfs = []

    for grp, df1 in grpd:
        df = df1.copy()

        total = df.shape[0]

        df.sort_values(cdst_col, inplace=True)
        order = np.array(df.reset_index(drop=True).index)

        percentile = (order+1) / total
        df[new_col] = percentile

        predict_dfs.append(df)

    return pd.concat(predict_dfs, sort=False)

def find_drug_or_disease_similarity(mg):
    """ Finds paths with CxXxCtD or CtDxXxD pattern..."""
    remove_paths = []

    sk = mg.start_kind
    ek = mg.end_kind

    for mp, info in mg.metapaths.items():
        if info['length'] != 3:
            continue
        else:
            # CxXxCtD pattern:
            if (info['edges'][0].split(' - ')[0] == sk and
                    info['edges'][1].split(' - ')[-1] == sk and
                    info['standard_edge_abbreviations'][2] == 'CtD') \
            or (info['standard_edge_abbreviations'][0] == 'CtD' and # CtDxXxD pattern
                    info['edges'][1].split(' - ')[0] == ek and
                    info['edges'][2].split(' - ')[-1] == ek):

                remove_paths.append(mp)
    return remove_paths

def glmnet_coefs(glmnet_obj, X, f_names):
    """Helper Function to quickly return the model coefs and correspoding fetaure names"""
    l = glmnet_obj.lambda_best_[0]

    coef = glmnet_obj.coef_[0]
    coef = np.insert(coef, 0, glmnet_obj.intercept_)

    names = np.insert(f_names, 0, 'intercept')

    z_intercept = coef[0] + sum(coef[1:] * X.mean(axis=0))
    z_coef = coef[1:] * X.values.std(axis=0)
    z_coef = np.insert(z_coef, 0, z_intercept)

    return pd.DataFrame([names, coef, z_coef]).T.rename(columns={0:'feature', 1:'coef', 2:'zcoef'})

def get_scores(y_true, y_proba):
    return roc_auc_score(y_true, y_proba), average_precision_score(y_true, y_proba)

def print_scores(scores):
    print("ROC_AUC: {0:5.3f}\t Average_Precision: {1:5.3f}".format(scores[0], scores[1]))

def add_probas_to_df(fold_num, curr_proba, cd_df):
    cd_df['probas_{}'.format(fold_num)] = curr_proba
    cd_df = add_percentile_column(cd_df, group_col='c_id', new_col='c_percentile_{}'.format(i), cdst_col='probas_{}'.format(i))
    cd_df = add_percentile_column(cd_df, group_col='d_id', new_col='d_percentile_{}'.format(i), cdst_col='probas_{}'.format(i))
    return cd_df

def write_preds(cd_df, filename):
    cd_df.to_csv(filename, index=False)

def make_coef_df(n_folds, coefs):
    # Merge the Coefficient data
    coef_dfs = list()
    for i in range(n_folds):
        coef_dfs.append(coefs[i].copy())
        coef_dfs[i] = coef_dfs[i].set_index('feature')
        coef_dfs[i].columns = [l+'_{}'.format(i) for l in coef_dfs[i].columns]
    coef_df = pd.concat(coef_dfs, axis=1).reset_index()
    return coef_df

def print_top_coefs(coef_df, fold, num=5):
    col = 'coef_{}'.format(fold)
    print(coef_df.sort_values(col, ascending=False)[['feature', col]].head(num), end='\n\n')

def write_coefs(coef_df, filename):
    coef_df.to_csv(filename, index=False)


# Read input files
nodes = gt.remove_colons(pd.read_csv(os.path.join(data_dir, 'nodes.csv')))
edges = gt.remove_colons(pd.read_csv(os.path.join(data_dir, 'edges.csv')))

comp_ids = set(nodes.query('label == "Compound"')['id'])
dis_ids = set(nodes.query('label == "Disease"')['id'])

# We will use the TREATS edges within the graph as the training for the model
gs_edges = edges.query('type == "TREATS_CtD"').reset_index(drop=True)

# Setup for first run.  This info will all be contained in prediticions.csv and can be regenerated via load.
if fold_number is None or fold_number == 0:

    # Just look at compounds and diseases in the gold standard
    compounds = gs_edges['start_id'].unique().tolist()
    diseases = gs_edges['end_id'].unique().tolist()

    print('Based soley on gold standard...')
    print('{:,} Compounds * {:,} Diseases = {:,} CD Pairs'.format(len(compounds), len(diseases),
                                                                  len(compounds)*len(diseases)))
    # Add in some other edges... anything with a degree > 1... or a CtD edge
    compounds = set(compounds)
    diseases = set(diseases)

    print('Adding some more compounds and diseases....')
    # Do some magic to find nodes with degree > 1
    frac = 0.15
    mg = MatrixFormattedGraph(nodes, edges)
    first_comp = nodes.query('label == "Compound"')['id'].iloc[0]
    first_disease = nodes.query('label == "Disease"')['id'].iloc[0]
    comp_degrees = mg.extract_degrees(end_nodes=[first_disease])
    comp_degrees = comp_degrees.loc[:, ['compound_id']+[c for c in comp_degrees.columns if c.startswith('C')]]
    comp_degrees['total'] = comp_degrees[[c for c in comp_degrees.columns if c.startswith('C')]].sum(axis=1)
    dis_degrees = mg.extract_degrees(start_nodes=[first_comp])
    dis_degrees = dis_degrees.loc[:, ['disease_id']+[c for c in dis_degrees.columns if c.startswith('D')]]
    dis_degrees['total'] = dis_degrees[[c for c in dis_degrees.columns if c.startswith('D')]].sum(axis=1)

    compounds.update(set(comp_degrees.query('total > 1').sample(frac=frac)['compound_id']))
    diseases.update(set(dis_degrees.query('total > 1').sample(frac=frac)['disease_id']))

    compounds = list(compounds)
    diseases = list(diseases)

    print('Now comps and diseases')
    print('{:,} Compounds * {:,} Diseases = {:,} CD Pairs'.format(len(compounds), len(diseases),
                                                                  len(compounds)*len(diseases)))

    # Ensure all the compounds and diseases actually are of the correct node type and in the network
    compounds = [c for c in compounds if c in comp_ids]
    diseases = [d for d in diseases if d in dis_ids]

    # Make a DataFrame for all compounds and diseases
    # Include relevent compund info and treatment status (ML Label)
    cd_df = pd.DataFrame(list(product(compounds, diseases)), columns=['c_id', 'd_id'])
    id_to_name = nodes.set_index('id')['name'].to_dict()
    cd_df['c_name'] = cd_df['c_id'].apply(lambda i: id_to_name[i])
    cd_df['d_name'] = cd_df['d_id'].apply(lambda i: id_to_name[i])

    merged = pd.merge(cd_df, gs_edges, how='left', left_on=['c_id', 'd_id'], right_on=['start_id', 'end_id'])
    merged['status'] = (~merged['start_id'].isnull()).astype(int)
    cd_df = merged.loc[:, ['c_id', 'c_name', 'd_id', 'd_name', 'status']]

    # Set up training and testing for fold CV
    num_pos = len(gs_edges)
    num_neg = negative_multiplier*num_pos

    pos_idx = cd_df.query('status == 1').index
    neg_idx = cd_df.query('status == 0').sample(n=num_neg, random_state=ini_seed*(seed+1)).index
    # Trining set is all indictations and randomly selected negatives
    train_idx = pos_idx.union(neg_idx)

    # Initialize variable for coefficients
    coefs = []

    # Different train-test split paradigm, based on compounds/n_folds for each test set
    # Blinds the model to a particular compound for each fold of the cv
    cv = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)

    if comp_xval:
        # Needs to be array for indexing operations
        compounds = np.array(compounds)
        # set up dummy y-vals
        y = np.repeat(1, len(compounds))

        for i, (train, test) in enumerate(cv.split(compounds, y)):
            # Subset the compounds for this fold
            test_comps = compounds[test]

            # Testing and trainnig are broken up by compound:
            # Both positives and negatives in test set will be pulled from same compounds
            pos_holdout_idx = cd_df.query('status == 1 and c_id in @test_comps').index
            neg_holdout_idx = cd_df.loc[neg_idx].query('c_id in @test_comps').index

            print('Fold {}: {} Test Postives, {} Test negatives: {} ratio'.format(
                    i, len(pos_holdout_idx), len(neg_holdout_idx), len(neg_holdout_idx)/len(pos_holdout_idx)))

            holdout_idx = pos_holdout_idx.union(neg_holdout_idx)
            cd_df.loc[holdout_idx, 'holdout_fold'] = i

    else:
        train_df = cd_df.loc[train_idx]

        # Dummy xvalues and true results for y values used
        X = np.zeros(len(train_idx))
        y = train_df['status'].values

        # Just split indications randomly so equal number of pos and neg examples in each fold
        for i, (train, test) in enumerate(cv.split(X, y)):
            holdout_idx = train_df.iloc[test].index
            cd_df.loc[holdout_idx, 'holdout_fold'] = i
else:
    print('Reading precomputed probas....')
    cd_df = pd.read_csv(os.path.join(out_dir, test_params, 'predictions.csv'))
    compounds = cd_df['c_id'].unique().tolist()
    diseases = cd_df['d_id'].unique().tolist()
    train_idx = cd_df[~cd_df['holdout_fold'].isnull()].index
    coefs = []
    print('Processing previous coefficients...')
    coef_df = pd.read_csv(os.path.join(out_dir, test_params, 'coefs.csv'))
    for i in range(fold_number):
        coef = coef_df[['feature', 'coef_{}'.format(i), 'zcoef_{}'.format(i)]].copy()
        coef.columns = ['feature', 'coef', 'zcoef']
        coefs.append(coef)


# Currently no functionality... but TODO may want combine this with ability to load different gold standards.
if not gs_treat:
    print('Using the original TREATS edge from Wikidata')
else:
    print('Removing Wikidata TREATS edges and repalcing with those from Gold Standard')

    def drop_edges_from_list(df, drop_list):
        idx = df.query('type in @drop_list').index
        df.drop(idx, inplace=True)

    # Filter out any compounds and diseases wrongly classified
    gs_edges = gs_edges.query('start_id in @compounds and end_id in @diseases')
    # Remove the TREATs edge form edges
    drop_edges_from_list(edges, ['TREATS_CtD'])
    gs_edges['type'] = 'TREATS_CtD'

    column_order = edges.columns
    edges = pd.concat([edges, gs_edges], sort=False)[column_order].reset_index(drop=True)


print('{:,} Nodes'.format(len(nodes)))
print('{:,} Edges'.format(len(edges)))

print('{:,} Compounds * {:,} Diseases = {:,} CD Pairs'.format(len(compounds),
                                                              len(diseases), len(compounds)*len(diseases)))


target_edges = edges.query('type == "TREATS_CtD"').copy()
other_edges = edges.query('type != "TREATS_CtD"').copy()


# Option to only do a single fold per run of the script.
if fold_number is not None:
    r = (fold_number, fold_number+1)
else:
    r = (n_folds, )

for i in range(*r):

    print('Beginning fold {}'.format(i))
    fold_train_idx = cd_df.loc[train_idx].query('holdout_fold != @i').index

    if target_edge_abv not in remove_edges:
        to_remove = cd_df.query('holdout_fold == @i')
        to_keep = remove_edges_from_gold_standard(to_remove, target_edges)
        edges = pd.concat([other_edges, to_keep], sort=False)
        print('Training Positives: {}'.format(cd_df.query('holdout_fold != @i')['status'].sum()))
        print('Testing Positives: {}'.format(cd_df.query('holdout_fold == @i')['status'].sum()))
        mg = MatrixFormattedGraph(nodes, edges, 'Compound', 'Disease', w=w, max_length=max_length)
        blacklist = mg.generate_blacklist(target_edge_abv)
    else:
        edges = other_edges
        print('Training Positives: {}'.format(cd_df.query('holdout_fold != @i')['status'].sum()))
        print('Testing Positives: {}'.format(cd_df.query('holdout_fold == @i')['status'].sum()))
        # No target edge (TREATS) in network, so only need to build matricies once...
        if i == 0:
            mg = MatrixFormattedGraph(nodes, edges, 'Compound', 'Disease', w=w, max_length=max_length)
        blacklist = []

    # Convert graph to Matrices for ML feature extraction
    # Extract prior
    prior = mg.extract_prior_estimate('CtD', start_nodes=compounds, end_nodes=diseases)
    prior = prior.rename(columns={'compound_id': 'c_id', 'disease_id':'d_id'})

    if include_degree_features:
        # Extract degree Features
        degrees = mg.extract_degrees(start_nodes=compounds, end_nodes=diseases)
        degrees = degrees.rename(columns={'compound_id': 'c_id', 'disease_id':'d_id'})
        degrees.columns = ['degree_'+c if '_id' not in c else c for c in degrees.columns]
        # Generate blacklisted features and drop
        degrees.drop([b for b in blacklist if b.startswith('degree_')], axis=1, inplace=True)

    # Extract Metapath Features (DWPC)
    mp_blacklist = [b.split('_')[-1] for b in blacklist]
    if remove_similarity_mps:
        mp_blacklist = mp_blacklist + remove_mps + find_drug_or_disease_similarity(mg)
    # Option to define other Metapaths for Blacklisting
    mp_blacklist += [mp for mp, val in mg.metapaths.items() if
                        len(set(val['standard_edge_abbreviations']).intersection(set(remove_edges))) > 0]
    mps = [mp for mp in mg.metapaths.keys() if mp not in mp_blacklist]
    dwpc = mg.extract_dwpc(metapaths=mps, start_nodes=compounds, end_nodes=diseases, n_jobs=n_jobs)
    dwpc = dwpc.rename(columns={'compound_id': 'c_id', 'disease_id':'d_id'})
    dwpc.columns = ['dwpc_'+c if '_id' not in c else c for c in dwpc.columns]

    # Merge extracted features into 1 DataFrame
    print('Merging Features...')
    feature_df = pd.merge(cd_df, prior, on=['c_id', 'd_id'], how='left')
    if include_degree_features:
        feature_df = pd.merge(feature_df, degrees, on=['c_id', 'd_id'], how='left')
    feature_df = pd.merge(cd_df, dwpc, on=['c_id', 'd_id'], how='left')

    features = [f for f in feature_df.columns if f.startswith('degree_') or f.startswith('dwpc_')]
    if include_degree_features:
        degree_features = [f for f in features if f.startswith('degree_')]
    dwpc_features = [f for f in features if f.startswith('dwpc_')]

    # Transform Features
    if include_degree_features:
        dt = DegreeTransform()
    dwpct = DWPCTransform()

    X_train = feature_df.loc[fold_train_idx, features].copy()
    y_train = feature_df.loc[fold_train_idx, 'status'].copy()

    if include_degree_features:
        print('Transforming Degree Features')
        X_train.loc[:, degree_features] = dt.fit_transform(X_train.loc[:, degree_features])
    print('Tranforming DWPC Features')
    X_train.loc[:, dwpc_features] = dwpct.fit_transform(X_train.loc[:, dwpc_features])

    # Train our ML Classifer (ElasticNet Logistic Regressor)
    print('Training Classifier...')
    classifier = LogitNet(alpha=alpha, n_jobs=n_jobs, min_lambda_ratio=1e-8, n_lambda=150, standardize=True,
                  random_state=(ini_seed+1)*(seed+1), scoring=scoring)

    classifier.fit(X_train, y_train)

    coefs.append(glmnet_coefs(classifier, X_train, features))

    print('Positivie Coefficients: {}\nNegative Coefficitents: {}'.format(len(coefs[i].query('coef > 0')), len(coefs[i].query('coef < 0'))))
    coef_df = make_coef_df(i+1, coefs)
    print('Top 5 Coefficients:')
    print_top_coefs(coef_df, i, 5)

    # Get probs for all pairs
    print('Beginning extraction of all probabilities')
    print('Transforming all features...')
    if include_degree_features:
        feature_df.loc[:, degree_features] = dt.transform(feature_df.loc[:, degree_features])
    feature_df.loc[:, dwpc_features] = dwpct.transform(feature_df.loc[:, dwpc_features])

    print('Calculating Probabilities')
    all_probas = classifier.predict_proba(feature_df.loc[:, features])[:, 1]
    cd_df = add_probas_to_df(i, all_probas, cd_df)

    print('Metrics for test set:')
    fold_test_idx = cd_df.loc[train_idx].query('holdout_fold == @i').index
    y_true = cd_df.loc[fold_test_idx, 'status']
    y_probas = cd_df.loc[fold_test_idx, 'probas_{}'.format(i)]
    print_scores(get_scores(y_true, y_probas))

    print('Metrics for all predictions:')
    y_true = cd_df['status']
    y_probas = all_probas
    print_scores(get_scores(y_true, y_probas))

    print('Writing out data up to current fold\n')
    write_preds(cd_df, os.path.join(out_dir, test_params, 'predictions.csv'))
    write_coefs(coef_df, os.path.join(out_dir, test_params, 'coefs.csv'))

