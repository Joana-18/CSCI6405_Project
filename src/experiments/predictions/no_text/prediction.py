import os
import optuna
import pandas as pd
import numpy as np
from sklearn.metrics import f1_score, classification_report
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import OneHotEncoder
from sklearn.multioutput import MultiOutputClassifier
from xgboost import XGBClassifier
import csv
import random

# File to save best hyperparams
name = '/scratch/ssd004/scratch/amorim/src/CSCI6405/best_hyperparams.csv'
header = ['model', 'params', 
          'accuracy_mean_up', 'accuracy_std_up', 
          'f1_score_false_mean_up', 'f1_score_false_std_up', 
          'recall_false_mean_up', 'recall_false_std_up', 
          'precision_false_mean_up','precision_false_std_up', 
          'f1_score_true_mean_up','f1_score_true_std_up',
          'recall_true_mean_up', 'recall_true_std_up',  
          'precision_true_mean_up', 'precision_true_std_up',

          'accuracy_mean_down', 'accuracy_std_down', 
          'f1_score_false_mean_down', 'f1_score_false_std_down', 
          'recall_false_mean_down', 'recall_false_std_down', 
          'precision_false_mean_down','precision_false_std_down', 
          'f1_score_true_mean_down','f1_score_true_std_down',
          'recall_true_mean_down', 'recall_true_std_down',  
          'precision_true_mean_down', 'precision_true_std_down',

          'accuracy_mean_com', 'accuracy_std_com', 
          'f1_score_false_mean_com', 'f1_score_false_std_com', 
          'recall_false_mean_com', 'recall_false_std_com', 
          'precision_false_mean_com','precision_false_std_com', 
          'f1_score_true_mean_com','f1_score_true_std_com',
          'recall_true_mean_com', 'recall_true_std_com',  
          'precision_true_mean_com', 'precision_true_std_com']

WRITE_HEADER = True
if os.path.isfile(os.path.join(name)):
    WRITE_HEADER = False

# Load sets
def data_loader():
    ## Training
    train_data = pd.read_csv(r"/scratch/ssd004/scratch/amorim/src/CSCI6405/experiments/feature_importance/train.csv")
    train_data['title_sentiment'] = pd.Categorical(train_data.title_sentiment)
    train_data['text_sentiment'] = pd.Categorical(train_data.text_sentiment)
    train_data['subreddit'] = pd.Categorical(train_data.subreddit)
    train_data['category'] = pd.Categorical(train_data.category)
    train_data['karma_interval'] = pd.Categorical(train_data.karma_interval)
    train_data['upvote_popular'] = train_data.upvote_popular.astype(bool)
    train_data['downvote_popular'] = train_data.downvote_popular.astype(bool)
    train_data['comments_popular'] = train_data.comments_popular.astype(bool)

    ## Validation
    val_data = pd.read_csv(r"/scratch/ssd004/scratch/amorim/src/CSCI6405/experiments/feature_importance/val.csv")
    val_data['title_sentiment'] = pd.Categorical(val_data.title_sentiment)
    val_data['text_sentiment'] = pd.Categorical(val_data.text_sentiment)
    val_data['subreddit'] = pd.Categorical(val_data.subreddit)
    val_data['category'] = pd.Categorical(val_data.category)
    val_data['karma_interval'] = pd.Categorical(val_data.karma_interval)
    val_data['upvote_popular'] = val_data.upvote_popular.astype(bool)
    val_data['downvote_popular'] = val_data.downvote_popular.astype(bool)
    val_data['comments_popular'] = val_data.comments_popular.astype(bool)

    ## Test
    test_data = pd.read_csv(r"/scratch/ssd004/scratch/amorim/src/CSCI6405/experiments/feature_importance/test.csv")
    test_data['title_sentiment'] = pd.Categorical(test_data.title_sentiment)
    test_data['text_sentiment'] = pd.Categorical(test_data.text_sentiment)
    test_data['subreddit'] = pd.Categorical(test_data.subreddit)
    test_data['category'] = pd.Categorical(test_data.category)
    test_data['karma_interval'] = pd.Categorical(test_data.karma_interval)
    test_data['upvote_popular'] = test_data.upvote_popular.astype(bool)
    test_data['downvote_popular'] = test_data.downvote_popular.astype(bool)
    test_data['comments_popular'] = test_data.comments_popular.astype(bool)

    return train_data, val_data, test_data

# One-hot-encoding of categorical features
def one_hot_encoding(train_data, val_data, test_data):
    cols_to_encode = ['title_sentiment', 'text_sentiment', 'karma_interval']

    print(train_data['text_sentiment'].isna().sum(), flush=True)
    print(train_data['karma_interval'].isna().sum(), flush=True)
    ENCODER = OneHotEncoder(handle_unknown='ignore')
    ENCODER.fit(train_data[cols_to_encode])

    ## Training
    hot_encoded_train = pd.DataFrame(
        ENCODER.transform(train_data[cols_to_encode]).toarray())
    train_data_he = train_data.join(hot_encoded_train)
    train_data_he = train_data_he.drop(columns = cols_to_encode)
    train_data_he.columns = train_data_he.columns.astype(str)

    ## Validation
    hot_encoded_val = pd.DataFrame(
        ENCODER.transform(val_data[cols_to_encode]).toarray())
    val_data_he = val_data.join(hot_encoded_val)
    val_data_he = val_data_he.drop(columns = cols_to_encode)
    val_data_he.columns = val_data_he.columns.astype(str)

    ## Test
    hot_encoded_test = pd.DataFrame(
        ENCODER.transform(test_data[cols_to_encode]).toarray())
    test_data_he = test_data.join(hot_encoded_test)
    test_data_he = test_data_he.drop(columns = cols_to_encode)
    test_data_he.columns = test_data_he.columns.astype(str)

    return train_data_he, val_data_he, test_data_he

# Split into features and labels
def data_split(train_data_he, val_data_he, test_data_he):
    ## Training
    train_X = train_data_he.loc[:, ~train_data_he.columns.isin(['upvote_popular', 
                                                'downvote_popular', 
                                                'comments_popular', 
                                                'title', 'self_text', 
                                                'upvotes', 'downvotes',
                                                'comments', 'subreddit_subs',
                                                'category', 'subreddit'])]
    train_y = train_data_he[['upvote_popular', 'downvote_popular', 'comments_popular']]
    print(train_X.columns, flush=True)

    ## Validation
    val_X = val_data_he.loc[:, ~val_data_he.columns.isin(['upvote_popular', 
                                        'downvote_popular', 
                                        'comments_popular', 
                                        'title', 'self_text', 
                                        'upvotes', 'downvotes',
                                        'comments', 'subreddit_subs',
                                                'category', 'subreddit'])]
    val_y = val_data_he[['upvote_popular', 'downvote_popular', 'comments_popular']]
    print(val_X.columns, flush=True)

    ## Test
    test_X = test_data_he.loc[:, ~test_data_he.columns.isin(['upvote_popular', 
                                            'downvote_popular', 
                                            'comments_popular', 
                                            'title', 'self_text', 
                                            'upvotes', 'downvotes',
                                            'comments', 'subreddit_subs',
                                                'category', 'subreddit'])]
    test_y = test_data_he[['upvote_popular', 'downvote_popular', 'comments_popular']]
    print(test_X.columns, flush=True)

    return train_X, train_y, val_X, val_y, test_X, test_y

# Function to save best results per model
def save_results(result):
    # Source https://www.pythontutorial.net/python-basics/python-write-csv-file/
    with open(name, 'a+', encoding='UTF8', newline='') as f:
        writer = csv.writer(f)
        if WRITE_HEADER:
            writer.writerow(header)
        writer.writerow(result)

# Auxiliary function to keep track of metrics for all 10 runs w/ best parameters
def append_metrics(class_metrics_dict, metric, result):
    if metric in class_metrics_dict:
        class_metrics_dict[metric].append(result)
    else:
        class_metrics_dict[metric] = [result]
    return class_metrics_dict

# Prediction function
def predict(best_params, result, train_X, train_y, val_X, val_y,
            test_X, test_y, rf = False, mlp = False, xgboost = False):
    
    metrics_val = {}
    metrics_test = {}

    # 10 runs w/ best parameters w/ different random states
    for i in range(0, 10):
        rs = random.randint(0, 100)
        if rf:
            classifier = RandomForestClassifier(random_state = rs)
        elif mlp:
            classifier = MLPClassifier(random_state = rs)
        else:
            classifier = XGBClassifier(random_state = rs)
        classifier.set_params(**best_params)
        classifier.fit(train_X, train_y)

        # Validation results
        y_pred_val = classifier.predict(val_X)
        for i, class_name in enumerate(['upvote_popular', 'downvote_popular', 
                                        'comments_popular']):
            # Validation report
            output_dict = classification_report(val_y[class_name], 
                                                y_pred_val[:, i], 
                                                output_dict = True)  
            # Save run metrics per class
            if class_name not in metrics_val:
                metrics_val[class_name] = {}
            metrics_val[class_name] = append_metrics(
                metrics_val[class_name], 'accuracy', 
                output_dict['accuracy']
            )
            metrics_val[class_name] = append_metrics(
                metrics_val[class_name], 'f1_score_false', 
                output_dict['False']['f1-score']
            )
            metrics_val[class_name] = append_metrics(
                metrics_val[class_name], 'recall_false', 
                output_dict['False']['recall']
            )
            metrics_val[class_name] = append_metrics(
                metrics_val[class_name], 'precision_false', 
                output_dict['False']['precision']
            )
            metrics_val[class_name] = append_metrics(
                metrics_val[class_name], 'f1_score_true', 
                output_dict['True']['f1-score']
            )
            metrics_val[class_name] = append_metrics(
                metrics_val[class_name], 'recall_true',
                output_dict['True']['recall']
            )
            metrics_val[class_name] = append_metrics(
                metrics_val[class_name], 'precision_true', 
                output_dict['True']['precision']
            )

        # Test results
        y_pred_test = classifier.predict(test_X)
        for i, class_name in enumerate(['upvote_popular', 'downvote_popular', 
                                        'comments_popular']):
            # Test report
            output_dict = classification_report(test_y[class_name], 
                                                y_pred_test[:, i], 
                                                output_dict = True)                                              
            # Save run metrics per class
            if class_name not in metrics_test:
                metrics_test[class_name] = {}  
            metrics_test[class_name] = append_metrics(
                metrics_test[class_name], 'accuracy', 
                output_dict['accuracy']
            )
            metrics_test[class_name] = append_metrics(
                metrics_test[class_name], 'f1_score_false', 
                output_dict['False']['f1-score']
            )
            metrics_test[class_name] = append_metrics(
                metrics_test[class_name], 'recall_false', 
                output_dict['False']['recall']
            )
            metrics_test[class_name] = append_metrics(
                metrics_test[class_name], 'precision_false', 
                output_dict['False']['precision']
            )
            metrics_test[class_name] = append_metrics(
                metrics_test[class_name], 'f1_score_true', 
                output_dict['True']['f1-score']
            )
            metrics_test[class_name] = append_metrics(
                metrics_test[class_name], 'recall_true',
                output_dict['True']['recall']
            )
            metrics_test[class_name] = append_metrics(
                metrics_test[class_name], 'precision_true', 
                output_dict['True']['precision']
            )

    # Average Validation results
    for class_name in ['upvote_popular', 'downvote_popular', 'comments_popular']:
        items = metrics_val[class_name].items()
        print(f"\nValidation {class_name}", flush=True)

        for key, value in items:
            mean = np.round(np.mean(metrics_val[class_name][key]), 5)
            std = np.round(np.std(metrics_val[class_name][key]), 5)
            print(f"{key}: {mean} +/- {std}", flush=True)

            
    # Average Test results
    for class_name in ['upvote_popular', 'downvote_popular', 'comments_popular']:
        items = metrics_test[class_name].items()
        print(f"\nTest {class_name}", flush=True)

        for key, value in items:
            mean = np.round(np.mean(metrics_test[class_name][key]), 5)
            std = np.round(np.std(metrics_test[class_name][key]), 5)
            print(f"{key}: {mean} +/- {std}", flush=True)

            result.append(mean)
            result.append(std)
            
    # Save test metrics to csv
    save_results(result)

# Training function
def train(classifier, train_X, train_y, val_X, val_y):
    classifier.fit(train_X, train_y)
    
    # Evaluate on validation set
    y_pred = classifier.predict(val_X)
    f1 = f1_score(val_y, y_pred, average = 'weighted')
    return f1


# Random forest's objective function
def RF_objective_function(trial):
    # Params
    params = {
        'criterion' : trial.suggest_categorical('criterion', 
                                            ['gini', 'entropy', 'log_loss']),
        'max_depth' : trial.suggest_int('max_depth', 5, 10),
        'min_samples_split' : trial.suggest_int('min_samples_split', 2, 20),
        'n_estimators': trial.suggest_int('n_estimators', 100, 1000)    
    }
    
    print(params, flush=True)
    if W_MOC:
        clf = MultiOutputClassifier(
            RandomForestClassifier(**params, random_state = 42, n_jobs = -1, 
                                   class_weight='balanced'))
    else:
        clf = RandomForestClassifier(**params, random_state = 42, n_jobs = -1, 
                                     class_weight='balanced')
    f1 = train(clf, train_X, train_y, val_X, val_y)
    
    return f1
    

# MLP's objective function
def MLP_objective_function(trial):
    # Params
    # source: https://gist.github.com/toshihikoyanase/7e75b054395ec0a6dbb144a300862f60
    layers = []
    for i in range(1, 3):
        layers.append(trial.suggest_int(f'n_units_{i}', 1, 100))
    params = {
        'hidden_layer_sizes' : tuple(layers),
        'activation' : trial.suggest_categorical('activation', 
                                            ['relu', 'tanh']),
        'alpha' : trial.suggest_float('alpha', 1e-3, 1e-1, log=True),
        'solver' : trial.suggest_categorical('solver', ['sgd', 'adam']),
        'learning_rate' : trial.suggest_categorical('learning_rate', 
                                                ['constant', 'invscaling', 
                                                 'adaptive'])
    }
    
    print(params, flush=True)
    if W_MOC:
        clf = MultiOutputClassifier(MLPClassifier(**params, max_iter=100000, 
                                                random_state = 42))
    else:
        clf = MLPClassifier(**params, max_iter=100000, 
                                              random_state = 42)
    f1 = train(clf, train_X, train_y, val_X, val_y)

    return f1


# XGBoost's objective function
def XGBoost_objective_function(trial):
    # Params
    params = {
        'max_depth': trial.suggest_int('max_depth', 3, 10),
        'learning_rate': trial.suggest_float('learning_rate', 1e-3, 0.1, log=True),
        'n_estimators': trial.suggest_int('n_estimators', 100, 1000),
        'subsample': trial.suggest_float('subsample', 0.5, 1.0),
        'reg_alpha': trial.suggest_float('reg_alpha', 1e-7, 1e-1, log=True),
        'reg_lambda': trial.suggest_float('reg_lambda', 1e-7, 1e-1, log=True),
    }
    
    if W_MOC:
        clf = MultiOutputClassifier(XGBClassifier(**params, random_state = 42))
    else:
        clf = XGBClassifier(**params, random_state = 42)
    f1 = train(clf, train_X, train_y, val_X, val_y)
    return f1

# Random forest's prediction
def RF_prediction(train_X, train_y, val_X, val_y, test_X, test_y):
    if W_MOC:
        result = ['RF - No Text - MOC']
        study_name = 'rf-study-no_cat-sub_MOC'
        storage='sqlite:///CSCI6405ProjectFINAL_NoText_RF_MOC.db'
    else:
        result = ['RF - No Text - w/o MOC']
        study_name = 'rf-study-no_cat-sub_no-MOC'
        storage='sqlite:///CSCI6405ProjectFINAL_NoText_RF_NOMOC.db'
    print(f"============ {result} TRAIN ============", flush=True)
    # Hyperparameter tuning
    sampler = optuna.samplers.TPESampler(seed = 42)
    study = optuna.create_study(direction = 'maximize', sampler = sampler,
                                study_name=study_name, 
                                storage=storage, 
                                load_if_exists=True)
    study.optimize(RF_objective_function, n_trials = 500)

    # Best parms
    best_params = study.best_params
    result.append(best_params)

    print("============ RF VALIDATION ============", flush=True)
    print("Best parameters:", best_params, flush=True)
    best_value = study.best_value
    print("Best F1:", best_value, flush=True)

    # Fit model with best params
    print("============ RF TEST ============", flush=True)
    predict(best_params, result, train_X, train_y, val_X, val_y, 
            test_X, test_y, rf = True)


# MLP's prediction
def MLP_prediction(train_X, train_y, val_X, val_y, test_X, test_y):
    if W_MOC:
        result = ['MLP - No Text - MOC']
        study_name = 'mlp-study-no_cat-sub_MOC'
        storage='sqlite:///CSCI6405ProjectFINAL_NoText_MLP_MOC.db'
    else:
        result = ['MLP - No Text - w/o MOC']
        study_name = 'mlp-study-no_cat-sub_no-MOC'
        storage='sqlite:///CSCI6405ProjectFINAL_NoText_MLP_noMOC.db'
    print(f"============ {result} TRAIN ============", flush=True)
    # Hyperparameter tuning
    sampler = optuna.samplers.TPESampler(seed = 42)
    study = optuna.create_study(direction = 'maximize', sampler = sampler,
                                study_name=study_name, 
                                storage=storage, 
                                load_if_exists=True)
    study.optimize(MLP_objective_function, n_trials = 500)

    # Best parms
    best_params = study.best_params
    best_params['hidden_layer_sizes'] = tuple(
        (best_params['n_units_1'], best_params['n_units_2'])
    )
    best_params.pop('n_units_1', None)
    best_params.pop('n_units_2', None)
    result.append(best_params)

    print("============ MLP VALIDATION ============", flush=True)
    print("Best parameters:", best_params, flush=True)
    best_value = study.best_value
    print("Best F1:", best_value, flush=True)

    # Fit model with best params
    print("============ MLP TEST ============", flush=True)
    predict(best_params, result, train_X, train_y, val_X, val_y, 
            test_X, test_y, mlp = True)


# XGBoost's prediction
def XGBoost_prediction(train_X, train_y, val_X, val_y, test_X, test_y):
    if W_MOC:
        result = ['XGBoost - No Text - MOC']
        study_name = 'xgboost-study-no_cat-sub_MOC'
        storage='sqlite:///CSCI6405ProjectFINAL_NoText_XGB_MOC.db'
    else:
        result = ['XGBoost - No Text - w/o MOC']
        study_name = 'xgboost-study-no_cat-sub_no-MOC'
        storage='sqlite:///CSCI6405ProjectFINAL_NoText_XGB_NOMOC.db'
    print(f"============ {result} TRAIN ============", flush=True)
    # Hyperparameter tuning
    sampler = optuna.samplers.TPESampler(seed = 42)
    study = optuna.create_study(direction = 'maximize', sampler = sampler,
                                study_name=study_name, 
                                storage=storage, 
                                load_if_exists=True)
    study.optimize(XGBoost_objective_function, n_trials = 500)

    # Best parms
    best_params = study.best_params
    result.append(best_params)

    print("============ XGBoost VALIDATION ============", flush=True)
    print("Best parameters:", best_params, flush=True)
    best_value = study.best_value
    print("Best F1:", best_value, flush=True)

    # Fit model with best params
    print("============ XGBoost TEST ============", flush=True)
    predict(best_params, result, train_X, train_y, val_X, val_y, 
            test_X, test_y, xgboost = True)

W_MOC = False
# W_MOC = True

train_data, val_data, test_data = data_loader()
train_data_he, val_data_he, test_data_he = one_hot_encoding(train_data, 
                                                            val_data, test_data)
train_X, train_y, val_X, val_y, test_X, test_y = data_split(train_data_he, 
                                                            val_data_he, 
                                                            test_data_he)

# RF_prediction(train_X, train_y, val_X, val_y, test_X, test_y)
MLP_prediction(train_X, train_y, val_X, val_y, test_X, test_y)
# XGBoost_prediction(train_X, train_y, val_X, val_y, test_X, test_y)