import numpy as np
import os
import pandas as pd
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import scipy.stats
import sys
import chemprop

LION_DIR = os.path.dirname(os.path.abspath(__file__))
SPLITS_BASE = os.path.join(LION_DIR, '0701_lion_lnpdb')
RESULTS_BASE = os.path.join(LION_DIR, 'results')

TARGET_COL = 'Experiment_value'
SMILES_COL = 'IL_SMILES'
CV_NUM = 5
PRED_SPLIT_VARIABLES = ['Experiment_ID']


def path_if_none(newpath):
    if not os.path.exists(newpath):
        os.makedirs(newpath)


def fix_extra_x(csv_path):
    """Convert booleans to 1/0 and fill NaNs for chemprop v1."""
    df = pd.read_csv(csv_path)
    for col in df.columns:
        if pd.api.types.is_bool_dtype(df[col]):
            df[col] = df[col].astype(int)
        elif df[col].dtype == object:
            df[col] = df[col].map({'True': 1, 'False': 0}).fillna(df[col])
    df = df.fillna(0)
    df.to_csv(csv_path, index=False)


def make_pred_vs_actual(split_folder, ensemble_size=CV_NUM, standardize_predictions=True):
    splits_dir = os.path.join(SPLITS_BASE, split_folder) if split_folder else SPLITS_BASE
    for cv in range(ensemble_size):
        fold_dir = os.path.join(splits_dir, f'fold_{cv}')
        results_dir = os.path.join(RESULTS_BASE, split_folder or '0701_lion_lnpdb', f'fold_{cv}')
        path_if_none(results_dir)

        output = pd.read_csv(os.path.join(fold_dir, 'test.csv'))
        metadata = pd.read_csv(os.path.join(fold_dir, 'test_metadata.csv'))
        output = pd.concat([metadata, output], axis=1)

        pred_cache = os.path.join(results_dir, 'predicted_vs_actual.csv')
        if os.path.exists(pred_cache):
            output = pd.read_csv(pred_cache)
        else:
            preds_path = os.path.join(fold_dir, 'preds.csv')
            if not os.path.exists(preds_path):
                arguments = [
                    '--test_path', os.path.join(fold_dir, 'test.csv'),
                    '--features_path', os.path.join(fold_dir, 'test_extra_x.csv'),
                    '--checkpoint_dir', fold_dir,
                    '--preds_path', preds_path,
                    '--smiles_columns', SMILES_COL,
                    '--target_columns', TARGET_COL,
                ]
                args = chemprop.args.PredictArgs().parse_args(arguments)
                chemprop.train.make_predictions(args=args)

            current_predictions = pd.read_csv(preds_path)
            current_predictions.drop(columns=[SMILES_COL], inplace=True)
            for col in current_predictions.columns:
                if standardize_predictions:
                    vals = current_predictions[col]
                    std = np.std(vals)
                    mean = np.mean(vals)
                    current_predictions[col] = [(v - mean) / std for v in vals]
                current_predictions.rename(columns={col: f'fold_{cv}_pred_{col}'}, inplace=True)
            output = pd.concat([output, current_predictions], axis=1)
            output.to_csv(pred_cache, index=False)


def analyze_predictions_cv(split_folder, ensemble_number=CV_NUM, min_values_for_analysis=10):
    results_dir = os.path.join(RESULTS_BASE, split_folder or '0701_lion_lnpdb')

    all_ns = {}
    all_pearson = {}
    all_pearson_p_val = {}
    all_kendall = {}
    all_spearman = {}
    all_rmse = {}
    all_unique = []

    for i in range(ensemble_number):
        fold_results = os.path.join(results_dir, f'fold_{i}', 'predicted_vs_actual.csv')
        preds_vs_actual = pd.read_csv(fold_results)
        pred_split_names = []
        for _, row in preds_vs_actual.iterrows():
            pred_split_name = '_'.join(str(row[v]) for v in PRED_SPLIT_VARIABLES)
            pred_split_names.append(pred_split_name)
        all_unique += list(set(pred_split_names))

    unique_pred_split_names = set(all_unique)
    for un in unique_pred_split_names:
        all_ns[un] = []
        all_pearson[un] = []
        all_pearson_p_val[un] = []
        all_kendall[un] = []
        all_spearman[un] = []
        all_rmse[un] = []

    for i in range(ensemble_number):
        fold_results = os.path.join(results_dir, f'fold_{i}', 'predicted_vs_actual.csv')
        preds_vs_actual = pd.read_csv(fold_results)
        pred_split_names = []
        for _, row in preds_vs_actual.iterrows():
            pred_split_name = '_'.join(str(row[v]) for v in PRED_SPLIT_VARIABLES)
            pred_split_names.append(pred_split_name)
        preds_vs_actual['Prediction_split_name'] = pred_split_names

        pred_col = f'fold_{i}_pred_{TARGET_COL}'

        for pred_split_name in unique_pred_split_names:
            data_subset = preds_vs_actual[
                preds_vs_actual['Prediction_split_name'] == pred_split_name
            ].reset_index(drop=True)

            actual = data_subset[TARGET_COL]
            pred = data_subset[pred_col] if pred_col in data_subset.columns else pd.Series([float('nan')] * len(actual))

            all_ns[pred_split_name].append(len(pred))

            if len(actual) >= min_values_for_analysis and pred_col in data_subset.columns:
                pearson = scipy.stats.pearsonr(actual, pred)
                spearman, _ = scipy.stats.spearmanr(actual, pred)
                kendall, _ = scipy.stats.kendalltau(actual, pred)
                rmse = np.sqrt(mean_squared_error(actual, pred))

                all_pearson[pred_split_name].append(pearson[0])
                all_pearson_p_val[pred_split_name].append(pearson[1])
                all_kendall[pred_split_name].append(kendall)
                all_spearman[pred_split_name].append(spearman)
                all_rmse[pred_split_name].append(rmse)

                plot_dir = os.path.join(results_dir, 'per_fold_plots', pred_split_name, f'fold_{i}')
                path_if_none(plot_dir)
                plt.figure()
                plt.scatter(pred, actual, color='black')
                plt.plot(np.unique(pred), np.poly1d(np.polyfit(pred, actual, 1))(np.unique(pred)))
                plt.xlabel(f'Predicted {TARGET_COL}')
                plt.ylabel(f'Actual {TARGET_COL}')
                plt.title(pred_split_name)
                plt.savefig(os.path.join(plot_dir, 'pred_vs_actual.png'))
                plt.close()
            else:
                for d in (all_pearson, all_pearson_p_val, all_kendall, all_spearman, all_rmse):
                    d[pred_split_name].append(float('nan'))

    crossval_results_path = os.path.join(results_dir, 'crossval_performance')
    path_if_none(crossval_results_path)

    pd.DataFrame.from_dict(all_ns).to_csv(os.path.join(crossval_results_path, 'n_vals.csv'), index=True)
    pd.DataFrame.from_dict(all_pearson).to_csv(os.path.join(crossval_results_path, 'pearson.csv'), index=True)
    pd.DataFrame.from_dict(all_pearson_p_val).to_csv(os.path.join(crossval_results_path, 'pearson_p_val.csv'), index=True)
    pd.DataFrame.from_dict(all_kendall).to_csv(os.path.join(crossval_results_path, 'kendall.csv'), index=True)
    pd.DataFrame.from_dict(all_spearman).to_csv(os.path.join(crossval_results_path, 'spearman.csv'), index=True)
    pd.DataFrame.from_dict(all_rmse).to_csv(os.path.join(crossval_results_path, 'rmse.csv'), index=True)

    # Pooled summary: mean across folds per experiment
    rows = []
    for exp_id in unique_pred_split_names:
        pearsons = [v for v in all_pearson[exp_id] if not np.isnan(v)]
        spearmans = [v for v in all_spearman[exp_id] if not np.isnan(v)]
        kendalls = [v for v in all_kendall[exp_id] if not np.isnan(v)]
        rmses = [v for v in all_rmse[exp_id] if not np.isnan(v)]
        rows.append({
            'Experiment_ID': exp_id,
            'n_folds': len(pearsons),
            'mean_pearson': np.mean(pearsons) if pearsons else float('nan'),
            'mean_spearman': np.mean(spearmans) if spearmans else float('nan'),
            'mean_kendall': np.mean(kendalls) if kendalls else float('nan'),
            'mean_rmse': np.mean(rmses) if rmses else float('nan'),
        })
    summary = pd.DataFrame(rows).sort_values('Experiment_ID')
    summary.to_csv(os.path.join(crossval_results_path, 'summary.csv'), index=False)
    print(f"Results written to {crossval_results_path}")


def main(argv):
    task_type = argv[1]
    split_folder = argv[2] if len(argv) > 2 else ''
    splits_dir = os.path.join(SPLITS_BASE, split_folder) if split_folder else SPLITS_BASE

    if task_type == 'train':
        epochs = 50
        for i, arg in enumerate(argv):
            if arg.replace('–', '-') == '--epochs':
                epochs = argv[i + 1]
        for cv in range(CV_NUM):
            fold_dir = os.path.join(splits_dir, f'fold_{cv}')
            for split in ('train', 'valid', 'test'):
                fix_extra_x(os.path.join(fold_dir, f'{split}_extra_x.csv'))
            arguments = [
                '--epochs', str(epochs),
                '--save_dir', fold_dir,
                '--seed', '42',
                '--dataset_type', 'regression',
                '--data_path', os.path.join(fold_dir, 'train.csv'),
                '--smiles_columns', SMILES_COL,
                '--target_columns', TARGET_COL,
                '--features_path', os.path.join(fold_dir, 'train_extra_x.csv'),
                '--separate_val_path', os.path.join(fold_dir, 'valid.csv'),
                '--separate_val_features_path', os.path.join(fold_dir, 'valid_extra_x.csv'),
                '--separate_test_path', os.path.join(fold_dir, 'test.csv'),
                '--separate_test_features_path', os.path.join(fold_dir, 'test_extra_x.csv'),
                '--data_weights_path', os.path.join(fold_dir, 'train_weights.csv'),
                '--loss_function', 'mse', '--metric', 'rmse',
            ]
            args = chemprop.args.TrainArgs().parse_args(arguments)
            chemprop.train.cross_validate(args=args, train_func=chemprop.train.run_training)

    elif task_type == 'analyze':
        make_pred_vs_actual(split_folder)
        analyze_predictions_cv(split_folder)


if __name__ == '__main__':
    main(sys.argv)
