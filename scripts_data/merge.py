import numpy as np
import os
import pandas as pd
from rdkit import Chem
from helpers import path_if_none, change_column_order
from config import (TARGET_COL_DELIVERY, TARGET_COL_TOXICITY,
                    COL_TYPE_Y, COL_TYPE_X, COL_TYPE_META, COL_TYPE_WEIGHT)


def z_score_normalize(df, col_name):
    if col_name in df.columns:
        df[f'unnormalized_{col_name.replace("quantified_", "")}'] = df[col_name]
        series = df[col_name]
        mean, std = series.mean(), series.std()
        if pd.isna(std) or std == 0:
            df[col_name] = 0.0
        else:
            df[col_name] = round(((series - mean) / std), 5)
    return df


def build_col_types(df, extra_x_variables, extra_x_categorical):
    y_val_cols = [TARGET_COL_TOXICITY, TARGET_COL_DELIVERY, 'smiles']
    col_type = {'Column_name': [], 'Type': []}
    for column in df.columns:
        col_type['Column_name'].append(column)
        if column in y_val_cols:
            col_type['Type'].append(COL_TYPE_Y)
        elif column in extra_x_variables:
            col_type['Type'].append(COL_TYPE_X)
        elif column in extra_x_categorical:
            col_type['Type'].append(COL_TYPE_META)
        elif column == 'Sample_weight':
            col_type['Type'].append(COL_TYPE_WEIGHT)
        else:
            col_type['Type'].append(COL_TYPE_META)
    return col_type


def apply_ohe(df, extra_x_variables, extra_x_categorical):
    df = df.copy()
    extra_x_variables = list(extra_x_variables)
    for x_cat in extra_x_categorical:
        dummies = pd.get_dummies(df[x_cat], prefix=x_cat)
        df = pd.concat([df, dummies], axis=1)
        extra_x_variables.extend(dummies.columns)
    return df, extra_x_variables


def merge_datasets(experiment_list, path_to_folders='../data_files', write_path='../data'):
    all_df = pd.DataFrame({})

    experiment_df = pd.read_csv(os.path.join(path_to_folders, 'experiment_metadata.csv'))
    if experiment_list is None:
        experiment_list = list(experiment_df.Experiment_ID)

    for folder in experiment_list:
        print("Processing:", folder)
        try:
            main_path = os.path.join(path_to_folders, folder, 'main_data.csv')
            main_temp = pd.read_csv(main_path)
        except FileNotFoundError:
            continue

        if 'Unnamed' in str(main_temp.columns):
            print(f'Warning: Unnamed columns in {folder}')

        main_temp = z_score_normalize(main_temp, TARGET_COL_DELIVERY)

        data_n = len(main_temp)
        form_path = os.path.join(path_to_folders, folder, 'formulations.csv')
        formulation_temp = pd.read_csv(form_path)

        try:
            ind_path = os.path.join(path_to_folders, folder, 'individual_metadata.csv')
            individual_temp = pd.read_csv(ind_path)
        except FileNotFoundError:
            individual_temp = pd.DataFrame({})

        if len(formulation_temp) == 1:
            formulation_temp = pd.concat([formulation_temp] * data_n, ignore_index=True)
        elif len(formulation_temp) != data_n:
            raise ValueError(f'Formulation length mismatch in {folder}')

        if len(individual_temp) == 1:
            individual_temp = pd.concat([individual_temp] * data_n, ignore_index=True)
        if not individual_temp.empty and len(individual_temp) != data_n:
            raise ValueError(f'Individual metadata length mismatch in {folder}')

        experiment_temp = experiment_df[experiment_df.Experiment_ID == folder]
        experiment_temp = pd.concat([experiment_temp] * data_n, ignore_index=True).reset_index(drop=True)

        cols_to_drop = [c for c in experiment_temp.columns if c in individual_temp.columns]
        experiment_temp = experiment_temp.drop(columns=cols_to_drop)

        folder_df = pd.concat([main_temp, formulation_temp, individual_temp, experiment_temp],
                               axis=1).reset_index(drop=True)

        if 'Sample_weight' not in folder_df.columns:
            folder_df['Sample_weight'] = [float(folder_df.Experiment_weight[i]) for i in range(len(folder_df))]

        all_df = pd.concat([all_df, folder_df], ignore_index=True)

    # Standardize cell-type / route names
    replacements = {
        'im': 'intramuscular', 'iv': 'intravenous', 'a549': 'lung_epithelium',
        'bdmc': 'macrophage', 'bmdm': 'dendritic_cell', 'hela': 'generic_cell',
        'hek': 'generic_cell', 'igrov1': 'generic_cell'
    }
    all_df = all_df.replace(replacements)
    all_df['Model_type'] = all_df['Model_type'].replace('muscle', 'Mouse')

    # Generate class labels
    tox_classes, del_classes = generate_classes(all_df)
    all_df['toxicity_class'] = tox_classes
    all_df['delivery_class'] = del_classes

    # Finalize toxicity target
    if TARGET_COL_TOXICITY in all_df.columns:
        all_df['unnormalized_toxicity'] = (all_df[TARGET_COL_TOXICITY]).round(5)
        all_df[TARGET_COL_TOXICITY] = (all_df[TARGET_COL_TOXICITY] / 100.0).clip(upper=1.0).round(5)

    all_df = all_df.where(all_df != True, 1.0).where(all_df != False, 0.0)
    all_df['MolWt'] = np.log1p(all_df['MolWt'])
    all_df['Lipid/Cells'] = np.log1p(all_df['Lipid/Cells'])
    all_df['mRNA/Cells'] = np.log1p(all_df['mRNA/Cells'])

    # ── Delivery dataset ──
    del_extra_x_variables = [
        'Ionizable_Lipid_Mol_Ratio', 'Phospholipid_Mol_Ratio', 'Cholesterol_Mol_Ratio',
        'PEG_Lipid_Mol_Ratio', 'Ionizable_Lipid_to_mRNA_weight_ratio', 'Num_tails',
        'Num_carbon_in_tail', 'MolWt', 'num_unsaturated_cc_bonds', 'num_protonatable_nitrogens',
    ]
    del_extra_x_categorical = ['Helper_lipid_ID', 'Cargo_type', 'Model_type']

    del_df, del_extra_x_variables = apply_ohe(all_df, del_extra_x_variables, del_extra_x_categorical)
    del_df = del_df[del_df[TARGET_COL_DELIVERY].notna()].reset_index(drop=True)
    del_df = del_df.drop(columns=[TARGET_COL_TOXICITY, 'unnormalized_toxicity', 'toxicity_class'],
                         errors='ignore')
    del_col_type = build_col_types(del_df, del_extra_x_variables, del_extra_x_categorical)

    print("Creating all_del.csv")
    change_column_order(os.path.join(write_path, 'all_del.csv'), del_df,
                        first_cols=[TARGET_COL_DELIVERY, 'smiles'])
    pd.DataFrame(del_col_type).to_csv(os.path.join(write_path, 'col_types_del.csv'), index=False)

    # ── Toxicity dataset ──
    tox_extra_x_variables = [
        'Ionizable_Lipid_Mol_Ratio', 'Phospholipid_Mol_Ratio', 'Cholesterol_Mol_Ratio',
        'PEG_Lipid_Mol_Ratio', 'Ionizable_Lipid_to_mRNA_weight_ratio', 'Num_tails',
        'Num_carbon_in_tail', 'MolWt', 'num_unsaturated_cc_bonds', 'num_protonatable_nitrogens',
        'mRNA/Cells', 'Lipid/Cells'
    ]
    tox_extra_x_categorical = ['Helper_lipid_ID', 'Cargo_type', 'Model_type']

    tox_df, tox_extra_x_variables = apply_ohe(all_df, tox_extra_x_variables, tox_extra_x_categorical)
    tox_df = tox_df[tox_df[TARGET_COL_TOXICITY].notna()].reset_index(drop=True)
    tox_df = tox_df.drop(columns=[TARGET_COL_DELIVERY, 'delivery_class'], errors='ignore')
    tox_col_type = build_col_types(tox_df, tox_extra_x_variables, tox_extra_x_categorical)

    print("Creating all_tox.csv")
    change_column_order(os.path.join(write_path, 'all_tox.csv'), tox_df,
                        first_cols=[TARGET_COL_TOXICITY, 'smiles'])
    pd.DataFrame(tox_col_type).to_csv(os.path.join(write_path, 'col_types_tox.csv'), index=False)


def generate_classes(all_df):
    tox_classes, del_classes = [], []
    for _, row in all_df.iterrows():
        try:
            tox = row.get(TARGET_COL_TOXICITY, np.nan)
            if pd.isna(tox):       tox_class = np.nan
            elif tox > 80:         tox_class = 0
            elif tox > 70:         tox_class = 1
            else:                  tox_class = 2
            tox_classes.append(tox_class)
        except Exception:
            tox_classes.append(np.nan)

        try:
            delivery = row.get(TARGET_COL_DELIVERY, np.nan)
            if pd.isna(delivery):  del_class = np.nan
            elif delivery > 1.0:   del_class = 2
            elif delivery >= -1.0: del_class = 1
            else:                  del_class = 0
            del_classes.append(del_class)
        except Exception:
            del_classes.append(np.nan)

    return tox_classes, del_classes


def main():
    merge_datasets(None)


if __name__ == '__main__':
    main()
