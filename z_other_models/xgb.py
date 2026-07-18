import os
import pandas as pd
import numpy as np
import xgboost as xgb
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import mean_squared_error, mean_absolute_error
from rdkit import Chem
from rdkit.Chem import rdMolDescriptors

# --- Configuration ---
DATA_DIR = "../data/crossval_splits/624_del_base"
TARGET_COL = "quantified_delivery" # UPDATE THIS to the actual name of your target column
SMILES_COL = "smiles"
RESULTS_FILE = "xgb_cv_results.csv"

# Morgan Fingerprint parameters
FP_RADIUS = 2
FP_NBITS = 2048

def generate_morgan_fps(smiles_list, radius=FP_RADIUS, n_bits=FP_NBITS):
    """Converts a list of SMILES strings to a numpy array of Morgan fingerprints."""
    fps = []
    for smi in smiles_list:
        mol = Chem.MolFromSmiles(str(smi))
        if mol is not None:
            fp = rdMolDescriptors.GetMorganFingerprintAsBitVect(mol, radius, nBits=n_bits)
            fps.append(np.array(fp))
        else:
            # Handle invalid SMILES gracefully by returning a zero vector
            fps.append(np.zeros((n_bits,)))
    return np.array(fps)

def load_split_data(fold_path, split_name):
    """Loads the main, extra_x, and weights files for a given split."""
    # File paths
    main_path = os.path.join(fold_path, f"{split_name}.csv")
    extra_path = os.path.join(fold_path, f"{split_name}_extra_x.csv")
    weights_path = os.path.join(fold_path, f"{split_name}_weights.csv")
    
    # Load dataframes
    df_main = pd.read_csv(main_path)
    df_extra = pd.read_csv(extra_path)
    df_weights = pd.read_csv(weights_path)
    
    # Extract arrays
    smiles = df_main[SMILES_COL].values
    y = df_main[TARGET_COL].values
    
    # Assuming the extra features are all numerical columns in the extra_x file
    extra_x = df_extra.values
    
    # Assuming weights file has a single column containing the weights
    weights = df_weights.iloc[:, 0].values
    
    return smiles, extra_x, y, weights

def process_features(smiles, extra_x):
    """Generates fingerprints and concatenates them with extra features."""
    fps = generate_morgan_fps(smiles)
    # Combine Morgan FPs with extra numerical features
    X = np.hstack([fps, extra_x])
    return X

def main():
    results = []
    
    # Loop through folds 0 to 4
    for fold_idx in range(5):
        fold_name = f"fold_{fold_idx}"
        fold_path = os.path.join(DATA_DIR, fold_name)
        
        print(f"--- Processing {fold_name} ---")
        
        # 1. Load Data
        print("Loading data...")
        train_smiles, train_extra, y_train, w_train = load_split_data(fold_path, "train")
        test_smiles, test_extra, y_test, _ = load_split_data(fold_path, "test") # test weights usually not needed for evaluation
        
        # Optional: Load validation set for early stopping during XGBoost training
        valid_smiles, valid_extra, y_valid, w_valid = load_split_data(fold_path, "valid")
        
        # 2. Process Features (SMILES -> Morgan FP -> Concat with Extra X)
        print("Generating features...")
        X_train = process_features(train_smiles, train_extra)
        X_valid = process_features(valid_smiles, valid_extra)
        X_test = process_features(test_smiles, test_extra)
        
        # 3. Initialize and Train XGBoost Model
        print("Training model...")
        model = xgb.XGBRegressor(
            n_estimators=500,
            learning_rate=0.05,
            max_depth=6,
            subsample=0.8,
            colsample_bytree=0.8,
            early_stopping_rounds=50,

            random_state=42,
            n_jobs=-1
        )
        
        # Fit model with validation set for early stopping
        model.fit(
            X_train, y_train,
            sample_weight=w_train,
            eval_set=[(X_valid, y_valid)],
            sample_weight_eval_set=[w_valid],
            verbose=False
        )
        
        # 4. Predict and Evaluate on Test Set
        print("Evaluating on test set...")
        y_pred = model.predict(X_test)
        
        # Calculate Metrics
        pearson_corr, _ = pearsonr(y_test, y_pred)
        spearman_corr, _ = spearmanr(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        mae = mean_absolute_error(y_test, y_pred)
        
        # Store results
        fold_metrics = {
            "Fold": fold_idx,
            "Pearson_R": pearson_corr,
            "Spearman_Rho": spearman_corr,
            "RMSE": rmse,
            "MAE": mae,
            "Best_Iteration": model.best_iteration
        }
        results.append(fold_metrics)
        
        print(f"Results for {fold_name}: Pearson={pearson_corr:.4f}, Spearman={spearman_corr:.4f}")
        print("-" * 30)

    # 5. Save Results
    print("Saving cross-validation results...")
    results_df = pd.DataFrame(results)
    
    # Calculate and append the mean of all folds
    mean_metrics = results_df.mean().to_dict()
    mean_metrics["Fold"] = "Mean"
    results_df = pd.concat([results_df, pd.DataFrame([mean_metrics])], ignore_index=True)
    
    results_df.to_csv(RESULTS_FILE, index=False)
    print(f"Results successfully saved to {RESULTS_FILE}")
    print("\nSummary:")
    print(results_df)

if __name__ == "__main__":
    main()