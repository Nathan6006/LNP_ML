"""export_models.py - copy trained deployment fold-models into a top-level models/ folder.

train.py / train_tox.py save each fold under
    new_data/crossval_splits/<split>/fold_<i>/model_<i>/{final_model, *.pkl}
For deployment we gather the 5 fold models per property into a clean, self-contained
    models/<name>/model_<i>/...
that screen.py reads exclusively (so the screen never reaches back into the split tree).

Usage (from scripts/):
    python export_models.py del_deploy_B  del_deploy
    python export_models.py lnpcd_tox_deploy_B  tox_deploy
"""
import argparse
import os
import shutil
import sys

from config import DEFAULT_CV_FOLDS
from model_common import model_dir_name

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def main():
    ap = argparse.ArgumentParser(description="Copy trained fold-models into models/<name>/.")
    ap.add_argument("split_folder", help="Split folder under new_data/crossval_splits/.")
    ap.add_argument("out_name", help="Destination under models/ (e.g. del_deploy, tox_deploy).")
    ap.add_argument("--data_dir", default="../new_data")
    ap.add_argument("--dest", default=None,
                    help="Destination models dir (default: <BASE>/models/<out_name>).")
    ap.add_argument("--cv", type=int, default=DEFAULT_CV_FOLDS)
    args = ap.parse_args()

    dst_root = args.dest or os.path.join(BASE, "models", args.out_name)
    os.makedirs(dst_root, exist_ok=True)
    n = 0
    for cv in range(args.cv):
        src = os.path.join(args.data_dir, "crossval_splits", args.split_folder,
                           f"fold_{cv}", model_dir_name(cv))
        if not os.path.isdir(src):
            print(f"  fold {cv}: {src} not found — skipping.")
            continue
        dst = os.path.join(dst_root, model_dir_name(cv))
        if os.path.exists(dst):
            shutil.rmtree(dst)
        shutil.copytree(src, dst)
        print(f"  fold {cv}: {src} -> {dst}")
        n += 1
    if not n:
        sys.exit("No fold models copied — did training run?")
    print(f"\nCopied {n} fold-model(s) to {dst_root}")


if __name__ == "__main__":
    main()
