BASE_MODEL = "DeepChem/ChemBERTa-77M-MTR"

DEFAULT_CV_FOLDS = 5

DATA_FILES = {
    "del": ("LNPDB_vitro_del_processed.csv", "col_types_del.csv"),
    "tox": ("all_tox.csv", "col_types_tox.csv"),
}
