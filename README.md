# ConfFold

## Prepare data

Download and extract the compressed file `qm9.tar.gz` from [this shared Drive](https://drive.google.com/drive/folders/1BBRpaAvvS2hTrH81mAE4WvyLIKMyhwN7) putting them in the subdirectory `data`.


## Standardize molecules

Checking if conformers match its SMILES string.
You can check all valid conformers by running

`python standardize_confs.py`

This saves all valid conformers into the file specified by argument `--out_dir` in script file `standardize_confs.py`

## Training model

You can train the model by running

`python train.py`
 
 All tunable parameters and directory to datasets can be found in `utils/parsing.py`
