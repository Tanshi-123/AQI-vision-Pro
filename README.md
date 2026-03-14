# AQI_prediction

This repository contains code to train an image-based AQI classifier and serve predictions.

Quick start

1. Create a virtual environment and install dependencies:

```powershell
python -m venv venv
& venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

2. Prepare dataset

- Provide CSVs listing image filenames and labels (columns: `filename`, `label`).
- Put images in a folder and pass its path to `train.py`.

3. Train

```powershell
python train.py --train-csv data/train_data.csv --val-csv data/val_data.csv --image-dir "data/All_img" --epochs 10
```

4. Run inference API

```powershell
python app.py
# then POST files to http://localhost:5000/predict as form-data field `image`
```

Files added/updated

- `src/preprocessing.py` — image preprocessing utilities
- `src/model.py` — model builder (MobileNetV2 backbone)
- `train.py` — training script that saves `models/model.h5`
- `app.py` — Flask app for inference
- `requirements.txt` and `requiremnets.txt` — dependency lists

Next steps

- Verify CSV column names and dataset layout; adjust `x_col`/`y_col` flags for `train.py`.
- Optionally unfreeze backbone for fine-tuning.
- Add evaluation and unit tests.
