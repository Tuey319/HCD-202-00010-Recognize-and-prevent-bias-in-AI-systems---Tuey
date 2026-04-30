# HCD-202 Fallacy Hunter Fairness Experiment

This workspace contains a notebook-based experiment that checks whether a fallacy classification model behaves differently across equivalent arguments written in different styles.
The model is published on Hugging Face Hub, so users can load it directly without downloading or uploading a local checkpoint manually.

## Contents

- `fallacy_hunter_fairness.ipynb` - main notebook for generating test cases, running predictions, and analyzing fairness across writing styles.
- `fallacy_model/` - local Hugging Face checkpoint copy for offline use or development.
- `test_data.csv` - generated or saved test set used in the experiment.
- `experiment_results.csv` - saved prediction and analysis output from the notebook.

## Requirements

The notebook uses Python with:

- `transformers`
- `torch`
- `pandas`
- `numpy`
- `scipy`
- `matplotlib`
- `seaborn`
- `tqdm`

## How to Run

1. Open `fallacy_hunter_fairness.ipynb` in VS Code or Jupyter.
2. Install the dependencies if needed.
3. Update `MODEL_PATH` in the configuration cell so it points to the Hugging Face repo id, for example `WinterJet2021/Fallacy-Hunter`.
4. Run the notebook from top to bottom.

## Expected Output

The notebook generates predictions for 90 arguments across three styles and summarizes whether the model's labels or confidence scores differ by style.

## Notes

- If you want to work offline, keep the `fallacy_model/` folder next to the notebook as a fallback copy.
- For normal use, load the model directly from Hugging Face Hub by setting `MODEL_PATH` to the repo id.
- The notebook is designed to work as a self-contained fairness audit for the provided data and model files.
