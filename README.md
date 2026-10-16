# Sentiment Classification: GRU vs DistilBERT

> Comparing a Bidirectional GRU with a fine-tuned DistilBERT on political comment sentiment.

## Overview

- Individual Deep Learning final-exam project (UAS, dataset A) classifying comments as negative, neutral or positive.
- Compares two model families, each as an original and an improved variant, and analyses how each one generalizes.

## Data / Scope

- About 241,000 English comments about Indian politics (Modi, BJP, Congress), labeled 0 (negative), 1 (neutral), 2 (positive); roughly 23% / 34% / 43%.
- After removing nulls and duplicate comments, about 214,000 comments remained.

## Workflow

- Cleaning: removed null and duplicate comments, lowercased, stripped URLs, converted emoji to text, truncated to 50 words.
- EDA: class balance, text-length distribution and word clouds per sentiment.
- Base model: Embedding + Bidirectional GRU (1024 and 512 units) with dropout and class weights, 15 epochs.
- Improved GRU with a smaller capacity and stronger regularization.
- DistilBERT fine-tuned with Hugging Face Transformers, plus an improved variant with a frozen encoder.
- Compared all four models by accuracy, macro F1 and train-versus-validation loss.

## Results

- The original GRU reached about 95.7% accuracy (macro F1 0.955), but its validation loss climbed steadily while training loss fell, a clear sign of overfitting.
- Both DistilBERT variants scored roughly 96-97% accuracy; the improved variant with a frozen encoder showed the most stable train and validation losses.
- The improved GRU traded accuracy for stability and scored well below the original GRU.
- Takeaway: high accuracy alone was misleading, and loss curves showed which models truly generalized.

## Tech Stack

Python, TensorFlow/Keras, Hugging Face Transformers (DistilBERT), scikit-learn, NLTK, emoji

## Repository Structure

```
notebook/    Full notebook with code, outputs and written interpretation
figures/     Key charts exported from the notebook
requirements.txt
```

## How to Run

```bash
pip install -r requirements.txt
jupyter notebook notebook/UAS_DL_Sentiment_GRU_vs_DistilBERT.ipynb
```

The notebook was written in Google Colab; the dataset is not included, so upload it to the Colab session (or edit the path in the first data-loading cell).

## Team

Individual project by Nadya Angelie Lislie (Deep Learning course, BINUS University).

## Notes

- Student ID numbers were removed from the notebook text.
- TODO: the summary table and bar chart in the final section use hard-coded numbers that differ from the evaluation printed earlier (for example improved GRU 0.44 vs 0.857, DistilBERT 0.9726 vs 0.9585). Re-run the evaluation cells and regenerate the table before relying on those figures.
- The comment dataset is not redistributed here.

## Author

Nadya Angelie Lislie, BINUS University (Computer Science and Statistics). [LinkedIn](https://www.linkedin.com/in/nadya-angelie-lislie/) | [GitHub](https://github.com/Nadya-AngelieLislie)

## Experiment Tracking

`src/experiment_tracking.py` provides an optional MLflow wrapper (`track_run`) for logging parameters, metrics and artifacts; it does nothing when MLflow is not installed. Run `mlflow ui` after logging runs to compare experiments.

## Continuous Integration

GitHub Actions (`.github/workflows/ci.yml`) runs `ruff` and `pytest` repository-health checks on every push; run them locally with `make lint test`.
