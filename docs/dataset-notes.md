# Dataset Notes

## Purpose
This note explains how the current dataset files in the repository should be interpreted and used within the coursework workflow.

## Current dataset files
- `datasets/givenData.csv`
- `datasets/kaggleData.csv`

In the current repository snapshot, these two files expose the same 16-column schema and have identical file hashes. Unless the project later documents a different provenance or role, they should be treated as alternative copies of the same underlying tabular dataset rather than as separate evidence sources.

## Observed schema
The dataset is a small tabular supervised-classification dataset with:

- target: `LUNG_CANCER`
- demographic inputs: `GENDER`, `AGE`
- symptom and lifestyle indicators such as `SMOKING`, `YELLOW_FINGERS`, `WHEEZING`, `COUGHING`, `SHORTNESS OF BREATH`, and `CHEST PAIN`

Current raw observations from `datasets/givenData.csv`:

- 309 rows
- 16 columns
- raw target balance of 270 `YES` and 39 `NO`
- several column names contain spaces or trailing whitespace
- binary predictors are encoded with `1` and `2`
- repeated rows appear to be present and need explicit handling

## Recommended usage in this repository
- Choose one canonical input file for the notebook and name it explicitly in the analysis.
- If both CSV files remain, document why both are retained.
- Do not edit the raw CSV files in place to clean names, remove duplicates, or recode values.
- Perform renaming, recoding, deduplication, and feature engineering in the notebook or another reproducible analytical step.
- Keep the explanation of dataset limitations close to the modelling and interpretation sections so the coursework does not overstate what the data can support.

## Interpretation cautions
- This is a compact, imbalanced tabular dataset, so headline metric scores can be misleading without a fuller discussion.
- Predictive associations in the notebook should not be described as clinical proof or causal evidence.
- Feature importance should be used to support cautious interpretation, not to claim diagnostic validity.
