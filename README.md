# Lung Cancer Classification Coursework Repository

## Introduction
This repository supports a group machine learning coursework project built around a Jupyter notebook-led analysis of lung cancer prediction. The work is organised as an end-to-end supervised classification exercise: inspect the data, justify preprocessing choices, explore patterns, train and compare classifiers, and interpret the features that appear most predictive.

The repository is intended for group members, tutors, and markers reviewing the academic workflow. It is not structured as a reusable software package or deployable product.

## Project objective
The objective is to develop and evaluate classification models that predict the `LUNG_CANCER` outcome from patient demographic, lifestyle, and symptom variables. The coursework emphasis is on:

- framing the classification problem clearly
- preparing the data in a justified and reproducible way
- comparing models with appropriate classification metrics
- identifying and discussing the most influential predictive features
- reflecting critically on limitations, bias, and uncertainty

## Repository structure
- `blankTemplate.ipynb`: main notebook template and primary coursework artefact. It contains the planned narrative from problem definition through conclusion.
- `datasets/givenData.csv`: tabular input data currently used by the notebook.
- `datasets/kaggleData.csv`: second copy of the same tabular dataset currently present in the repository.
- `scripts/update_notebook_badges.py`: small helper script for refreshing the team badge cell in the notebook.
- `docs/`: supporting documentation for workflow, design notes, dataset handling, and live deviations.

## Dataset overview
The repository currently includes two CSV files in `datasets/`. Both expose the same 16-column lung cancer classification schema and, in the current repository state, match at the file-content level.

- Target column: `LUNG_CANCER`
- Predictor types: one demographic field (`AGE`), one sex/gender field (`GENDER`), and a set of binary symptom or lifestyle indicators such as `SMOKING`, `WHEEZING`, `COUGHING`, `SHORTNESS OF BREATH`, and `CHEST PAIN`
- Current raw size: 309 rows x 16 columns
- Current raw class balance: 270 `YES` and 39 `NO`

The notebook already treats duplicate handling, column cleanup, and binary recoding as important preprocessing steps. Those decisions should remain explicit and justified in the final analysis.

## Notebook workflow
`blankTemplate.ipynb` is the centre of the repository. The expected workflow is:

1. define the supervised classification task and the coursework success criteria
2. inspect the dataset structure, column meanings, and data quality issues
3. carry out justified preprocessing, including duplicate handling, encoding, and any feature engineering
4. perform exploratory analysis and visualisation to understand class balance, patterns, and potential modelling issues
5. train and compare classification models using appropriate validation and evaluation metrics
6. interpret the final model through feature importance and error analysis
7. conclude against the brief and summarise limitations and next steps

The supporting documentation explains how the repo should be used, but the notebook remains the primary place where evidence, results, and narrative should live.

## Outputs / deliverables
This repository is set up to support the following coursework outputs:

- a completed notebook with narrative, code, figures, and interpretation
- classification performance summaries such as precision, recall, F1-score, confusion matrices, and model comparison tables
- discussion of the features that appear most important to prediction
- presentation-ready findings for a group submission

## Disclaimer
This repository is for coursework, research, and educational analysis only. It is not a clinical decision-support system, medical device, or validated diagnostic tool. Any model outputs, plots, or feature interpretations in this repository must not be used for patient care or medical decision-making.

## Supporting docs
- [AGENTS.md](AGENTS.md)
- [systemDesign.md](systemDesign.md)
- [Architecture notes](docs/architecture-notes.md)
- [Dataset notes](docs/dataset-notes.md)
- [Active deviations](docs/deviations.md)
