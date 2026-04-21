# Active Deviations

This file tracks only live mismatches between the intended project design and the current repository state.

## 1. The notebook is still more scaffold than finished analysis
- Intended state: `blankTemplate.ipynb` should contain a complete end-to-end coursework workflow, including trained models, fair model comparison, final metrics, feature-importance discussion, and a direct conclusion against the brief.
- Current state: the notebook already has a strong section structure and some early data-cleaning and EDA cells, but large parts of the modelling, results, and conclusion sections remain as prompts or placeholders rather than completed analysis.
- Reason: the repository is currently closer to a coursework template or in-progress draft than to a finished submission.
- Resolution path: complete sections 4 to 6 with implemented models, evaluation outputs, interpretation, and final narrative.

## 2. Duplicate handling is provisional rather than fully documented
- Intended state: duplicate records should be identified and handled through a reproducible rule that is explained in the notebook.
- Current state: the notebook notes duplicated rows and currently removes a trailing block of rows by index, which is a provisional assumption rather than a clearly generalised cleaning rule.
- Reason: exploratory cleanup appears to have started before the final preprocessing policy was formalised.
- Resolution path: replace the hard-coded row drop with a reproducible duplicate-detection rule or document a stronger justification for any targeted exclusion.

## 3. The roles of `givenData.csv` and `kaggleData.csv` are not yet explicit
- Intended state: collaborators should be able to tell which dataset file is canonical for analysis and why any second copy is retained.
- Current state: both CSV files are present, they currently match at the file-content level, and the notebook only reads `datasets/givenData.csv`.
- Reason: source copies have been kept in the repo without an accompanying explanation of provenance or intended usage.
- Resolution path: either document the role of each file clearly or simplify the workflow around one canonical dataset reference.

## 4. Some notebook markdown is still draft-grade rather than submission-ready
- Intended state: notebook narrative should use an academic-professional tone throughout.
- Current state: several markdown prompts and notes are still phrased as open questions, informal reminders, or placeholder comments.
- Reason: the notebook is still serving as a working template for the group.
- Resolution path: replace draft notes with final explanatory text once the relevant analysis decisions are complete.
