# PBL 1: First Project Review

- **Review date:** 28.09.2026 (you present on the monitor)
- **Submit:** the final presentation in Google Classroom by 11:59 PM. The email says 27.08.2026, which is probably a typo for **27.09.2026**, the night before the review.
- **Deliverables:** a 10-minute presentation and a complete report (soft copy only).
- **Note:** the problem statement is fixed, but we may swap in different datasets. Marks are given only against the rubric below.

## Rubric (20 marks)

| # | Criterion | Marks | What to show |
|---|---|---|---|
| 1 | Problem Identification & Dataset Selection | 4 | Problem statement and why it matters; each of the 4 datasets with source, rows × columns, features, target; why each dataset fits its task |
| 2 | Exploratory Data Analysis & Insights | 4 | Play Store and Video Game Sales: missing values, cleaning, distributions, outliers, correlations, category/genre breakdowns. **Write out each insight in words, not only plots** |
| 3 | Regression and Classification Implementation | 4 | CCPP regression and Bank Marketing classification: preprocessing (encoding, scaling, train/test split), the models used, the pipeline |
| 4 | Comparative Performance Analysis | 4 | At least 3 models per task in a metrics table. Regression: MAE, RMSE, R². Classification: accuracy, precision, recall, F1, ROC-AUC, confusion matrix. Add cross-validation |
| 5 | Result Interpretation & Presentation | 4 | Best model and why, feature importance, business meaning, limitations, next steps, and a clean 10-minute delivery |

## Files

- `brief/`: dataset list, group list, rubric email
- `presentation/Group10_PBL1_Presentation.pptx`: 16 slides; each slide is tagged with the rubric criterion it covers, and speaker notes hold the talking points
- `report/Group10_PBL1_Report.pdf`: 22-page report with 29 figures
- All numbers come from the notebooks in `code/notebooks/` (run them in order 01 → 04)

## Suggested speaker split (10 min)

| Slides | Content | Rubric | ~Time |
|---|---|---|---|
| 1–3 | Title, problem statement, datasets | 1 | 2 min |
| 4–7 | Play Store and Video Game EDA | 2 | 2.5 min |
| 8–12 | Regression and classification pipelines, model comparison | 3, 4 | 3 min |
| 13–16 | Interpretation, recommendations, conclusion, Q&A | 5 | 2.5 min |

## Likely questions

- Why drop `duration`? It is only known after the call ends, so it leaks the target.
- Why not accuracy? The "always no" baseline already scores 88.7%.
- Why does Ridge equal Linear Regression? There are only 4 features and ~7.6K rows, so there is nothing to regularise.
- How was the 0.65 threshold chosen? On out-of-fold training predictions, never on the test set.
- Why XGBoost? It had the best cross-validation score in both tasks and captures non-linearity and interactions.
