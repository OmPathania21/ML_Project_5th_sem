# Datasets: first-look observations

Raw files are in `raw/`. Never edit them; save cleaned versions to `processed/`.

## 1. Google Play Store (EDA 1): `raw/play_store/googleplaystore.csv`
- 10,841 rows × 13 columns; 483 fully duplicated rows; 9,660 unique app names.
- `Rating` is missing for 1,474 apps. The ratings that exist sit at 4.0–4.5 (median 4.3), skewed toward high values.
- Numbers stored as text need parsing: `Reviews`; `Size` ("19M", "91k", 1,695 are "Varies with device"); `Installs` ("10,000+"); `Price` ("$4.99").
- Row 10472 is corrupted: its columns are shifted, giving Rating = 19 and Category = "1.9". Drop it.
- 92.6% of apps are Free and 7.4% Paid. The largest categories are FAMILY, GAME and TOOLS.
- `googleplaystore_user_reviews.csv` has 64k reviews with sentiment labels. It's optional; about 42% of the rows are empty.

## 2. Video Game Sales (EDA 2): `raw/video_game_sales/vgsales.csv`
- 16,598 games × 11 columns; no duplicates.
- `Year` is missing for 271 games and `Publisher` for 58. Data after 2016 is incomplete (only 4 games), so cut at 2016.
- Sales (in millions) are heavily right-skewed: median global sales are 0.17M, but Wii Sports sold 82.7M. Use log scales in plots.
- 12 genres (Action has the most games) and 31 platforms. The regions are NA, EU, JP and Other.

## 3. Combined Cycle Power Plant (Regression): `raw/ccpp/CCPP/Folds5x2_pp.xlsx`
- 9,568 rows × 5 columns. There are 5 sheets holding the same data shuffled, so use Sheet1 only.
- The features are AT (temperature), V (exhaust vacuum), AP (ambient pressure) and RH (humidity). The target is PE (MW).
- No missing values; 41 duplicate rows.
- Correlations with PE: AT −0.95 and V −0.87 are strong; AP +0.52 and RH +0.39 are weaker. AT and V are also correlated with each other, which is a multicollinearity point to mention.

## 4. Bank Marketing (Classification): `raw/bank_marketing/bank-additional/bank-additional-full.csv`
- Semicolon-separated (`sep=';'`). 41,188 rows × 20 features + target `y`; 12 duplicate rows.
- Imbalanced: 88.7% "no" and 11.3% "yes". Use F1, recall and ROC-AUC, and class weights or SMOTE.
- There are no NaNs, but "unknown" is used as a missing value: `default` 8,597, `education` 1,731, `housing` and `loan` 990 each.
- **Drop `duration`**: it's only known after the call ends, so it leaks the outcome.
- `pdays` = 999 means "never contacted before" (96% of rows). Turn it into a flag.
- The 5 macro-economic features (`emp.var.rate`, `euribor3m`, `nr.employed`, ...) are highly correlated with each other.
