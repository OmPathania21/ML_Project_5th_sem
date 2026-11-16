// Builds the PBL III deck (Group 10: Diabetes Prediction with Ensemble Learning) from code/outputs.
// Usage (from code/):  node deck/build_pbl3_deck.js <out.pptx>
const pptxgen = require("pptxgenjs");
const fs = require("fs");
const path = require("path");

const ROOT = path.join(__dirname, "..");
const FIG = path.join(ROOT, "outputs", "figures");
const S = JSON.parse(fs.readFileSync(path.join(ROOT, "outputs", "models", "diabetes_summary.json"), "utf8").replace(/NaN/g, "null"));
const OUT = process.argv[2] || path.join(__dirname, "Group10_PBL3_Presentation.pptx");

const C = { navy: "0F1F3D", navy2: "1B2F55", blue: "2A78D6", orange: "EB6834", green: "1BAF7A", ink: "1A1A1A", ink2: "52514E",
  muted: "898781", tint: "EEF4FC", tint2: "FDF0EA", tint3: "E8F6EF", line: "E1E0D9", white: "FFFFFF", ice: "B7D3F6", zebra: "F7F9FC" };
const HF = "Arial", BF = "Calibri";
const W = 13.333, M = 0.6, CW = W - 2 * M;
const RUBRIC = {
  1: "Rubric 1 · Problem Identification & Dataset Selection · 4 marks",
  2: "Rubric 2 · Implementation of Ensemble Learning Approaches · 8 marks",
  3: "Rubric 3 · Comparative Performance Analysis · 4 marks",
  4: "Rubric 4 · Result Interpretation & Presentation · 4 marks",
};

const pres = new pptxgen();
pres.layout = "LAYOUT_WIDE";
pres.title = "Diabetes Prediction with Ensemble Learning";
pres.author = "Group 10";

let n = 0;
const pct = (v, d = 1) => `${(v * 100).toFixed(d)}%`, f2 = (v) => Number(v).toFixed(2), f3 = (v) => Number(v).toFixed(3), f4 = (v) => Number(v).toFixed(4);
const fmt = (v) => Math.round(v).toLocaleString("en-US");
const cv = S.cv, test = S.test_default, thr = S.threshold;
const T = Object.fromEntries(test.map((r) => [r.Model, r]));
const CVm = Object.fromEntries(cv.map((r) => [r.Model, r]));
const best = S.best, bestName = S.best_model, lr = S.lr_test, dt = S.dt_test;
const ens = test.filter((r) => r.Family !== "Baseline");

function pngSize(name) { const b = fs.readFileSync(path.join(FIG, `${name}.png`)); return [b.readUInt32BE(16), b.readUInt32BE(20)]; }
function fig(s, name, x, y, w, h, align = "left") {
  const [pw, ph] = pngSize(name);
  let iw = w, ih = (w * ph) / pw;
  if (ih > h) { ih = h; iw = (h * pw) / ph; }
  const ix = align === "center" ? x + (w - iw) / 2 : x;
  s.addImage({ path: path.join(FIG, `${name}.png`), x: ix, y, w: iw, h: ih });
  return { x: ix, y, w: iw, h: ih };
}
function contentSlide(title, rubric, notes) {
  const s = pres.addSlide(); n += 1;
  s.background = { color: C.white };
  s.addText(RUBRIC[rubric], { shape: pres.shapes.ROUNDED_RECTANGLE, rectRadius: 0.08, x: M, y: 0.32, w: 6.1, h: 0.32,
    fill: { color: C.tint }, color: C.blue, fontFace: BF, fontSize: 11, bold: true, margin: [0, 8, 0, 8], valign: "middle", isTextBox: true });
  s.addText(title, { x: M, y: 0.72, w: CW, h: 0.8, fontFace: HF, fontSize: 28, bold: true, color: C.ink, margin: 0, valign: "middle", isTextBox: true });
  s.addText(`${n}`, { x: W - M - 0.6, y: 7.05, w: 0.6, h: 0.3, fontFace: BF, fontSize: 10, color: C.muted, align: "right", margin: 0, isTextBox: true });
  s.addText("Group 10 · PBL Review 3", { x: M, y: 7.05, w: 4, h: 0.3, fontFace: BF, fontSize: 10, color: C.muted, margin: 0, isTextBox: true });
  if (notes) s.addNotes(notes);
  return s;
}
function card(s, x, y, w, h, fill = C.tint) { s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x, y, w, h, rectRadius: 0.1, fill: { color: fill }, line: { color: fill } }); }
function stat(s, x, y, w, h, big, label, color = C.blue, fill = C.tint) {
  card(s, x, y, w, h, fill);
  s.addText([{ text: big, options: { fontFace: HF, fontSize: 24, bold: true, color, breakLine: true } },
    { text: label, options: { fontFace: BF, fontSize: 12.5, color: C.ink2 } }], { x: x + 0.2, y: y + 0.08, w: w - 0.4, h: h - 0.16, valign: "middle", margin: 0, isTextBox: true });
}
function textCard(s, x, y, w, h, head, body, accent = C.blue, fill = C.tint, size = 13) {
  card(s, x, y, w, h, fill);
  s.addText([{ text: head, options: { bold: true, fontSize: size + 2, color: accent, breakLine: true } }, { text: body, options: { fontSize: size, color: C.ink2 } }],
    { x: x + 0.2, y: y + 0.1, w: w - 0.4, h: h - 0.2, fontFace: BF, valign: "top", margin: 0, isTextBox: true });
}
function numbered(s, num, head, body, x, y, w, h, color = C.blue, size = 13) {
  s.addText(num, { shape: pres.shapes.OVAL, x, y: y + 0.02, w: 0.42, h: 0.42, fill: { color }, color: C.white, fontFace: HF, fontSize: 13, bold: true, align: "center", valign: "middle", margin: 0, isTextBox: true });
  s.addText([{ text: head, options: { bold: true, color: C.ink, breakLine: true } }, { text: body, options: { color: C.ink2 } }],
    { x: x + 0.58, y, w: w - 0.58, h, fontFace: BF, fontSize: size, valign: "top", margin: 0, isTextBox: true });
}
function bullets(s, items, x, y, w, h, size = 14, color = C.ink2) {
  const runs = [];
  items.forEach((it, i) => runs.push({ text: it, options: { color, bullet: { indent: 16 }, breakLine: i < items.length - 1 } }));
  s.addText(runs, { x, y, w, h, fontFace: BF, fontSize: size, valign: "top", margin: 0, paraSpaceAfter: 7, isTextBox: true });
}
function table(s, header, rows, x, y, colW, rowH, opts = {}) {
  const cell = (t, o = {}) => ({ text: t, options: { fontFace: BF, fontSize: opts.size || 12, color: C.ink2, valign: "middle", margin: [2, 5, 2, 5], ...o } });
  const t = [header.map((h, j) => cell(h, { bold: true, color: C.white, fill: { color: C.navy }, align: j && opts.numeric ? "center" : "left" }))];
  rows.forEach((r, i) => {
    const hl = (opts.highlight || []).includes(i), base = (opts.muted || []).includes(i);
    t.push(r.map((v, j) => cell(v, { fill: { color: hl ? C.tint2 : i % 2 ? C.white : C.zebra }, bold: hl,
      color: hl && j === 0 ? C.orange : base ? C.muted : j === 0 ? C.ink : C.ink2, align: j && opts.numeric ? "center" : "left" })));
  });
  s.addTable(t, { x, y, w: colW.reduce((a, b) => a + b, 0), colW, rowH, border: { type: "solid", pt: 0.5, color: C.line } });
}

// ───────────── 1. Title
{
  const s = pres.addSlide(); n += 1;
  s.background = { color: C.navy };
  s.addText("MACHINE LEARNING PROJECT  ·  PBL REVIEW 3  ·  GROUP 10", { x: M, y: 0.7, w: CW, h: 0.4, fontFace: BF, fontSize: 13, bold: true, color: C.ice, charSpacing: 2, margin: 0, isTextBox: true });
  s.addText("Diabetes Prediction with Ensemble Learning", { x: M, y: 1.25, w: 11.8, h: 1.0, fontFace: HF, fontSize: 40, bold: true, color: C.white, margin: 0, valign: "top", isTextBox: true });
  s.addText("An intelligent diabetes risk model from patient health records: ten ensemble classifiers compared against " +
    "Logistic Regression and Decision Tree baselines", { x: M, y: 2.35, w: 11.3, h: 0.8, fontFace: BF, fontSize: 18, color: C.ice, margin: 0, isTextBox: true });
  [["DATASET", `${fmt(S.rows_clean)} patient records`], ["ENSEMBLES", "Bagging · Boosting · Voting · Stacking"], ["BEST RESULT", `F1 ${f3(best.F1)} · ROC-AUC ${f3(best["ROC-AUC"])}`]]
    .forEach(([t, c], i) => {
      const x = M + i * 4.1;
      s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x, y: 3.6, w: 3.85, h: 0.95, rectRadius: 0.1, fill: { color: C.navy2 }, line: { color: C.navy2 } });
      s.addText([{ text: t, options: { fontSize: 10.5, bold: true, color: i === 2 ? C.orange : C.ice, charSpacing: 1, breakLine: true } },
        { text: c, options: { fontSize: 15, bold: true, color: C.white } }], { x: x + 0.22, y: 3.65, w: 3.5, h: 0.85, fontFace: BF, valign: "middle", margin: 0, isTextBox: true });
    });
  const team = [["Sanskriti Rastogi", "RA2411003012439"], ["Sahil Raj", "RA2411003012492"], ["Yash Santosh Lohare", "RA2411003012493"], ["Om Pathania", "RA2411003012495"]];
  const cw = (CW - 3 * 0.25) / 4;
  s.addText("GROUP 10", { x: M, y: 5.25, w: 3, h: 0.3, fontFace: BF, fontSize: 11, bold: true, color: C.ice, charSpacing: 2, margin: 0, isTextBox: true });
  team.forEach(([nm, roll], i) => s.addText([{ text: nm, options: { bold: true, color: C.white, fontSize: 14, breakLine: true } }, { text: roll, options: { color: C.ice, fontSize: 11.5 } }],
    { x: M + i * (cw + 0.25), y: 5.6, w: cw, h: 0.62, fontFace: BF, margin: 0, isTextBox: true }));
  s.addText("Guide: Dr. Arnab Maity", { x: M, y: 6.6, w: 6, h: 0.35, fontFace: BF, fontSize: 13, color: C.ice, margin: 0, isTextBox: true });
  s.addNotes("Our PBL III topic: develop an intelligent diabetes prediction model using patient health records, and investigate " +
    "the effectiveness of different ensemble classifiers in identifying individuals at risk of diabetes.");
}

// ───────────── 2. Problem
{
  const s = contentSlide("Problem statement and objectives", 1,
    "Diabetes is often detected late. Health records already contain the warning signs. The question is which machine-learning " +
    "approach turns them into the most reliable risk flag, and specifically whether ensembles beat a single model.");
  card(s, M, 1.75, 5.6, 4.3, C.tint);
  s.addText([
    { text: "The problem", options: { bold: true, fontSize: 19, color: C.ink, breakLine: true } },
    { text: "Diabetes often goes undiagnosed until complications appear. Routine records (age, BMI, blood pressure, smoking, HbA1c, " +
      "glucose) already contain the warning signs. Can a model flag at-risk patients reliably?", options: { fontSize: 15, color: C.ink2, breakLine: true } },
    { text: " ", options: { fontSize: 8, breakLine: true } },
    { text: "Why ensembles?", options: { bold: true, fontSize: 19, color: C.ink, breakLine: true } },
    { text: "Risk depends on non-linear, interacting factors. Combining many trees lowers variance (bagging) or bias (boosting) " +
      "compared with any single model.", options: { fontSize: 15, color: C.ink2 } },
  ], { x: M + 0.3, y: 1.95, w: 5.0, h: 4.0, fontFace: BF, valign: "top", margin: 0, paraSpaceAfter: 6, isTextBox: true });
  [["Build", "A leakage-free pipeline on 96K patient records with imbalance-aware evaluation"],
    ["Implement", "10 ensembles: Bagging, Random Forest, Extra Trees, AdaBoost, Gradient Boosting, XGBoost, LightGBM, CatBoost, Voting, Stacking"],
    ["Compare", "Against Logistic Regression and Decision Tree baselines, using F1-score, ROC-AUC, PR-AUC and significance tests"],
    ["Interpret", "Explain which factors drive risk (SHAP) and what the model means clinically"]]
    .forEach(([h, b], i) => numbered(s, `${i + 1}`, h, b, M + 6.0, 1.8 + i * 1.12, CW - 6.0, 1.05, i === 1 ? C.orange : C.blue));
}

// ───────────── 3. Dataset
{
  const s = contentSlide("Dataset: 100,000 electronic patient records", 1,
    "Kaggle Diabetes Prediction Dataset: demographics, comorbidities, smoking, BMI, HbA1c and glucose. After removing duplicates " +
    "and 18 'Other' gender records we keep about 96 thousand patients. Only 8.8% are diabetic, so we use F1 and AUC, not accuracy.");
  table(s, ["Attribute", "Type", "Range / values"], [
    ["gender", "categorical", "Female, Male"], ["age", "continuous", "0.08 – 80 years"], ["hypertension", "binary", "0 / 1 (7.5% yes)"],
    ["heart_disease", "binary", "0 / 1 (3.9% yes)"], ["smoking_history", "categorical", "never, former, current, not current, ever, No Info"],
    ["bmi", "continuous", "10 – 96 kg/m²"], ["HbA1c_level", "continuous", "3.5 – 9.0 %"], ["blood_glucose_level", "continuous", "80 – 300 mg/dL"],
    ["diabetes (target)", "binary", "1 = diabetic"],
  ], M, 1.75, [2.4, 1.6, 3.9], 0.47, { size: 12.5, highlight: [8] });
  const x = M + 8.3, w = CW - 8.3;
  stat(s, x, 1.75, w, 1.15, `${fmt(S.rows_raw)} → ${fmt(S.rows_clean)}`, `records after removing ${fmt(S.duplicates_removed)} duplicates + 18 rare "Other"`);
  stat(s, x, 3.05, w, 1.15, pct(S.prevalence), "diabetic: imbalanced, so F1 and AUC matter, not accuracy", C.orange, C.tint2);
  stat(s, x, 4.35, w, 1.15, `${fmt(S.train)} / ${fmt(S.test)}`, "stratified train / test split (80 / 20)");
  s.addText("Source: kaggle.com/datasets/iammustafatz/diabetes-prediction-dataset", { x: M, y: 6.55, w: CW, h: 0.35, fontFace: BF, fontSize: 11, color: C.muted, margin: 0, isTextBox: true });
}

// ───────────── 4. EDA
{
  const s = contentSlide("What drives diabetes risk in the data", 1,
    "Left: risk rises steeply with age, BMI, hypertension and heart disease. Right: HbA1c against glucose. Above HbA1c 6.6% or " +
    "glucose 200 every patient is diabetic; those are two-thirds of all diabetics and easy to find. The hard third have normal-range lab values.");
  fig(s, "db_risk_factors", M, 1.7, CW, 2.75, "center");
  fig(s, "db_hba1c_glucose", M, 4.5, 5.6, 2.5);
  textCard(s, M + 5.9, 4.6, CW - 5.9, 2.3, "Key insight: easy cases and hard cases",
    `Every patient with HbA1c > 6.6% or glucose > 200 is diabetic: ${pct(S.rule_share)} of patients, ${pct(S.rule_capture)} of all diabetics. ` +
    "The other third of diabetics have normal-range lab values: to find them a model must combine age, BMI, comorbidities and lab values.", C.orange, C.tint2, 13);
}

// ───────────── 5. Pipeline
{
  const s = contentSlide("Implementation: a leakage-free pipeline", 2,
    "Clean, then split before anything is fitted. Encoding and scaling live inside a pipeline, refitted in every fold. All twelve " +
    "models use the same 5 stratified folds. LightGBM is tuned with a randomised search, the threshold is chosen on out-of-fold " +
    "predictions, and the test set is used once.");
  const steps = [["Clean", `Remove ${fmt(S.duplicates_removed)} duplicates and 18 rare-gender rows; keep "No Info" smoking as a category`],
    ["Split", "Stratified 80/20; test set locked away"], ["Pipeline", "One-hot (gender, smoking) + standardise numeric, fitted per fold"],
    ["12 models", "2 baselines + 10 ensembles, same 5 stratified folds"], ["Tune", "LightGBM randomised search: 40 configs × 5 folds, PR-AUC"],
    ["Threshold", "F1-optimal cut-off on out-of-fold training predictions"], ["Evaluate", "Test metrics, McNemar test, SHAP explanations"]];
  const bw = (CW - 6 * 0.15) / 7;
  steps.forEach(([h, b], i) => {
    const x = M + i * (bw + 0.15), hl = [3, 4, 5].includes(i);
    card(s, x, 1.85, bw, 2.55, hl ? C.tint2 : C.tint);
    s.addText(`${i + 1}`, { shape: pres.shapes.OVAL, x: x + 0.15, y: 2.0, w: 0.42, h: 0.42, fill: { color: hl ? C.orange : C.blue }, color: C.white, fontFace: HF, fontSize: 13, bold: true, align: "center", valign: "middle", margin: 0, isTextBox: true });
    s.addText([{ text: h, options: { bold: true, fontSize: 14, color: C.ink, breakLine: true } }, { text: b, options: { fontSize: 11.5, color: C.ink2 } }],
      { x: x + 0.15, y: 2.55, w: bw - 0.3, h: 1.8, fontFace: BF, valign: "top", margin: 0, isTextBox: true });
  });
  textCard(s, M, 4.7, (CW - 0.3) / 2, 2.1, "Why F1 and ROC-AUC, not accuracy?",
    "Only 8.8% of patients are diabetic: a model that always says “no” is 91% accurate and useless. F1 balances finding diabetics " +
    "(recall) with not raising false alarms (precision); ROC-AUC measures ranking quality at every threshold.", C.blue, C.tint, 13);
  textCard(s, M + (CW - 0.3) / 2 + 0.3, 4.7, (CW - 0.3) / 2, 2.1, "No data leakage",
    "Duplicates removed before splitting; preprocessing fitted inside each fold; the threshold and hyperparameters chosen on training " +
    "data only; the test set used exactly once.", C.orange, C.tint2, 13);
}

// ───────────── 6. Ensembles implemented
{
  const s = contentSlide("Ten ensemble methods implemented (plus two baselines)", 2,
    "Three families. Bagging trains many trees on bootstrap samples and averages them, which lowers variance. Boosting adds trees " +
    "one after another, each fixing the previous ones' errors, which lowers bias. Meta-ensembles combine the best models: voting " +
    "averages their probabilities; stacking learns how much to trust each one.");
  table(s, ["Family", "Model", "Core idea", "Key settings"], [
    ["Baseline", "Logistic Regression", "Linear decision boundary", "L2, standardised features"],
    ["Baseline", "Decision Tree (CART)", "Single tree of Gini splits", "Fully grown"],
    ["Bagging", "Bagging", "Average trees trained on bootstrap samples", "200 trees"],
    ["Bagging", "Random Forest", "Bagging + random feature subset per split", "300 trees, min leaf 2"],
    ["Bagging", "Extra Trees", "Random split thresholds as well", "300 trees, min leaf 2"],
    ["Boosting", "AdaBoost", "Re-weight misclassified patients; weighted vote", "300 stumps, lr 0.5"],
    ["Boosting", "Gradient Boosting", "Each tree fits the gradient of the log-loss", "300 trees, depth 3"],
    ["Boosting", "XGBoost", "Regularised 2nd-order boosting", "400 trees, depth 5, lr 0.05"],
    ["Boosting", "LightGBM", "Histogram, leaf-wise boosting", "400 trees, 31 leaves (then tuned)"],
    ["Boosting", "CatBoost", "Ordered boosting, symmetric trees", "500 iters, depth 6"],
    ["Meta", "Soft Voting", "Average probabilities of RF, XGB, LGBM, CatBoost", "equal weights"],
    ["Meta", "Stacking", "Logistic meta-learner on out-of-fold probabilities", "5-fold inner CV"],
  ], M, 1.7, [1.5, 2.6, 5.0, 3.03], 0.395, { size: 12, muted: [0, 1] });
}

// ───────────── 7. How ensembles improve
{
  const g = S.gb_stage_auc_first_last, rf = S.rf_auc_by_trees;
  const s = contentSlide("Why ensembles work: variance and bias reduction in action", 2,
    "Left: gradient boosting. As trees are added one at a time, test ROC-AUC climbs, which is bias being reduced step by step. " +
    "Right: random forest. One tree alone is weak; averaging more trees steadily raises test AUC, which is variance reduction.");
  fig(s, "db_ensemble_size", M, 1.7, CW, 3.9, "center");
  const cw = (CW - 0.3) / 2;
  textCard(s, M, 5.75, cw, 1.05, "Boosting reduces bias", `Test ROC-AUC ${f3(g[0])} after 1 tree → ${f3(g[1])} after 300 sequential trees.`, C.orange, C.tint2, 13);
  textCard(s, M + cw + 0.3, 5.75, cw, 1.05, "Bagging reduces variance", `One tree: ROC-AUC ${f3(rf["1"])} → average of 300 trees: ${f3(rf["300"])}.`, C.blue, C.tint, 13);
}

// ───────────── 8. Tuning & threshold
{
  const t = S.tuned_params;
  const s = contentSlide("Tuning, threshold and model selection", 2,
    "LightGBM was tuned over 9 hyperparameters with a randomised search (40 configurations, 5 folds, scored on PR-AUC). " +
    "We also searched for the F1-optimal decision threshold on out-of-fold training predictions. Honest finding: 0.5 was already " +
    "near-optimal, because the ensemble's F1 is flat across thresholds and its probabilities are well calibrated.");
  table(s, ["Hyperparameter", "Selected"], Object.entries(t).map(([k, v]) => [k, `${v}`]), M, 1.75, [2.6, 1.4], 0.38, { size: 12 });
  fig(s, "db_threshold_calibration", M + 4.3, 1.75, CW - 4.3, 3.4, "center");
  textCard(s, M + 4.3, 5.35, CW - 4.3, 1.45, `Finding: 0.5 is already optimal (out-of-fold best ${f2(S.best_threshold)})`,
    `${bestName}'s F1 curve (left, orange) is flat from 0.35 to 0.95, so moving the threshold changes test F1 by < 0.01. ` +
    "Right: the boosted models are well calibrated, so their probabilities can be read as risk.", C.orange, C.tint2, 12.5);
}

// ───────────── 9. CV comparison
{
  const top = cv[0];
  const s = contentSlide("5-fold cross-validation: every ensemble beats both baselines", 3,
    "Bars show the mean of 5 folds, with a whisker for the spread. Grey bars are baselines, blue bagging, orange boosting, green meta-ensembles. " +
    "The boosting family and the meta-ensembles lead on F1 and ROC-AUC.");
  fig(s, "db_cv_comparison", M, 1.7, 8.4, 5.2);
  const x = M + 8.7, w = CW - 8.7;
  stat(s, x, 1.75, w, 1.2, f3(top["CV f1"]), `best CV F1 (${top.Model}) vs ${f3(CVm["Logistic Regression"]["CV f1"])} Logistic Regression`, C.orange, C.tint2);
  stat(s, x, 3.15, w, 1.2, f3(top["CV roc_auc"]), `CV ROC-AUC of ${top.Model} vs ${f3(CVm["Logistic Regression"]["CV roc_auc"])} LR`);
  stat(s, x, 4.55, w, 1.2, `± ${f3(top["CV F1 std"])}`, "standard deviation across folds: stable results");
}

// ───────────── 10. Test table
{
  const s = contentSlide("Held-out test set: all 12 models", 3,
    `Unseen ${fmt(S.test)} patients, default threshold 0.5. The ensembles lead on F1 and AUC; Logistic Regression has a good AUC ` +
    "but much lower recall; the single tree overfits.");
  const rows = test.map((r) => [r.Model, r.Family, pct(r.Accuracy, 2), f3(r.Precision), f3(r.Recall), f4(r.F1), f4(r["ROC-AUC"]), f4(r["PR-AUC"])]);
  const hi = test.findIndex((r) => r.Model === test[0].Model), mu = test.map((r, i) => (r.Family === "Baseline" ? i : -1)).filter((i) => i >= 0);
  table(s, ["Model", "Family", "Accuracy", "Precision", "Recall", "F1", "ROC-AUC", "PR-AUC"], rows, M, 1.7, [2.5, 1.6, 1.25, 1.25, 1.25, 1.35, 1.45, 1.48], 0.37, { numeric: true, size: 11.5, highlight: [hi], muted: mu });
  s.addText(`Highlighted: best F1 at threshold 0.5. Grey: baselines. McNemar test vs Logistic Regression: p = ${S.mcnemar.p.toExponential(1)}.`,
    { x: M, y: 6.65, w: CW, h: 0.3, fontFace: BF, fontSize: 11, color: C.muted, margin: 0, isTextBox: true });
}

// ───────────── 11. ROC/PR + confusion
{
  const cm = (p) => p; // placeholder for readability
  const s = contentSlide("Ranking quality and errors on unseen patients", 3,
    "Left: ROC and precision–recall curves. All ensembles hug the top-left corner. Right: confusion matrices. The tuned ensemble " +
    "finds more diabetics than the baseline while raising very few false alarms.");
  fig(s, "db_roc_pr", M, 1.7, CW, 2.75, "center");
  fig(s, "db_confusion", M, 4.55, CW, 2.4, "center");
}

// ───────────── 12. SHAP
{
  const sh = Object.entries(S.shap_mean_abs);
  const s = contentSlide("What the model learned: SHAP explanations", 4,
    "Each dot is a patient. Red means a high feature value, blue a low one. Points to the right push the prediction towards diabetes. " +
    "HbA1c and glucose dominate, followed by age, BMI and comorbidities, matching the medical literature.");
  fig(s, "db_shap_summary", M, 1.7, 6.0, 5.2);
  fig(s, "db_shap_dependence", M + 6.3, 1.7, CW - 6.3, 2.6, "center");
  textCard(s, M + 6.3, 4.5, CW - 6.3, 2.3, "Clinically sensible",
    `Top drivers by mean |SHAP|: ${sh.slice(0, 4).map(([k]) => k.replace(/_/g, " ")).join(", ")}. Risk jumps sharply at HbA1c ≈ 6.5% and glucose ≈ 200 mg/dL, ` +
    "the diagnostic thresholds used by doctors, which the model learned from data alone.", C.orange, C.tint2, 13);
}

// ───────────── 13. Conclusion
{
  const s = pres.addSlide(); n += 1;
  s.background = { color: C.navy };
  s.addText(RUBRIC[4], { shape: pres.shapes.ROUNDED_RECTANGLE, rectRadius: 0.08, x: M, y: 0.32, w: 6.1, h: 0.32, fill: { color: C.navy2 }, color: C.ice, fontFace: BF, fontSize: 11, bold: true, margin: [0, 8, 0, 8], valign: "middle", isTextBox: true });
  s.addText("Conclusion, limitations and next steps", { x: M, y: 0.72, w: CW, h: 0.8, fontFace: HF, fontSize: 30, bold: true, color: C.white, margin: 0, valign: "middle", isTextBox: true });
  const cw = (CW - 0.6) / 3;
  [["What we found", C.ice, [
    `${bestName} (chosen by CV): F1 ${f3(best.F1)}, ROC-AUC ${f3(best["ROC-AUC"])}, accuracy ${pct(best.Accuracy, 2)} on unseen patients.`,
    `Baselines: Logistic Regression F1 ${f3(lr.F1)}, Decision Tree F1 ${f3(dt.F1)}.`,
    "Every ensemble beat both baselines on F1 and ROC-AUC; boosting led; the gains are statistically significant.",
    "SHAP shows clinically meaningful drivers: HbA1c, glucose, age, BMI."]],
  ["Limitations", C.orange, [
    "Lab values recorded at a few discrete levels: results may be optimistic versus raw clinical data.",
    "Diabetics with normal-range labs remain the hardest cases (most false negatives).",
    "Single public dataset; no external hospital validation."]],
  ["Next steps", C.ice, [
    "Validate on an external clinical cohort.",
    "Cost-sensitive threshold for screening (favour recall).",
    "Deploy as a risk calculator with per-patient SHAP explanations."]]]
    .forEach(([h, col, items], i) => {
      const x = M + i * (cw + 0.3);
      s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x, y: 1.8, w: cw, h: 4.6, rectRadius: 0.1, fill: { color: C.navy2 }, line: { color: C.navy2 } });
      s.addText(h, { x: x + 0.3, y: 2.0, w: cw - 0.6, h: 0.45, fontFace: HF, fontSize: 18, bold: true, color: col, margin: 0, isTextBox: true });
      bullets(s, items, x + 0.3, 2.6, cw - 0.6, 3.7, 13.5, "E6ECF5");
    });
  s.addNotes("One line per finding; be upfront about limitations.");
}

// ───────────── 14. Thank you
{
  const s = pres.addSlide(); n += 1;
  s.background = { color: C.navy };
  s.addText("Thank you", { x: M, y: 2.3, w: CW, h: 1.2, fontFace: HF, fontSize: 54, bold: true, color: C.white, margin: 0, isTextBox: true });
  s.addText("Questions?", { x: M, y: 3.5, w: CW, h: 0.7, fontFace: BF, fontSize: 26, color: C.ice, margin: 0, isTextBox: true });
  s.addText("Group 10  ·  Sanskriti Rastogi  ·  Sahil Raj  ·  Yash Santosh Lohare  ·  Om Pathania", { x: M, y: 5.2, w: CW, h: 0.4, fontFace: BF, fontSize: 15, color: C.white, margin: 0, isTextBox: true });
  s.addText("Code: notebook 09_diabetes_ensembles · scikit-learn, XGBoost, LightGBM, CatBoost, SHAP", { x: M, y: 5.65, w: CW, h: 0.4, fontFace: BF, fontSize: 13, color: C.ice, margin: 0, isTextBox: true });
  s.addNotes("Likely questions: bagging vs boosting? why F1 not accuracy? how was the threshold chosen? what is SHAP? why does stacking not win by more?");
}

pres.writeFile({ fileName: OUT }).then(() => console.log("wrote", OUT, `(${n} slides)`));
