# BRFSS Heart Disease — Baseline Report

---

## 1. Tổng quan

| Hạng mục | Giá trị |
|---|---|
| Bài toán | Binary classification — heart disease (`Heart_Disease`) |
| Target raw | `_MICHD` (ưu tiên); fallback `CVDINFR4 OR CVDCRHD4` nếu `_MICHD` không có |
| Mã hoá target | `1 → 1.0` (positive), `2 → 0.0` (negative); `7, 9` và sentinel khác → NaN, drop |
| Dữ liệu | BRFSS 2019, 2020, 2021 (path qua `CONFIG["y2019/y2020/y2021"]`) |
| Lọc respondent | `DISPCODE ∈ {110, 1100}` (completed interview) |
| Yêu cầu feature | **Intersection across years** — feature phải resolve được ở MỌI năm load |
| Random seed | `42` (sklearn / xgboost / catboost / torch / cuda / numpy / DataLoader generator) |

---

## 2. Feature set (candidate, trước intersection)

`FEATURE_CANDIDATES` chứa **18 input features + 1 target**. Mỗi canonical feature có danh sách tên cột thô làm fallback giữa các năm:

| Canonical | Type | Raw column candidates |
|---|---|---|
| General_Health | ordinal | `GENHLTH` |
| Checkup | ordinal | `CHECKUP1` |
| Exercise | binary | `EXERANY2` |
| Skin_Cancer | binary | `CHCSCNCR` |
| Other_Cancer | binary | `CHCOCNCR` |
| Depression | binary | `ADDEPEV3` |
| Diabetes | nominal | `DIABETE4`, `DIABETE3` |
| Arthritis | binary | `HAVARTH5`, `HAVARTH4` |
| Sex | nominal | `SEXVAR`, `_SEX`, `BIRTHSEX` |
| Age_Category | ordinal | `_AGEG5YR`, `_AGE65YR`, `_AGE_G` |
| Height_(cm) | numeric | `HTM4`, `HTIN4`, `HEIGHT3` |
| Weight_(kg) | numeric | `WTKG3`, `WEIGHT2` |
| BMI | numeric | `_BMI5`, `BMI` |
| Smoking_History | binary | `SMOKE100` |
| Alcohol_Consumption | numeric | `_DRNKWK1`, `AVEDRNK3`, `DRNKANY5` |
| Fruit_Consumption | numeric | `FRUIT2`, `FRUIT1` |
| Green_Vegetables_Consumption | numeric | `FVGREEN1`, `FVGREEN` |
| FriedPotato_Consumption | numeric | `FRENCHF1`, `FRENCHF` |
| Heart_Disease (target) | – | `_MICHD` |

Survey weight: `SurveyWeight` ← raw `_LLCPWT`.

---

## 3. Preprocessing (decode rules)

| Decoder | Hành vi |
|---|---|
| Yes/No (Exercise, SkinCancer, OtherCancer, Depression, Arthritis, Smoking) | `1→1.0`, `2→0.0`; `7, 9, generic-sentinel` → NaN |
| General_Health | Đảo chiều: `1→4, 2→3, 3→2, 4→1, 5→0` (cao = khỏe); `7,9` → NaN |
| Checkup | `1→4, 2→3, 3→2, 4→1, 8→0` (8 = "Never"); `7,9,77,99,777,999` → NaN |
| Age_Category | `s − 1` nếu raw ∈ {`_AGEG5YR`, `_AGE65YR`, `_AGE_G`}; sentinel → NaN |
| Sex | `1→"Male"`, `2→"Female"`; `7, 9` → NaN |
| Diabetes | `1→"Yes"`, `2→"Pregnancy"`, `3→"No"`, `4→"Prediabetes"`; `7, 9` → NaN |
| Height | `HTM4`: trực tiếp cm; `HTIN4`: ×2.54; `HEIGHT3`: feet×12+inches → ×2.54, validate range feet [3,9], inches [0,11] |
| Weight | `WTKG3`: ÷100; `WEIGHT2`: nếu ≥9000 → ×1−9000 (kg), ngược lại ÷2.2046 (lb→kg) |
| BMI | `_BMI5`/`BMI`: ÷100 |
| Alcohol_Consumption | `_DRNKWK1`: mask 99900 → NaN; `AVEDRNK3`: replace 77,99 → NaN, mask 88 → 0 (None); `DRNKANY5`: 1→1, 2→0; replace 7,9 → NaN |
| Diet (Fruit / GreenVeg / FriedPotato) | Code BRFSS XXX: `1XX → (XX) per/day`; `2XX → (XX)/7`; `3XX → (XX)/30`; `555 → 0` (Never); `777, 999, sentinel` → NaN. Output: lần per ngày |

Sentinel set generic: `{7, 8, 9, 77, 88, 99, 777, 888, 999, 7777, 8888, 9999, 99900}` (áp tuỳ decoder; AVEDRNK3 không áp generic vì 88 mang nghĩa "None").

**Outlier clipping** (khi `clip_outliers=True`, default): Height ngoài `[140, 210]` cm → NaN; Weight ngoài `[45, 200]` kg → NaN.

**Drop rows** chỉ khi `Heart_Disease` thiếu (sau decode).

---

## 4. Intersection enforcement (across years)

Bước riêng sau merge:

1. Tính `resolved_by_year[y] = {feat | mapping[feat] resolve được, feat ≠ target}`.
2. `intersect_features = ∩ tất cả năm`.
3. Filter: `INPUT_FEATURES ← [f for f in INPUT_FEATURES if f in intersect_features]`.
4. Cùng filter cho `NOMINAL_COLS / BINARY_COLS / ORDINAL_COLS / NUMERIC_COLS / TT_CATEGORICAL_COLS / TT_CONTINUOUS_COLS`.
5. Trim `df_all` chỉ còn cột `INPUT_FEATURES + [TARGET, YEAR, WEIGHT]`.
6. Raise `ValueError` nếu intersection rỗng.
7. In log per-year: feature nào resolve, feature nào missing.

---

## 5. Split

**Hai chế độ** (chọn qua `CONFIG["split_mode"]`):

### Temporal (default; primary)
- `trainval = df[YEAR < temporal_test_year]`, `test = df[YEAR == temporal_test_year]` (default `2021`).
- Trong `trainval`: stratified split (`test_size=val_size_within_train=0.20`, `random_state=42`) → train + val.

### Random (sensitivity check)
- `train_test_split` stratified theo target, `test_size=0.20`, `random_state=42` → trainval / test.
- Trong trainval: stratified `test_size=0.20` → train / val.

**Survey weights** (`use_survey_weights=True`): build `w_train/w_val/w_test` từ cột `SurveyWeight`, `fillna(1.0)`, `w<=0 → 1.0`. Khi flag tắt → `None`.

`SplitData` có cả X/y/w cho 3 split + `train_years`, `test_years`.

---

## 6. Per-model preprocessing & training paradigm

### 6.1 LogReg / XGB (cùng pipeline preprocessing matrix)

**Matrix preprocessor** (`make_matrix_preprocessor`):

- `numeric_like = BINARY ∪ ORDINAL ∪ NUMERIC` (∩ `INPUT_FEATURES`):
  - `SimpleImputer(strategy="median")` (hoặc `add_indicator=True` nếu `preproc_variant="missing_indicator"`).
  - Chỉ với LogReg: thêm `StandardScaler(with_mean=False)`.
- `nominal = NOMINAL` (∩ `INPUT_FEATURES`):
  - `SimpleImputer(strategy="most_frequent")` (hoặc `strategy="constant", fill_value="Unknown"` nếu `explicit_unknown`).
  - `OneHotEncoder(handle_unknown="ignore")`.
- `ColumnTransformer(remainder="drop", sparse_threshold=0.3)`.

**LogReg**:
- `LogisticRegression(max_iter=4000, random_state=42)`; `class_weight="balanced"` khi `imbalance_mode="native_weight"`.
- Nếu `use_hpo=True`: `GridSearchCV` trên `HPO_GRIDS["logreg"] = {C: [0.01, 0.1, 1, 10], penalty: ["l2"], solver: ["lbfgs"]}`, `cv=StratifiedKFold(hpo_cv_folds=3)`, `scoring="roc_auc"`, `refit=True`, `n_jobs=-1`.
- `sample_weight=w_train` nếu có.

**XGB**:
- Khi không HPO: `XGBClassifier(n_estimators=1000, learning_rate=0.05, max_depth=5, subsample=0.8, colsample_bytree=0.8, reg_lambda=1.0, objective="binary:logistic", eval_metric="logloss", tree_method="hist", n_jobs=1, scale_pos_weight=ratio[native]/1.0[none], early_stopping_rounds=20)`. `fit` với `eval_set=[(Xt_val, y_val)]` và `sample_weight_eval_set=[w_val]`.
- Khi `use_hpo=True`: KHÔNG `early_stopping_rounds` (vì grid bao `n_estimators`); grid `{max_depth: [3,5,7], learning_rate: [0.05, 0.1], n_estimators: [300, 500], subsample: [0.8, 1.0], colsample_bytree: [0.8, 1.0]}`.

`ratio = neg/pos` (weighted nếu có survey weight).

### 6.2 CatBoost

**Frame prep** (`prepare_catboost_frame`):
- Nominal: `fillna("Unknown")`, cast str.
- Binary/Ordinal/Numeric: giữ NaN ở variant `baseline`; thêm `__missing` flag + median impute ở `missing_indicator`; chỉ median impute ở `explicit_unknown`.

**Training**:
- Khi không HPO: `CatBoostClassifier(iterations=1000, depth=6, learning_rate=0.05, loss_function="Logloss", eval_metric="AUC", random_seed=42, class_weights=[1.0, ratio]/None, thread_count=1, early_stopping_rounds=20)`. `fit` với `train_pool` (cat_features, weight) và `eval_set=val_pool`, `use_best_model=True`.
- Khi `use_hpo=True`: `cat_features` qua constructor; grid `{depth: [4,6,8], learning_rate: [0.05, 0.1], iterations: [300, 500]}`.

### 6.3 TabTransformer (PyTorch)

**Preprocess riêng**:
- Categorical (BINARY ∪ ORDINAL ∪ NOMINAL): cast str, NaN → `"__MISSING__"`. Vocab **chỉ fit từ train + thêm `"__UNK__"`** cho giá trị unseen ở val/test. Map về int.
- Continuous (NUMERIC_COLS): median impute (train median); standardize bằng (mean, std) train, std tối thiểu 1.0.

**Architecture**:
```
cat features → ModuleList(Embedding(card_i, d_token=32))
            → TransformerEncoder (n_layers=2, n_heads=4, d_model=32, ff=128, dropout=0.1, batch_first=True)
continuous → Linear(n_cont → d_token=32) → unsqueeze
concat + flatten → Linear(d_token × (n_cat + 1) → mlp_hidden=64) → ReLU → Dropout(0.1) → Linear(64 → 1)
```

**Training**:
- Seed: `torch.manual_seed(42)`, `torch.cuda.manual_seed_all(42)`, `np.random.seed(42)`, DataLoader `generator.manual_seed(42)`.
- Device: `cuda` nếu khả dụng, fallback CPU.
- Optimizer: `AdamW(lr=1e-3, weight_decay=1e-5)`.
- Loss: `BCEWithLogitsLoss(pos_weight=ratio if native_weight else None, reduction="none")` → mỗi sample × `w_batch` → tổng / sum(w) (weighted mean).
- Batch size: 1024.
- Early stopping: theo val `roc_auc_score` (weighted nếu có survey weight), `patience=5`, max `tt_epochs=25`. Save state khi val_auc cải thiện, load lại trước khi predict.

### 6.4 HPO scope summary

| Model | HPO hỗ trợ | Method | Grid |
|---|---|---|---|
| LogReg | Có | `GridSearchCV` 3-fold | `C ∈ {0.01, 0.1, 1, 10}` |
| XGB | Có | `GridSearchCV` 3-fold | `max_depth × lr × n_est × subsample × colsample` (3×2×2×2×2 = 48) |
| CatBoost | Có | `GridSearchCV` 3-fold | `depth × lr × iterations` (3×2×2 = 12) |
| TabTransformer | Không | – | – |

CV scoring = `roc_auc`, `refit=True`, `n_jobs=-1`. CV split: `StratifiedKFold(shuffle=True, random_state=42)`.

---

## 7. Evaluation

### Threshold tuning (`find_best_threshold`)
- Quét **181 ngưỡng** từ 0.05 đến 0.95 (linspace).
- Tính `Precision/Recall/F1` (weighted nếu có sample_weight) cho từng ngưỡng.
- Chọn ngưỡng tối đa metric chỉ định bởi `threshold_objective ∈ {"f1", "recall", "precision"}` (default `f1`).
- Tune trên **VAL**, áp lên test (không chạm test khi tune).

### Metrics output (per test set)

**Threshold-based** (compute_metrics, ở 0.5 và tuned threshold):
- Accuracy, Precision, Recall, F1, ConfusionMatrix.

**Probability-based** (compute_prob_metrics):
- ROC_AUC, PR_AUC, Prevalence (weighted hoặc không), **PR_AUC_Lift = PR_AUC / Prevalence**, Brier score.

**Visualization**:
- Confusion matrix @ 0.5 và @ tuned (matplotlib imshow).
- Calibration curve: 10 bins quantile, `sklearn.calibration.calibration_curve`.

Tất cả metric đều support `sample_weight`; khi `use_survey_weights=True` thì áp `_LLCPWT`.

---

# Đề xuất mở rộng — Hướng phát triển sau baseline

## 8. Filling missing features bị loại do intersection

**Vấn đề**: enforce intersection bỏ mất features lâm sàng quan trọng chỉ thiếu ở 1 năm (ví dụ `_RFCHOL3` cholesterol, `CVDSTRK3` stroke, một số biomarker rotating). Mục tiêu: **giữ feature + impute giá trị thiếu cho năm không có**.

**Setup ví dụ**: feature `High_Chol` có ở 2019+2020 (~670k rows) nhưng KHÔNG có ở 2021 (~355k rows). Cần fill 355k giá trị thiếu cho 2021 dựa trên dữ liệu 2019+2020.

### 8.1 KNN imputation

- **Cách hoạt động**: với mỗi row missing, tìm `K` láng giềng gần nhất trong tập có data (theo các features còn lại đã chuẩn hoá).
- **Continuous**: lấy mean của K láng giềng. **Discrete**: majority vote.
- **Per-sample**: mỗi row missing nhận một giá trị riêng dựa trên đặc điểm của row đó.
- **Implementation**: `sklearn.impute.KNNImputer(n_neighbors=5)`.
- **Ưu**: capture local pattern; **Nhược**: O(n²) chậm trên 1M rows, sensitive với feature scale.

### 8.2 Regression / Classification (MICE)

- **Cách hoạt động**: học `feature_missing = f(features khác)` từ data có đủ; predict cho data thiếu.
- **Continuous → regression** (BayesianRidge, LinearReg, RF Regressor).
- **Discrete → classification** (LogReg cho binary, multinomial cho multi-class).
- **Iterative**: lặp impute nhiều cột missing đan xen (MICE — Multiple Imputation by Chained Equations).
- **Implementation**: `sklearn.experimental.IterativeImputer` (mặc định regression-only, encode cat → int rồi snap về level hợp lệ); hoặc `miceforest` xử lý mixed-type natively bằng LightGBM.
- **Ưu**: chính xác cao, dùng đầy đủ thông tin features khác; **Nhược**: phức tạp khi mixed-type, chạy lâu hơn KNN.

### 8.3 Autoencoder imputation

- **Cách hoạt động**: train autoencoder reconstruct mọi features từ subset trên data có đầy đủ; với row thiếu, mask cột thiếu, feed forward, lấy output reconstructed làm imputed value.
- **Per-sample**: mỗi row được encode → decode riêng dựa trên features hiện có của nó.
- **Implementation**: PyTorch MLP autoencoder hoặc `denoising autoencoder` train trên 2019+2020, infer trên 2021.
- **Ưu**: capture non-linear dependencies; **Nhược**: cần training riêng, nhiều hyperparameters, ít interpretable.

### 8.4 Statistical fill (đơn giản, baseline impute)

- **Cách hoạt động**: dùng thống kê cố định trên cột.
  - Continuous → **median** của cột.
  - Ordinal → **median** (round nếu cần).
  - Binary / nominal → **mode** (giá trị xuất hiện nhiều nhất).
- **Variants**:
  - **Year-stratified**: tính median/mode RIÊNG cho từng năm có data, áp giá trị cho năm thiếu.
  - **Subgroup-stratified**: tính median/mode theo group `(Age × Sex)`, fill theo group của row missing.
- **Implementation**: `pandas.fillna()` + `series.mode()` / `series.median()`.
- **Ưu**: cực nhanh, ổn định, không model; **Nhược**: bỏ qua context per-row, kém chính xác.

### Protocol experiment (so với baseline-intersection-only)

| Run | Features | Imputation | Mục đích |
| --- | --- | --- | --- |
| 1 | Intersection 18 features | Không impute | Baseline reference |
| 2 | + K features non-intersection | A2 (statistical) | Giữ thêm features dễ |
| 3 | + K features non-intersection | A1.1 (KNN) | Test model-based đơn giản |
| 4 | + K features non-intersection | A1.2 (MICE) | Test model-based mạnh |
| 5 | + K features non-intersection | A1.3 (Autoencoder) | Test deep model-based |

**Metrics so**: Δ ROC-AUC, Δ PR-AUC Lift, Δ F1, Δ Brier so với baseline (Run 1).

---

## 9. Cải tiến SMOTE (theo slide `SMOTE.pdf`)

### 9.1 SMOTE gốc

- **Mục đích**: xử lý imbalanced data bằng over-sampling minority class.
- **Thuật toán**:
  - Với mỗi minority sample `x`, tìm `k = 5` láng giềng gần nhất.
  - Chọn ngẫu nhiên `x_neighbor` từ K láng giềng.
  - Sinh sample mới: `x_new = x + λ × (x_neighbor − x)`, `λ ∈ [0, 1]`.
- **Ưu**: không mất thông tin (không undersample), giảm overfit so với duplicate.
- **Nhược chính**: dữ liệu sinh ra **không phản ánh phân phối / đặc trưng** của dữ liệu gốc — đặc biệt khi minority nằm trên manifold non-linear, sample mới có thể rơi vào vùng "không hợp lý".

### 9.2 Các biến thể đã có

- **CHSMOTE**: Convex Hull-based SMOTE.
- **GMF-SMOTE**: Gaussian Mixture Filtering SMOTE.
- **G-SMOTE**: GMM-based SMOTE.

### 9.3 Hướng cải tiến đề xuất

| Bước | Kỹ thuật | Vai trò |
| --- | --- | --- |
| **1. Trích xuất đặc trưng** | Mô hình ngôn ngữ chuyển văn bản lâm sàng → vector ngữ nghĩa | Đưa data về không gian biểu diễn tốt hơn trước khi sinh sample |
| **2. Giảm chiều** | Autoencoder hoặc MLP nhẹ | Áp SMOTE trên không gian latent thay vì raw, tránh sample nằm sai vùng |
| **3. Phân cụm minority** | von Mises-Fisher Mixture Model | Phát hiện sub-population trong minority class, sinh sample contextual theo cụm |
| **4. Sinh sample mới** | Two-step sampling từ hàm mật độ xác suất chung | Sample mới phản ánh đúng phân phối, không nằm trên đường thẳng nội suy thuần như SMOTE gốc |

### 9.4 Tính mới (novel contribution)

1. **vMF-SMOTE**: dùng von Mises-Fisher distribution sinh dữ liệu mới thay vì nội suy tuyến tính — phù hợp khi feature space có cấu trúc directional / spherical.
2. **Neural feature extractor + SMOTE**: dùng mạng nơ-ron trích xuất đặc trưng, sau đó áp SMOTE trên latent space thay vì raw features — khắc phục nhược điểm SMOTE gốc trên dữ liệu high-dim với scale không đồng nhất.

### 9.5 Áp vào BRFSS heart disease

- Train autoencoder trên 18 features (sau preprocess) → latent `z ∈ ℝ^d` (`d = 8` hoặc `16`).
- Trên minority class (positive ~8%), fit von Mises-Fisher Mixture Model trên `z` → tìm `K` cụm bệnh nhân tương tự.
- Mỗi cụm sinh sample mới qua vMF sampling đến khi balance positive/negative.
- Decode về feature space (nếu cần) và đưa vào training pipeline thay cho `pos_weight`.

### Protocol experiment SMOTE-improved (so với baseline)

| Run | Imbalance handling | Mục đích |
| --- | --- | --- |
| 1 | `native_weight` (`pos_weight` / `class_weight`) | Baseline (tương đương baseline notebook) |
| 2 | SMOTE vanilla, áp **chỉ trên train** | Reference SMOTE đúng cách |
| 3 | SMOTENC (mixed-type variant) | Reference cho data có nominal |
| 4 | **vMF-SMOTE trên latent autoencoder** | Đề xuất chính |

**Metrics so**: Δ ROC-AUC, Δ PR-AUC Lift, Δ F1, Δ Brier; thêm **diversity / quality của synthetic samples** (ví dụ FID-like score so với phân phối gốc) để đánh giá nhược điểm cốt lõi của SMOTE gốc đã được khắc phục hay chưa.
