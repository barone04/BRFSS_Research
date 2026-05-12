# SMOTE Survey (2020–2026) — and Experiment Design on BRFSS

Báo cáo survey ngắn về **các cải tiến SMOTE 5 năm gần đây (2020–2026)**, đối chiếu với đề xuất hiện tại của bạn (`SMOTE.pdf`: vMF-SMOTE + Neural feature extractor + Two-step sampling joint density), và gợi ý **thiết kế thí nghiệm trên BRFSS heart disease**.

---

## 1. Background

### 1.1 SMOTE gốc (Chawla et al., 2002)

Cho mỗi minority sample `x`, chọn ngẫu nhiên 1 trong `k=5` nearest neighbors `x_neighbor`, sinh sample mới:

```
x_new = x + λ · (x_neighbor − x),   λ ∈ [0, 1]
```

### 1.2 Limitations đã được chỉ ra trong recent surveys

| Limitation | Nguồn |
|---|---|
| Bỏ qua local density / phân phối thật của minority | Nature Sci Rep 2025 [1] |
| Sinh duplicate / noisy samples ở vùng overlap | Mathematics (MDPI) 2024 [3] |
| Không phù hợp high-dimensional data | Frontiers Digital Health 2024 [4] |
| Sensitive với noise / outlier ở minority | PLOS One 2025 [5] |
| Clinical validity bị question (giảm signal) | MDPI MAKE 2024 [9] |

---

## 2. SMOTE Variants 2020–2026 (theo 5 hướng)

### 2.1 Spatial / Density-aware

Generate samples theo local density thay vì interpolation tuyến tính thuần.

| Method | Năm | Idea cốt lõi |
|---|---|---|
| **K-means SMOTE** | 2020 | Cluster minority bằng k-means, oversample chỉ trong cluster ít sample |
| **SD-KMSMOTE** | 2022 [10] | Spatial distribution của minority để chọn vùng synthesize |
| **MeanRadius-SMOTE** | 2022 [11] | Sinh trong hyper-sphere quanh từng minority point, bán kính theo mean distance |
| **ISMOTE** | 2025 [1] | Mở rộng không gian sinh: không chỉ trên đường thẳng giữa 2 points mà còn quanh chúng |
| **FLEX-SMOTE** | 2024 [2] | Adaptive theo density của vùng minority — works với mọi distribution |
| **DDSC-SMOTE** | 2024 | Spectral clustering + data distribution analysis |

**Đóng góp chung**: refining "WHERE to sample".

### 2.2 Borderline / Safe-level / Boundary-aware

Tập trung vào quyết định khu vực nào nên / không nên synthesize.

| Method | Idea |
|---|---|
| **Borderline-SMOTE** | Sinh samples gần decision boundary (where minority dễ bị nhầm) |
| **Safe-Level-SMOTE** | Mỗi minority có "safe level" — sinh theo trọng số safe-level |
| **CRN-SMOTE** | 2025 [5] | Cluster-Based Reduced Noise SMOTE — combine SMOTE + cluster-noise filter |
| **Counterfactual SMOTE** | 2025 [13] | Generate near decision boundary trong "safe region" qua counterfactual framework |

**Đóng góp chung**: refining "WHICH minority points" để sinh.

### 2.3 Hybrid (resampling + cleaning)

| Method | Idea |
|---|---|
| **SMOTE-ENN** | Sinh bằng SMOTE → loại bằng Edited Nearest Neighbors |
| **SMOTE-Tomek** | Sinh + remove Tomek links |
| **SMOTE-kTLNN** | 2023 [14] | SMOTE + Two-Layer Nearest Neighbor classifier + Iterative-Partitioning Filter |

**Đóng góp chung**: clean noise AFTER oversampling.

### 2.4 Deep Generative (GAN/VAE/Diffusion-based)

Thay nội suy bằng generative model.

| Method | Năm | Idea |
|---|---|---|
| **DeepSMOTE** | 2022 | Encoder-decoder + SMOTE trên latent space |
| **CGAN / CTGAN** | 2023 [15] | Conditional GAN sinh samples theo class label — best performance trên clinical AUC/F1 |
| **Majority-Guided VAE** | 2023 [16] | VAE đặt majority làm guide để sinh minority đa dạng hơn |
| **BM-WGAN** | 2024 [17] | Bootstrap + Wasserstein GAN cho minority distribution estimation |
| **IGAN-EDELM** | 2024 | Multi-head attention GAN + extreme learning machine cho fault diagnosis |
| **DDPM-based oversampling** | 2024 [18] | Diffusion model + greedy K sampling cho small imbalanced medical images |

**Đóng góp chung**: replace linear interpolation bằng learned distribution.

### 2.5 Distribution-aware / Theoretical

Phân tích/sinh từ **phân phối xác suất thực**.

| Method | Năm | Idea |
|---|---|---|
| **SMOTE probability distribution analysis** | 2022 [12] | Derive mathematical form của SMOTE patterns → so với true class-conditional density |
| **G-SMOTE** (Geometric SMOTE) | 2020 | Sinh trong hyper-spheroid quanh minority — generalized geometric region |
| **GMM-SMOTE / GMF-SMOTE** | Pre-2020/2020+ | Gaussian Mixture Model phân cụm minority, sinh từ component-wise |
| **CHSMOTE** | 2021 | Convex Hull-based SMOTE |
| **Đề xuất của bạn (vMF-SMOTE)** | 2026 | von Mises-Fisher Mixture trên minority + neural feature extractor + two-step sampling từ joint density |

**Đóng góp chung**: thay linear interpolation bằng sampling từ explicit/learned probability distribution.

---

## 3. Đề xuất của bạn — vMF-SMOTE — đứng ở đâu trong literature?

### 3.1 Tính độc đáo

| Khía cạnh | vMF-SMOTE (đề xuất) | Closest baselines |
|---|---|---|
| Distribution | **von Mises-Fisher** (directional, spherical) | GMM-SMOTE (Gaussian); G-SMOTE (geometric) |
| Feature space | **Latent từ neural encoder** (low-dim semantic) | DeepSMOTE (linear AE); CTGAN (no encoder) |
| Sampling | **Two-step**: chọn cluster → sample từ vMF component | GMM-SMOTE 1-step; CTGAN end-to-end |
| Tính ngữ nghĩa | Latent giữ structure của text/code lâm sàng | Chưa được explore |

### 3.2 Niche chưa được giải quyết bởi literature 2020–2026

1. **Directional/spherical structure ở minority class**: tất cả SMOTE hiện có giả định Euclidean. Khi feature space có cấu trúc directional (ví dụ embedding lâm sàng đã được L2-normalize), vMF là natural choice.
2. **Two-step sampling**: tách "chọn mode (cluster)" và "sample trong mode" → đa dạng hơn 1-step GMM.
3. **Semantic latent space**: dùng neural encoder cho structured medical data (đặc biệt khi có text field) — chưa thấy paper nào kết hợp.

### 3.3 Risks / Counter-arguments cần defense trong paper

- vMF tốt cho data directional, nhưng BRFSS đa phần tabular numeric/binary — có thực sự cần directional? **Defense**: encoder neural mapping → unit hypersphere bằng L2 norm.
- 2-step sampling phức tạp hơn 1-step; computational cost cao hơn → cần benchmark thời gian.
- vMF sampling trên high-dim sphere có thể bị curse of dimensionality.

---

## 4. Evaluation Metrics — beyond Acc/Pre/Rec/F1

Theo gợi ý của thầy bạn, ngoài metrics cơ bản, cần report:

### 4.1 Synthetic sample quality

| Metric | Mô tả | Áp dụng tabular thế nào |
|---|---|---|
| **FID** (Fréchet Inception Distance) [21] | Distance giữa distribution của synthetic vs real trong feature space của Inception network | Không trực tiếp dùng — Inception trained on ImageNet, không phù hợp tabular [22] |
| **FID-tabular** | Mean/cov của real vs synthetic trên **AE latent** thay vì Inception | ✅ Compute được trên BRFSS qua AE encoder |
| **MMD** (Maximum Mean Discrepancy) | Kernel-based distribution distance | Standard cho tabular synthetic data |
| **Wasserstein-2 distance** | Earth mover's distance giữa 2 distribution | Cùng họ với FID, hữu dụng |
| **Energy distance** | Distribution-free distance | Robust với kernel choice |

### 4.2 Class overlap / margin

| Metric | Mô tả |
|---|---|
| **Class Overlap Score** | % synthetic samples nằm trong region của majority class (gần neighbor majority) |
| **Margin Violation Rate** | % synthetic samples lọt qua decision boundary của classifier baseline |
| **Silhouette score (minority)** | Cụm minority có còn tách biệt sau khi thêm synthetic không |
| **kNN purity** | k láng giềng gần nhất của synthetic có thuộc minority không |

### 4.3 Uncertainty / Calibration

| Metric | Mô tả |
|---|---|
| **Prediction confidence distribution** | Histogram của max(p, 1-p) trên test → low confidence concentration cho thấy model unsure |
| **Conformal Prediction Coverage** [25] | Probability rằng true label nằm trong prediction set ở mức 1-α (e.g., 90%) |
| **Class-conditional coverage** [26] | Coverage tách riêng cho minority/majority — kiểm tra fairness |
| **Brier score** | Squared error giữa probability và label — đo calibration |
| **ECE** (Expected Calibration Error) | Bin probabilities, đo gap giữa confidence vs accuracy |

### 4.4 Efficiency

| Metric | Mô tả |
|---|---|
| **Sample generation time** | Thời gian sinh N synthetic samples |
| **Total pipeline time** | Generation + training + inference |
| **Memory footprint** | Peak GPU/CPU memory |

---

## 5. Thiết kế thí nghiệm trên BRFSS

### 5.1 Setup data

- **Dataset**: BRFSS 2019 + 2020 + 2021, intersection 15 features (sau preprocess baseline).
- **Split**: temporal — train = 2019+2020, val = 20% in train, test = 2021. **Stratified**.
- **Target**: `Heart_Disease` (`_MICHD`), prevalence ~8.4%.

### 5.2 Pipeline experiment

```
df_train (imbalanced, ~9% positive)
    ↓
[Resampling method] → balanced train (or oversampled minority)
    ↓
Train 4 models: LogReg, XGB, CatBoost, TabTransformer
    ↓
Evaluate trên test 2021
```

### 5.3 Method comparison (8 setups)

| # | Method | Mô tả |
|---|---|---|
| 1 | **No resample** (baseline) | Không oversampling, dùng `class_weight` native |
| 2 | **Random oversampling** | Duplicate random minority |
| 3 | **SMOTE vanilla** | Standard SMOTE |
| 4 | **SMOTE-Tomek** | SMOTE + Tomek link cleaning |
| 5 | **SMOTE-ENN** | SMOTE + Edited NN cleaning |
| 6 | **Borderline-SMOTE** | Borderline-aware |
| 7 | **CTGAN** | Conditional Tabular GAN [15] |
| 8 | **vMF-SMOTE (đề xuất)** | Neural encoder + vMF Mixture + 2-step sampling |

**Important**: oversample CHỈ trên train, KHÔNG đụng val/test (tránh leak như paper jcdd-11-00396).

### 5.4 Metrics report

Mỗi method × 4 model = 32 cells. Mỗi cell report:

**Classification metrics**:
- Accuracy, Precision, Recall, F1 (tuned threshold)
- AUC-ROC
- AUC-PR (PR-AUC Lift = PR-AUC / prevalence)
- Brier score

**Synthetic quality** (chỉ cho method có generate, runs 3–8):
- FID-tabular (qua AE encoder fit on real data)
- MMD-RBF
- Wasserstein-2 distance

**Class overlap**:
- Class overlap score (% synthetic gần majority neighbor)
- Margin violation rate (% synthetic lọt qua boundary của baseline XGB)

**Uncertainty**:
- Class-conditional conformal coverage (target 90% cho cả 2 class)
- ECE on test

**Efficiency**:
- Sample generation time (per 100k samples)
- Total pipeline time (train + test)
- Peak memory

### 5.5 Ablation cho vMF-SMOTE

Để justify từng component:

| Setup | Encoder | Distribution | Sampling |
|---|---|---|---|
| (a) vanilla SMOTE | – | linear | 1-step |
| (b) + neural encoder | ✅ AE | linear | 1-step |
| (c) + vMF | ✅ AE | vMF | 1-step |
| (d) + 2-step (full) | ✅ AE | vMF | 2-step |

→ Δ giữa (a)→(b), (b)→(c), (c)→(d) cho thấy đóng góp riêng của từng component.

### 5.6 Criterion for "vMF-SMOTE wins"

Theo thầy yêu cầu "tối thiểu tốt hơn 10% trong nhiều case study":

- **Strong claim**: PR-AUC Lift của vMF-SMOTE > +10% relative so với vanilla SMOTE trên ít nhất 2/4 models.
- **Weaker but defensible**: F1-tuned > +5% relative, plus FID < FID(vanilla SMOTE), plus margin violation rate thấp hơn.

### 5.7 Statistical rigor

- Bootstrap CI 95% cho AUC trên test (1000 iterations).
- DeLong test so AUC giữa method baseline (no resample) và vMF-SMOTE.
- Wilcoxon signed-rank test trên 4 models × 5 random seeds.

### 5.8 Computational budget

| Phase | Time estimate |
|---|---|
| Encoder pretraining (AE) | ~10 phút (CUDA) |
| vMF Mixture fit (minority subset) | < 5 phút |
| Sample generation per method | 1–10 phút |
| Downstream training (4 models × 8 methods) | ~8 giờ |
| Metrics computation | ~30 phút |
| **Total** | **~10 giờ Colab GPU** |

---

## 6. References (đầy đủ cho paper)

[1] Improved SMOTE with expanded sample generation space, *Scientific Reports* (Nature) 2025. https://www.nature.com/articles/s41598-025-09506-w

[2] Liu et al. **FLEX-SMOTE**: Synthetic over-sampling technique that flexibly adjusts to different minority class distributions, *PMC* 2024. https://pmc.ncbi.nlm.nih.gov/articles/PMC11573909/

[3] Strategic application of SMOTE and its variants to enhance AI, *Issues in Information Systems* 2025. https://iacis.org/iis/2025/2_iis_2025_70-85.pdf

[4] A review on over-sampling techniques in classification of multi-class imbalanced datasets: insights for medical problems, *Frontiers Digital Health* 2024. https://www.frontiersin.org/journals/digital-health/articles/10.3389/fdgth.2024.1430245/full

[5] **CRN-SMOTE** (Cluster-Based Reduced Noise SMOTE), *PLOS One* 2025. https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0317396

[6] Handling imbalanced medical datasets: review of a decade of research, *AI Review* (Springer) 2024. https://link.springer.com/article/10.1007/s10462-024-10884-2

[7] A survey on imbalanced learning: latest research, applications and future directions, *AI Review* (Springer) 2024. https://link.springer.com/article/10.1007/s10462-024-10759-6

[8] Do we need rebalancing strategies? A theoretical and empirical study around SMOTE and its variants, *OpenReview* 2024. https://openreview.net/forum?id=uLAAVg0ymc

[9] Impact of nature of medical data on ML/DL for imbalanced datasets: clinical validity of SMOTE is questionable, *MDPI MAKE* 2024. https://www.mdpi.com/2504-4990/6/2/39

[10] **SD-KMSMOTE**: spatial distribution of minority samples, *Scientific Reports* 2022. https://www.nature.com/articles/s41598-022-21046-1

[11] **MeanRadius-SMOTE**: oversampling for mechanical fault diagnosis, *PMC* 2022. https://pmc.ncbi.nlm.nih.gov/articles/PMC9324964/

[12] Sakho et al. **Theoretical distribution analysis of SMOTE for imbalanced learning**, *Machine Learning* (Springer) 2022. https://link.springer.com/article/10.1007/s10994-022-06296-4

[13] Counterfactual synthetic minority oversampling technique, *Healthcare Analytics* (ScienceDirect) 2025. https://www.sciencedirect.com/science/article/pii/S2666764925000062

[14] **SMOTE-kTLNN**: hybrid re-sampling with two-layer nearest neighbor classifier, *Expert Systems with Applications* (Elsevier) 2023. https://www.sciencedirect.com/science/article/abs/pii/S0957417423023503

[15] **CGAN/CTGAN** for clinical imbalanced data, *Mathematics* (MDPI) 2023. https://www.mdpi.com/2227-7390/11/16/3605

[16] **Majority-Guided VAE** for generative oversampling, *arXiv* 2023. https://arxiv.org/abs/2302.10910

[17] **BM-WGAN**: Bootstrap + Wasserstein GAN for imbalanced data, *Mathematical Biosciences and Engineering* 2024. https://www.aimspress.com/aimspress-data/mbe/2024/3/PDF/mbe-21-03-190.pdf

[18] DDPM + greedy K sampling for small imbalanced medical image datasets, *arXiv* 2024. https://arxiv.org/html/2412.12532v1

[19] Deep generative approaches for oversampling in imbalanced data: comprehensive review, *Applied Soft Computing* (ScienceDirect) 2024. https://www.sciencedirect.com/science/article/abs/pii/S1568494624014510

[20] The use of generative adversarial networks to alleviate class imbalance in tabular data: a survey, *Journal of Big Data* 2022. https://journalofbigdata.springeropen.com/articles/10.1186/s40537-022-00648-6

[21] **Fréchet Inception Distance** original paper, Heusel et al. 2017 (background). https://en.wikipedia.org/wiki/Fr%C3%A9chet_inception_distance

[22] Compound FID for quality, *arXiv* 2021 — discusses limitations of FID on non-ImageNet domains. https://arxiv.org/pdf/2106.08575

[23] Evaluating IS and FID metrics for quality and diversity, *ACM* 2024. https://dl.acm.org/doi/10.1145/3708778.3708790

[24] Synthetic clinical data generation with GANs (HIV antiretroviral therapy), *PubMed* 2023. https://pubmed.ncbi.nlm.nih.gov/37451495/

[25] **Class-Conditional Conformal Prediction for Imbalanced Data via Top-k Classes**, *OpenReview* 2024. https://openreview.net/forum?id=Dtxc7mlKRg

[26] Conformal Inference for Open-Set and Imbalanced Classification, *arXiv* 2025. https://arxiv.org/html/2510.13037v1

[27] Conformal uncertainty quantification for predictive fairness across patient demographics, *Health Information Science and Systems* (Springer) 2025. https://link.springer.com/article/10.1007/s13755-025-00412-z

[28] Noise-Adaptive Conformal Classification with Marginal Coverage, *arXiv* 2025. https://arxiv.org/html/2501.18060v1

[29] Class-Conditional Conformal Prediction with Many Classes (Berkeley statistics). https://www.stat.berkeley.edu/~ryantibs/papers/classconf.pdf

[30] Iacobescu et al. Evaluating Binary Classifiers for CVD Prediction (BRFSS 2021), *JCDD* 2024 (warning: SMOTE-before-split leakage example). https://doi.org/10.3390/jcdd11120396

---

## 7. Tóm tắt 1 trang cho thầy

**Survey 2020–2026** chia thành 5 hướng cải tiến SMOTE:

1. **Spatial/density-aware** (ISMOTE, FLEX-SMOTE, K-means SMOTE)
2. **Borderline/safe-level** (Borderline-SMOTE, CRN-SMOTE, Counterfactual SMOTE)
3. **Hybrid clean** (SMOTE-ENN, SMOTE-Tomek, SMOTE-kTLNN)
4. **Deep generative** (CTGAN, BM-WGAN, DeepSMOTE, DDPM)
5. **Distribution-aware** (GMM-SMOTE, Probability-analysis, **đề xuất vMF-SMOTE**)

**Đề xuất vMF-SMOTE chiếm niche chưa được khai thác**:
- Directional structure (vMF) thay vì Euclidean assumption.
- Neural encoder cho semantic latent space.
- Two-step sampling từ joint probability density.

**Experiment trên BRFSS** (8 methods × 4 models), report metrics đầy đủ:
- Classification: Acc, Pre, Rec, F1, AUC, PR-AUC Lift, Brier
- Synthetic quality: **FID-tabular, MMD, Wasserstein**
- Class overlap: **margin violation rate, kNN purity**
- Uncertainty: **conformal coverage, ECE**
- Efficiency: generation time, memory

**Success criterion**: vMF-SMOTE > +10% relative PR-AUC Lift so với vanilla SMOTE trên ít nhất 2/4 models, plus FID lower, plus margin violation lower.

---

## Bước tiếp theo

1. Bạn xem README này, confirm có muốn:
   - Thêm/bớt method nào trong comparison (8 methods)?
   - Thêm/bớt metric nào?
   - Có cần extend survey thêm (deep dive vào 1-2 paper cụ thể)?

2. Sau khi confirm → tôi code skeleton notebook `smote_experiment.ipynb` với:
   - Pipeline 8 methods × 4 models
   - All metrics (classification + synthetic quality + uncertainty + efficiency)
   - Output table side-by-side

Reply để tiếp tục.
