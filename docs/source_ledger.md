# Phase 0 Source Ledger

This ledger contains only the sources and facts supplied in the Phase 0 specification. The sources establish dataset and methodological context; they do not prove the project's hypotheses. No independent citations are added here.

## Primary dataset and benchmark

### WTBD — Wind Turbine Blade Defect dataset

- **Citation:** Ji, Lipeng; Cheng, Junjie; Wu, Shilong. “Multiclass Dataset for Intelligent Detection of Wind Turbine Blade Defects Using Drone Imagery.” *Scientific Data*, 2026.
- **Article DOI:** `10.1038/s41597-026-06762-x`
- **Dataset DOI:** `10.6084/m9.figshare.30210175`
- **Role:** sole primary dataset for the core experiment.
- **Contribution to this project:** provides a standardized benchmark with six visible surface-defect categories, real UAV imagery, PASCAL VOC bounding boxes, and a supplied standard split index. Its reported HOG/LBP feature analysis also supplies methodological context.
- **Frozen scope consequence:** WTBD contains no valid healthy/normal class, so the core task is six-class classification of annotated defect crops—not healthy-versus-damaged classification and not arbitrary full-image detection.

## Primary methodological literature

### Traditional image processing

- **Citation:** Deng, Liwei; Guo, Yangang; Chai, Borong. “Defect Detection on a Wind Turbine Blade Based on Digital Image Processing.” *Processes* 9(8), 1452, 2021.
- **DOI:** `10.3390/pr9081452`
- **Contribution to this project:** methodological context for traditional wind-turbine blade defect recognition using image processing, Log-Gabor-related processing, HOG features, and SVM; the supplied summary reports recognition above 92% for four defect types.

### Handcrafted versus transfer features

- **Citation:** “Defect identification of wind turbine blades based on defect semantic features with transfer feature extractor.” *Neurocomputing* 376, 2020.
- **DOI:** `10.1016/j.neucom.2019.09.071`
- **Contribution to this project:** compares transferred deep features with conventional HOG, SIFT, Tamura texture, and LBP features. It establishes that a handcrafted-versus-deep comparison is not itself a novel claim.

### CNN and limited-data precedent

- **Citation:** “Damage identification of wind turbine blades with deep convolutional neural networks.” *Renewable Energy* 174, 2021.
- **DOI:** `10.1016/j.renene.2021.04.040`
- **Contribution to this project:** context for a hierarchical computer-vision/CNN framework, comparisons with alternatives including SVM, and sensitivity analysis under limited data.

### Few-shot and scarcity precedent

- **Citation:** Gohar, Imad; Halimi, Abderrahim; Yew, Weng Kean; See, John. “Addressing Class Scarcity and Imbalance for Few-Shot Detection of Wind Turbine Blade Surface Defects.” *ISPACS 2025*.
- **DOI:** `10.1109/ISPACS68724.2025.11383379`
- **Contribution to this project:** demonstrates that scarce wind-turbine defect labels and few-shot detection are already active research topics.

### Lightweight CNN and Vision Transformer precedent

- **Citation:** Du, Liang; Lee, Soon-Hyung; Lee, Kyung-Min; Choi, Yong-Sung. “Lightweight CNN's Superiority in Industrial Defect Detection: A Case Study of Wind Turbine Blades.” *Machines* 14(1), 69, 2026.
- **DOI:** `10.3390/machines14010069`
- **Contribution to this project:** compares lightweight CNNs and Vision Transformers, including computational and data-efficiency considerations. It supports excluding Vision Transformers from this narrower benchmark rather than repeating a broad architecture leaderboard.

### Image-degradation precedent

- **Citation:** “A motion-blurred restoration method for surface damage detection of wind turbine blades.” *Measurement* 217, 2023.
- **DOI:** `10.1016/j.measurement.2023.113031`
- **Contribution to this project:** establishes motion blur as a relevant wind-turbine blade imaging problem and supplies context for controlled blur evaluation.

## Secondary and future datasets — not core training data

These datasets are possible later external-validation sources. They must not be combined with WTBD during core training.

### Blade30

- **Citation:** Yang, Cong et al. “Towards accurate image stitching for drone-based wind turbine blade inspection.” *Renewable Energy* 203, 267–279, 2023.
- **DOI:** `10.1016/j.renene.2022.12.063`
- **Supplied facts:** 1,302 real high-resolution drone images; 30 complete blades; multiple environmental settings; approximately 5400 × 3600 original resolution; the original publication reports 156 defects across 12 defect types; includes blade, defect, and contamination annotations.
- **Possible future contribution:** external or domain validation after scientifically valid class mapping is established.

### DTU Drone Inspection Images

- **Citation:** Shihavuddin, ASM; Chen, Xiao. “DTU - Drone inspection images of wind turbine.” *Mendeley Data*, 2018.
- **DOI:** `10.17632/hd96prn3nc.2`
- **Supplied facts:** drone inspection images of the same Nordtank turbine; temporal coverage in 2017 and 2018; CC BY-NC; later public annotation projects exist.
- **Possible future contribution:** optional external validation if class mapping and licensing permit scientifically valid evaluation.

### WTBs2025

- **Citation:** Zhang, Ruihua et al. “Wind turbine blades surface defect high-quality image dataset construction and performance validation.” *Scientific Data*, 2026.
- **Article DOI:** `10.1038/s41597-026-07455-1`
- **Dataset DOI:** `10.6084/m9.figshare.28876391`
- **Supplied facts:** 7,544 released images; nine surface-defect types; YOLO-style annotations; construction includes ordinary augmentation; GA-DCGAN augmentation was used because lightning-strike examples were scarce.
- **Scope restriction:** must not be the primary dataset for data-efficiency experiments. Augmentation and synthetic-image provenance could confound training-size comparisons or create leakage unless original-image ancestry is first verified.

## Ledger constraints

- WTBD is the only primary dataset for the core benchmark.
- Blade30, DTU, and WTBs2025 are future/external-validation candidates only.
- No source in this ledger proves H1–H5.
- No unsupported priority, novelty, state-of-the-art, or causal claim may be inferred from this ledger.
- New literature may be added only in a later, explicitly authorized phase or documented revision.
