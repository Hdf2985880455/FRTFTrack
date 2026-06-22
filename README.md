
# FRTFTrack: Reliability-Aware Temporal Feedback for Data-Efficient Vision-Language Tracking

This repository provides the project page for **FRTFTrack**, a reliability-aware temporal feedback framework for data-efficient vision-language tracking.

## Overview

FRTFTrack is built upon the generative vision-language tracking baseline **MMTrack**. It keeps the original vision-language backbone unchanged and introduces three lightweight components:

- **RE**: Reliability Estimation
- **FTQM**: Fine-grained Text-guided Query Modulation
- **RSU**: Reliability-aware State Update

The goal of FRTFTrack is to improve tracking stability and data efficiency under low-shot training settings.

## Framework

![FRTFTrack Framework](figures/fig01.png)

**Figure 1. Overall framework of FRTFTrack.** FRTFTrack is built upon MMTrack and introduces a reliability-aware temporal feedback loop. The estimated reliability is fed back to guide subsequent query modulation and state update, improving tracking stability under low-shot training settings.

## Main Components

### Reliability Estimation

![RE Module](figures/fig02.png)

**Figure 2. Reliability Estimation (RE).** RE estimates the reliability of the current prediction from aggregated decoder-state features and sentence-level text semantics. The reliability score provides feedback signals for query modulation and state update.

### Fine-grained Text-guided Query Modulation

![FTQM Module](figures/fig03.png)

**Figure 3. Fine-grained Text-guided Query Modulation (FTQM).** FTQM uses historical reliability to guide word-level text-token selection and generate a dynamic query residual. This module enhances language-aware query construction while controlling the influence of unreliable temporal states.

### Reliability-aware State Update

![RSU Module](figures/fig04.png)

**Figure 4. Reliability-aware State Update (RSU).** RSU controls the final tracking state according to prediction reliability and motion variation. When reliability is low and motion is abnormal, conservative fusion is used to reduce error propagation.

## Method

FRTFTrack forms a closed temporal feedback loop consisting of prediction, reliability estimation, query modulation, and state update.

The reliability estimation module estimates the reliability of the current prediction. The reliability signal is then fed back into subsequent query construction and state update, enabling the tracker to adjust the tracking process according to historical prediction quality.

## Results

FRTFTrack is evaluated on TNL2K, LaSOT, and OTB99-Lang under both full-data and low-shot training settings.

Under the full-data training setting, FRTFTrack maintains performance comparable to MMTrack while introducing only lightweight reliability-aware feedback modules. Under low-shot training settings, FRTFTrack achieves more pronounced improvements, showing better data efficiency when training samples are limited.

### Low-shot AUC comparison

| Setting | Dataset | MMTrack AUC | FRTFTrack AUC | Gain |
|---|---:|---:|---:|---:|
| 1/6 samples | TNL2K | 49.6 | 51.7 | +2.1 |
| 1/6 samples | LaSOT | 62.8 | 64.9 | +2.1 |
| 1/6 samples | OTB99-Lang | 65.2 | 68.7 | +3.5 |
| 1/3 samples | TNL2K | 51.0 | 53.9 | +2.9 |
| 1/3 samples | LaSOT | 65.0 | 67.6 | +2.6 |
| 1/3 samples | OTB99-Lang | 67.0 | 69.0 | +2.0 |

### Low-shot ablation on AUC

| Method | TNL2K | LaSOT | OTB99-Lang |
|---|---:|---:|---:|
| Baseline | 49.6 | 62.8 | 65.2 |
| +RE | 49.2 | 61.7 | 65.4 |
| +RE+FTQM | 51.5 | 64.7 | 68.7 |
| +RE+FTQM+RSU | 51.7 | 64.9 | 68.7 |

The ablation results show that RE alone mainly provides an auxiliary reliability signal, while the closed feedback loop formed by RE, FTQM, and RSU brings more consistent improvements under low-shot training.

## Paper

The manuscript is currently under submission.

## Citation

The citation will be updated after publication.

## Contact

For questions about this project, please contact Tianpeng Liu at tpliu@wtu.edu.cn.

## Code

The code will be released upon acceptance.

