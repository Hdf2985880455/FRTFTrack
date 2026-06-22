
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

## Main Components

### Reliability Estimation

![RE Module](figures/fig02.png)

### Fine-grained Text-guided Query Modulation

![FTQM Module](figures/fig03.png)

### Reliability-aware State Update

![RSU Module](figures/fig04.png)

## Method

FRTFTrack forms a closed temporal feedback loop consisting of prediction, reliability estimation, query modulation, and state update.

The reliability estimation module estimates the reliability of the current prediction. The reliability signal is then fed back into subsequent query construction and state update, enabling the tracker to adjust the tracking process according to historical prediction quality.

## Code

The code will be released upon acceptance.

