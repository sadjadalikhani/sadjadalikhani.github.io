---
layout: page
title: Dataset Similarity Evaluation Framework
description: Quantifying transferability between wireless datasets for communication and sensing tasks.
img: assets/img/6.jpg
importance: 3
category: work
related_publications: true
---

Wireless labs often collect data in isolation, making it hard to predict whether a pre-trained model will generalize to a new environment. I worked with collaborators at ASU and Bell Labs to build a dataset similarity evaluation framework that:

- Represents datasets via task-aligned fingerprints spanning spectrogram statistics, spatial correlations, and semantic metadata.
- Uses LWM embeddings to score cross-dataset affinity and recommend fine-tuning targets.
- Ships with automated reports so practitioners can quickly understand gaps before launching expensive data collection campaigns.

The framework is now part of our open benchmark, appears in the Asilomar 2024 paper, and is actively used when planning new twin deployments or foundation model experiments.
