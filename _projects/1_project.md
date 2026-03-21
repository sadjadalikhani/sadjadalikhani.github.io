---
layout: page
title: Large Wireless Model (LWM) Foundation Stack
description: A universal wireless representation learner covering baseband, spectrogram, and ray-tracing modalities.
img: assets/img/8.jpg
importance: 1
category: work
related_publications: true
---

The LWM project is the core of my Ph.D. research at Arizona State University. I co-designed a multi-modal data generation pipeline that fuses ray-tracing, digital twin context, and over-the-air captures to pretrain attention-based encoders for communication and sensing tasks. We maintain a fully reproducible stack that ships with:

- **Data curriculum:** staged sampling across environments, antenna topologies, and channel sparsity levels to keep the model stable during large-scale runs.
- **Sparse spatio-temporal attention blocks:** latency-aware transformers that outperform convolutional and recurrent baselines on channel prediction and beam selection.
- **Universal evaluation harness:** zero-shot transfer to dataset similarity estimation, channel subspace prediction, and interference diagnostics.

The resulting checkpoints power multiple publications (LWM, LWM-Temporal, LWM-Spectro) and serve as the backbone for the open-source releases on Hugging Face and lwm-wireless.net. I currently lead the roadmap for scaling the model, maintaining inference tooling, and coordinating collaborations with industry partners such as Nokia Bell Labs.
