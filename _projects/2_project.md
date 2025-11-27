---
layout: page
title: Digital Twin Aided Channel Estimation
description: Zone-specific subspace prediction and calibration strategies for RIS-assisted networks.
img: assets/img/10.jpg
importance: 2
category: work
related_publications: true
---

This project demonstrates how a high-fidelity digital twin can shoulder much of the CSI burden in future networks. Working with Prof. Ahmed Alkhateeb, I designed a two-stage pipeline:

1. **Twin-guided subspace prediction.** The twin provides coarse geometry-informed priors. We compress them into zone-specific latent subspaces that require minimal over-the-air pilots to select.
2. **Online calibration.** Sparse live measurements update the twin’s belief and correct for hardware detuning, RIS coupling, and user mobility.

When paired with our robust beamforming solver, the approach reduces feedback overhead by up to 70 % in multi-user RIS deployments while keeping outage probability in check. The full methodology is detailed in our ICMLCN and VTC papers and is now the baseline for several follow-on projects inside the lab.
