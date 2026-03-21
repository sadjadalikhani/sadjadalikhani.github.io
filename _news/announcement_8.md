---
layout: post
title: Winners of the LWM multi-task optimization challenge 2025 announced
date: 2025-10-30 16:11:00-0400
inline: false
news_url: https://www.linkedin.com/feed/update/urn:li:activity:7394079010152136704/?originTrackingId=z2fLKm5%2BUdnWRQ2%2F2G7%2F2w%3D%3D
related_posts: false
---


## 1. What was this challenge about?

The LWM challenge was designed to test how well **one wireless foundation model**
can generalize across several **different tasks** rather than being tuned for only one.

In short:

- Participants started from a **pretrained LWM-based backbone**.
- They were given **multiple wireless tasks** (e.g., beam prediction, channel estimation,
  localization, classification).
- They were allowed to design **lightweight task-specific heads** on top of the backbone.
- Final ranking depended on **multi-task performance**, not just one leaderboard metric.

The main question:

> Can we build wireless models that act as *universal feature extractors*,
> performing well across many tasks with minimal task-specific tuning?

<hr />

## 2. High-level setup

To keep things fair and focused on generalization:

- Teams received:
  - Training and validation splits per task.
  - Hidden test sets for evaluation.
- For the test phase:
  - Teams extracted **embeddings** for the test inputs using their model.
  - They submitted:
    - The embeddings
    - The small task heads
    - Configuration files and logs (for reproducibility).

The evaluation pipeline on our side:

1. Load the submitted head(s).
2. Apply them to the submitted embeddings.
3. Compute task-specific metrics (e.g., NMSE, accuracy, top-k, etc.).
4. Aggregate metrics into a single **multi-task score**.

This way:

- Large pretrained backbones could remain **private/confidential**.
- The comparison focused on **feature quality and generalization**.
- Everyone played under the **same protocol**.

<hr />

## 3. Tasks at a glance

Each team had to handle **multiple wireless tasks** using the same general backbone:

- Task A: Sub-6 GHz to mmWave **beam prediction**
- Task B: **Channel estimation** in compressed/feedback-limited settings
- Task C: **Localization / positioning** from channel features
- Task D: **LoS / NLoS classification** (and related variants)

Key idea:

> If a model is truly “foundational”, a single representation space should work
> well for all of these tasks with only small heads on top.

<hr />

## 4. Ranking and scoring

The final score was a combination of:

- Per-task metrics (normalized to comparable ranges)
- Aggregated into a **multi-task composite score**

Informally:

- Models that did extremely well on just one task but poorly on others
  did **not** rank high.
- Models with strong, consistent performance across all tasks
  were ranked at the top.

This encouraged:

- Robust **generalization**
- Careful **regularization**
- Thoughtful **architecture and training** of the LWM-based backbone

<hr />

## 5. What did we learn?

Some key observations from the 2025 edition:

- **Pretrained LWM-style backbones** can generalize surprisingly well
  across tasks that were not explicitly targeted during pretraining.
- Careful **fine-tuning strategies** (e.g., freezing most layers, adapting only
  the last few) often outperformed heavy, full-model retraining.
- Simpler heads with good regularization were competitive — and sometimes
  better — than very deep task-specific stacks.
- Dataset design and **consistent evaluation protocols** are crucial to
  meaningfully compare “universal” models.


