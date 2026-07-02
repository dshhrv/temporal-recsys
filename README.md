<div align="center">

# Temporal Recommendation Models

**Sequential, graph-based, and temporal graph recommendation experiments**

Experiments with SASRec, LightGCN, and TGN on MovieLens-1M under chronological full-catalog evaluation.

<p>
  <img src="https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python&logoColor=white" />
  <img src="https://img.shields.io/badge/PyTorch-Deep%20Learning-EE4C2C?logo=pytorch&logoColor=white" />
  <img src="https://img.shields.io/badge/Dataset-MovieLens--1M-FCC624" />
  <img src="https://img.shields.io/badge/Model-SASRec-5B8FF9" />
  <img src="https://img.shields.io/badge/Model-LightGCN-8E44AD" />
  <img src="https://img.shields.io/badge/Model-TGN-E67E22" />
</p>

[Overview](#overview) · [Models](#models) · [Pipeline](#pipeline) · [Evaluation](#evaluation) · [Results](#results) · [Project structure](#project-structure)

</div>

---

## Overview

This repository contains experiments with recommendation models based on three different approaches:

- sequential recommendation with SASRec
- graph collaborative filtering with LightGCN
- temporal graph recommendation with TGN

The experiments use MovieLens-1M interactions processed in chronological order. Models are evaluated through full-catalog ranking, where the target item is ranked against all candidate items not previously observed by the user.

The main goal is to compare how sequence-based, static graph-based, and temporal graph-based architectures behave on the same recommendation setting.

---

## Models

### SASRec

SASRec is a Transformer-based sequential recommendation model.

For each user, the model receives an ordered sequence of interactions and predicts the next item. The current experiment uses full cross-entropy training over the whole item catalog.

### LightGCN

LightGCN is a graph collaborative filtering model built on a user-item bipartite graph.

The model propagates user and item embeddings through graph layers without nonlinear transformations inside the message-passing stage.

### TGN

TGN is a Temporal Graph Network for dynamic user-item interactions.

The implementation includes:

- node memory updated after temporal interactions
- time encoding for interaction delays
- temporal neighborhood aggregation with attention
- edge features in messages and neighbor representations
- full-catalog scoring
- BPR and FullCE training modes

---

## Pipeline

```text
MovieLens-1M interactions
        ↓
Preprocessing and chronological split
        ↓
Train / validation / test events
        ↓
SASRec / LightGCN / TGN
        ↓
BPR or FullCE training
        ↓
Full-catalog ranking evaluation
        ↓
Recall@K, NDCG@K, MRR@K
```
