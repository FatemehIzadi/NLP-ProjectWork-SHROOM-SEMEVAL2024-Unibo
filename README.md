# SHROOM Hallucination Detection  

## Overview
This project presents a system for detecting hallucinations in model-generated text for **SemEval-2024 Task 6 — SHROOM**.  
The task is to classify whether a generated *hypothesis* is faithful to a given *source* and *reference* across three NLG settings:

- **Definition Modeling (DM)**
- **Paraphrase Generation (PG)**
- **Machine Translation (MT)**

A major challenge is that the training data is **unlabeled**, requiring pseudo-labeling, confidence filtering, and hybrid modeling strategies.

---

## Notebooks Overview

### **1. Baseline, Data Exploration & Initial Models**  
**`shroom_notebook.ipynb`**  
This notebook contains the full exploratory and baseline workflow:
- Dataset inspection and **EDA**
- Initial preprocessing pipeline  
- **DistilBERT** student training using pseudo-labels  
- **TF-IDF + Logistic Regression** baseline  
- **Mistral-7B prompting** for pseudo-label generation and zero-shot evaluation  

---

### **2. CatBoost Feature-Based Classifier**  
**`shroom-catboost.ipynb`**  
This notebook includes the feature-engineering pipeline and CatBoost classifier:
- SBERT embedding similarity features  
- RoBERTa-MNLI logits as features  
- Lexical overlap features  
- CatBoost training, tuning, and evaluation  
- Produces one of the two final models used in the ensemble  

---

### **3. DeBERTa Student + Final Ensemble**  
**`shroom-deberta.ipynb`**  
This notebook contains:
- Fine-tuning of the **mDeBERTa-v3-base-xnli** student model  
- Use of filtered high-confidence pseudo-labels  
- Integration of gold validation data  
- Final **Ensemble construction** combining DeBERTa + CatBoost  
- Optimal threshold selection and final results  

This notebook builds the final system achieving the best performance.

---
