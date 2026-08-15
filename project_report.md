# AI Agent for SMS Spam Detection and Intelligent Message Filtering
### Project Report

Generated on: 2026-08-15 10:25

## 1. Abstract
This project implements an AI-powered SMS spam detection and intelligent message filtering
system. It combines three complementary models — DistilBERT, BERT, and a Bidirectional LSTM —
to classify SMS messages as Spam or Ham, calculates a confidence-weighted risk score, detects
suspicious linguistic patterns, and exposes the entire pipeline through a professional Streamlit
security dashboard.

## 2. Introduction
SMS remains a common channel for phishing, financial fraud, and unsolicited promotional content.
Manually screening large volumes of messages is impractical, motivating an automated, model-driven
approach that flags risky messages and explains *why* they were flagged.

## 3. Problem Statement
Given the volume and evolving nature of spam and phishing SMS content, this project addresses the
need for an automated, explainable system capable of accurately separating spam from legitimate
messages while surfacing the specific signals that drove each decision.

## 4. Objectives
- Automatic Spam/Ham classification of SMS messages
- Comparison of DistilBERT, BERT, and BiLSTM architectures
- Confidence-weighted spam risk scoring (0-100)
- Intelligent message filtering (SAFE / SUSPICIOUS / SPAM)
- Rule-based suspicious indicator detection
- Bulk SMS analysis via CSV upload

## 5. Dataset
- Source: UCI / Kaggle SMS Spam Collection Dataset (`uciml/sms-spam-collection-dataset`)
- Total messages after cleaning: 5167
- Spam messages: 653
- Ham messages: 4514
- Spam percentage: 13.41%
- Ham percentage: 86.59%

## 6. Data Preprocessing
Messages were lowercased; URLs, email addresses, HTML tags, and special characters were stripped;
whitespace was normalized; missing and duplicate records were removed (403
duplicates removed, 0 missing/invalid records handled); labels were mapped to
Ham=0 / Spam=1.

## 7. Model Architecture

### DistilBERT
`distilbert-base-uncased` fine-tuned as a binary sequence classifier.

### BERT
`bert-base-uncased` fine-tuned as a binary sequence classifier.

### BiLSTM
```
Input -> Tokenizer -> Embedding -> Bidirectional LSTM -> Dropout -> Dense -> Sigmoid -> Spam/Ham
```

## 8. Model Training
- Max sequence length: 128 (BiLSTM: 100)
- Batch size: 16
- Transformer epochs: 2
- Device used: cuda

## 9. Evaluation Metrics
Accuracy, Precision, Recall, F1-score, and confusion matrices were computed for every model on a
held-out stratified test set.

## 10. Results

| Model | Accuracy | Precision | Recall | F1 |
|---|---:|---:|---:|---:|
| DistilBERT | 0.9942 | 0.9697 | 0.9846 | 0.9771 |
| BERT | 0.9961 | 0.9846 | 0.9846 | 0.9846 |
| BiLSTM | 0.9845 | 0.9672 | 0.9077 | 0.9365 |

**Best Model (by F1-score): BERT (98.46%)**

## 11. Spam Analysis
Top spam keywords: call, free, txt, mobile, text, stop, claim, reply, www, prize
Average spam message length: 137.89 characters
Average ham message length: 70.49 characters

## 12. Intelligent Filtering
Messages are categorized as SAFE, SUSPICIOUS, or SPAM based on the ensemble prediction, its
confidence, and the computed risk score, with configurable confidence thresholds.

## 13. Streamlit Application
`app.py` implements a multi-page dashboard (Home, Analyze SMS, Bulk SMS Analysis, Spam Dashboard,
Model Comparison, Dataset Explorer, About) that loads the saved models directly from
`SMS_Spam_AI_Project/models/` and runs independently of this notebook.

## 14. Conclusion
The ensemble of DistilBERT, BERT, and BiLSTM, combined with rule-based indicator detection, provides
an accurate and explainable spam-filtering system suitable for real-world SMS security triage.

## 15. Future Scope
- Multilingual SMS spam detection
- Real-time SMS monitoring
- Phishing URL analysis and reputation checks
- Explainable AI (attention/SHAP visualizations)
- Transformer ensemble/stacking models
- Voice-message spam detection
- WhatsApp/email spam detection
- Real-time mobile integration
- Continual learning from newly reported spam patterns
