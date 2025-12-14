# Fake News Classification Using Natural Language Processing

---

### 👥 Team Members

| Name | GitHub Handle | Contribution |
|------|---------------|--------------|
| Riddhima        | @github_handle | Data exploration, exploratory data analysis, dataset understanding |
| Joey            | @github_handle | Data preprocessing, text cleaning, feature preparation |
| Claire          | @github_handle | Model selection, training, hyperparameter tuning |
| Tanzeela        | @github_handle | Model evaluation, performance analysis, interpretation |
| Ayaan and Ishan | @github_handle | Project coordination, documentation, final presentation, application developement |

---

## 🎯 Project Highlights

- Built an end-to-end machine learning pipeline to classify news articles as fake or real using natural language processing.
- Fine-tuned a transformer-based language model (BERT) to capture semantic and contextual patterns in news text.
- Evaluated model performance using accuracy, precision, recall, and F1-score.
- Demonstrated how AI-driven text classification can support large-scale misinformation detection.
- Connected technical findings to real-world applications and stakeholder impact.

---

## 👩🏽‍💻 Setup and Installation

### Clone the repository
```bash
git clone https://github.com/your-username/fake-news-classification.git
cd fake-news-classification

---

## 🏗️ Project Overview

This project was developed as part of the Break Through Tech AI Program through an AI Studio Challenge. The goal of the challenge is to apply machine learning techniques to real-world problems while following an industry-style workflow.

The project focuses on misinformation detection using natural language processing. Working with an AI Studio host company, the team designed and implemented a reproducible NLP pipeline that includes data preprocessing, model development, evaluation, and interpretation of results.

Misinformation poses significant risks to public trust, political systems, and economic decision-making. By leveraging transformer-based language models, this project demonstrates how AI can assist in identifying misleading content and supporting fact-checking efforts at scale.

---

## 📊 Data Exploration

The dataset consists of labeled news articles categorized as fake or real. Each record contains article text and a corresponding label. Initial data exploration focused on understanding class balance, text length distributions, and stylistic differences between fake and real news articles.

Exploratory Data Analysis revealed noticeable differences in vocabulary usage, article length, and tone between the two classes. Fake news articles tended to use more sensational language, while real news articles showed more consistent formatting and neutral phrasing.

Challenges included noisy text data, inconsistent labeling formats, and varying article lengths. Assumptions were made to standardize labels and clean text while preserving semantic meaning.

---

## 🧠 Model Development

The modeling process began with baseline text classification methods to establish a reference for performance. These baselines helped validate the usefulness of more advanced approaches.

The final model fine-tunes a pretrained transformer-based architecture (BERT) for binary classification. This model was chosen for its ability to capture contextual relationships within text rather than relying solely on keyword frequency.

Hyperparameter tuning focused on learning rate, batch size, and maximum sequence length. The dataset was split into training and validation sets to evaluate generalization performance and reduce overfitting.

---

## 📈 Results & Key Findings

The transformer-based model achieved strong performance across all evaluation metrics, outperforming baseline approaches. Model performance was measured using accuracy, precision, recall, and F1-score.

Results showed improved recall for fake news detection, reducing the likelihood of misleading articles being classified as real. The model demonstrated strong generalization to unseen data, indicating effective learning of semantic patterns.

These findings highlight the effectiveness of transformer-based NLP models for misinformation detection tasks.

---

## 🚀 Next Steps

- Incorporate explainability tools such as SHAP or attention visualization to improve model transparency.
- Expand the dataset to improve robustness and reduce potential bias.
- Evaluate model fairness across different news topics and sources.
- Deploy the trained model as an API or web-based application for real-time classification.

---

## 📝 License

This project is licensed under the MIT License.

---

## 📄 References

- Devlin et al., “BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding”
- Hugging Face Transformers Documentation
- Scikit-learn Documentation

## 🙏 **Acknowledgements** (Optional but encouraged)

Thank you Jenna Hunte, Saggar Thaker, Accenture, and Break Through Tech
