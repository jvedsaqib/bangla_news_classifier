# Bangla News Article Classification

## Overview
This project focuses on the development of a Machine Learning (ML)-based system for classifying Bengali news articles into predefined categories. The classification system addresses the unique linguistic characteristics of the Bengali language and aims to improve information retrieval, content recommendation, and automated content analysis.

## Table of Contents
- [Introduction](#introduction)
- [Data Collection](#data-collection)
- [Data Preprocessing](#data-preprocessing)
- [Features](#features)
- [Classification](#classification)
- [Classifiers](#classifiers)
- [Performance Metrics](#performance-metrics)
- [Technologies Used](#technologies-used)
- [Future Scope](#future-scope)
- [Conclusion](#conclusion)
- [References](#references)

## Introduction
The rapid growth of digital media has led to an unprecedented increase in the availability of news articles. Efficiently classifying these articles into relevant categories is crucial for improving information retrieval and content analysis. This project categorizes Bengali news articles into categories such as National, Science, Education, International, Sports, Politics, Kolkata, and State.

## Data Collection
We collected Bengali news articles from three publicly available datasets:
1. **Shironaam Dataset (Hugging Face)**: Contains a collection of Bengali news headlines.
2. **Bengali News Articles Dataset (Kaggle)**: Offers a rich collection of articles from renowned newspapers.
3. **Potrika Bangla Newspaper Dataset (Kaggle)**: Includes in-depth reporting on diverse topics.

## Data Preprocessing
The collected data underwent several preprocessing steps:
- **Data Cleaning**: Removed duplicates and irrelevant entries.
- **Merging**: Combined data from all datasets into a unified dataset.
- **Balancing**: Addressed class imbalance by curating samples for under-represented categories.

## Features
### Traditional Features
- **Term Frequency (TF)**: Measures the frequency of a specific word in each article.
- **Inverse Document Frequency (IDF)**: Measures how common or rare a word is across the entire dataset.
- **TF-IDF**: A combined metric that represents the importance of a word within a specific article.

### Deep Learning-Based Features
- **Tokenizer**: Preprocesses articles into a format suitable for deep learning models.
- **Padding Sequences**: Standardizes sequences to a fixed length for model input.

## Classification
The dataset was split into training, validation, and testing sets:
- **50%** for training
- **25%** for validation
- **25%** for testing

### Classifiers Used
1. **Logistic Regression**
2. **Random Forest**
3. **Naïve Bayes**
4. **Support Vector Machine (SVM)**
5. **Long Short-Term Memory (LSTM)**

## Performance Metrics
The models were evaluated using metrics such as accuracy, precision, recall, and F1-score. The LSTM model achieved the highest accuracy of **91%**, demonstrating its effectiveness in classifying Bengali news articles.

## Technologies Used
- **Editors**: Jupyter Notebook, Kaggle Notebook Editor, Google Colab
- **Programming Language**: Python
- **Libraries**:
  - Data Manipulation: Pandas, Numpy
  - Natural Language Processing: NLTK, Sklearn
  - Visualization: Matplotlib, Seaborn, Wordcloud
  - Machine Learning: TensorFlow, Keras

## Future Scope
1. **Addressing Class Imbalance**: Implement techniques like SMOTE for better performance on under-represented categories.
2. **Integration of Advanced Deep Learning Models**: Use pre-trained models like BanglaBERT for improved accuracy.
3. **Real-Time Classification System**: Develop a system for instant categorization of news articles.
4. **Multi-label Classification Capability**: Allow articles to be tagged with multiple relevant categories.
5. **Incorporating Sentiment Analysis**: Add sentiment analysis to provide insights into the tone of news articles.

## Conclusion
This project successfully built a machine learning system to classify Bengali news articles into distinct categories. While strong performance was achieved for well-represented categories, challenges remain in accurately classifying under-represented categories. This work lays a strong foundation for future Bengali NLP tasks.

## References
1. [Dialect AI Shironaam Dataset](https://huggingface.co/datasets/dialect-ai/shironaam)
2. [Classification Bengali News Articles - IndicNLP](https://www.kaggle.com/datasets/csoham/classification-bengali-news-articles-indicnlp)
3. [Potrika Bangla Newspaper Dataset](https://www.kaggle.com/datasets/sabbirhossainujjal/potrika-bangla-newspaper-datasets)
4. [Bengali Stop Words List (GitHub)](https://github.com/stopwords-iso/stopwords-bn)
5. [Scikit-learn Documentation](https://scikit-learn.org/)
6. [TensorFlow/Keras Documentation](https://www.tensorflow.org/)
7. [NLTK Library](https://www.nltk.org/)

---

Feel free to contribute to this project by submitting issues or pull requests!
