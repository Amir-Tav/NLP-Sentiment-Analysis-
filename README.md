
# Sentiment Analysis Using NLP 📊

Welcome to the world of **Natural Language Processing (NLP)**! In this project, we'll explore sentiment analysis from customer reviews using some powerful NLP techniques. Buckle up as we dive into the code, data, and some fascinating insights!

## Table of Contents
1. [Overview](#overview)
2. [Getting Started](#getting-started)
3. [Data Preprocessing](#data-preprocessing)
4. [Sentiment Analysis](#sentiment-analysis)
5. [Results](#results)
6. [Conclusion](#conclusion)

---

## Overview
This project aims to classify customer sentiments based on Amazon product reviews. We use **NLP** tools to preprocess the text data, analyze it, and eventually predict whether reviews are positive or negative.

### Libraries Used
```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import nltk
```

---

## Getting Started

### Dataset
The dataset we are working with is the **Amazon Fine Food Reviews** dataset. You can find it [here](https://www.kaggle.com/datasets/snap/amazon-fine-food-reviews).

First, we load the dataset and take a subset of 500 reviews to keep things manageable. 

```python
df = pd.read_csv('data/Reviews.csv')  # Reading the reviews data
df = df.head(500)  # Taking a subset of 500 reviews
print(df.shape)  # Prints: (500, 10)
```

---

## Data Preprocessing

Before diving into analysis, we need to clean and preprocess the data. This includes tokenizing the text, removing stop words, and other common NLP tasks.

### Tokenizing the Text
We use `nltk` to tokenize the words and prepare them for analysis.

```python
from nltk.tokenize import word_tokenize

df['tokenized'] = df['Text'].apply(lambda x: word_tokenize(x.lower()))
```

### Removing Stop Words
Stop words (common words like "the", "is", "and") don't contribute much meaning and can be removed.

```python
from nltk.corpus import stopwords

stop_words = set(stopwords.words('english'))
df['filtered_tokens'] = df['tokenized'].apply(lambda x: [word for word in x if word not in stop_words])
```

---

## Sentiment Analysis

Now for the exciting part! We analyze the sentiment of reviews by looking at their textual data.

### Word Cloud Visualization

A quick look at the most frequent words in positive and negative reviews:

```python
from wordcloud import WordCloud

# Generate word clouds
positive_reviews = " ".join(df[df['Score'] > 3]['Text'])
wordcloud = WordCloud(width=800, height=400).generate(positive_reviews)

# Display the word cloud
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis("off")
plt.show()
```

### Sentiment Classification
To classify sentiment, we can use basic techniques such as checking for positive or negative keywords.

```python
# Sample code to classify based on score (positive/negative sentiment)
df['sentiment'] = df['Score'].apply(lambda x: 'positive' if x > 3 else 'negative')
```

---

## Results

After analyzing the data, we found some interesting insights. For example, the majority of reviews in the dataset are positive, which is common for product reviews.

### Data Visualization

We also took a look at the distribution of review scores:

```python
sns.countplot(x='Score', data=df)
plt.title('Distribution of Review Scores')
plt.show()
```

---

## Conclusion

This project highlights the basics of sentiment analysis using NLP techniques. We used a simple dataset and some basic text-processing techniques to analyze and classify sentiment. While this is just scratching the surface of NLP, it demonstrates how powerful these techniques can be for understanding large-scale textual data.
