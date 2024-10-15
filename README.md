<<<<<<< HEAD
# NLP-Sentiment-Analysis-


# Sentiment Analysis with Natural Language Processing (NLP)

This project demonstrates how to conduct **Sentiment Analysis** using various NLP techniques, including VADER and RoBERTa, to analyze sentiments in Amazon reviews. By employing these models, we can assess the positivity or negativity of reviews and gain insights into customer opinions. 

## Project Overview

- **Objective**: Analyze sentiments from Amazon reviews and compare results from different sentiment analysis models.
- **Models Used**: 
  - VADER (Valence Aware Dictionary and sEntiment Reasoner)
  - RoBERTa (a robustly optimized BERT pre-training approach)
- **Dataset**: Amazon reviews dataset containing text and star ratings.
- **Libraries Used**: Pandas, NLTK, Transformers, Matplotlib, Seaborn, Scipy, TQDM.

## Key Steps

1. **Install Dependencies**:
   Install the required libraries:
   ```bash
   %pip install pandas nltk transformers matplotlib seaborn tqdm
Load the Data: We load the Amazon reviews dataset and explore its structure:

python
Always show details

Copy code
import pandas as pd

df = pd.read_csv('amazon_reviews.csv')
print(df.head())
Brief Data Visualization: Visualize the distribution of review scores:

python
Always show details

Copy code
ax = df['Score'].value_counts().sort_index() \\
    .plot(kind='bar',
          title='Count of Reviews by Stars',
          figsize=(10, 5))
ax.set_xlabel('Review Stars')
plt.show()
Basic NLTK: Tokenize the text and perform part-of-speech tagging:

python
Always show details

Copy code
import nltk

example = df['Text'][50]
print(example)
tokens = nltk.word_tokenize(example)
tagged = nltk.pos_tag(tokens)
entities = nltk.chunk.ne_chunk(tagged)
entities.pprint()
VADER Sentiment Scoring: Using NLTK's SentimentIntensityAnalyzer to obtain sentiment scores:

python
Always show details

Copy code
from nltk.sentiment import SentimentIntensityAnalyzer
from tqdm.notebook import tqdm

sia = SentimentIntensityAnalyzer()
res = {}
for i, row in tqdm(df.iterrows(), total=len(df)):
    text = row['Text']
    myid = row['Id']
    res[myid] = sia.polarity_scores(text)

vaders = pd.DataFrame(res).T
vaders = vaders.reset_index().rename(columns={'index': 'Id'})
vaders = vaders.merge(df, how='left')
vaders.head()
Plot VADER Results: Visualize the sentiment scoring results:

python
Always show details

Copy code
import seaborn as sns
import matplotlib.pyplot as plt

ax = sns.barplot(data=vaders, x='Score', y='compound')
ax.set_title('Compound Score by Amazon Star Review')
plt.show()

fig, axs = plt.subplots(1, 3, figsize=(12, 3))
sns.barplot(data=vaders, x='Score', y='pos', ax=axs[0])
sns.barplot(data=vaders, x='Score', y='neu', ax=axs[1])
sns.barplot(data=vaders, x='Score', y='neg', ax=axs[2])
axs[0].set_title('Positive')
axs[1].set_title('Neutral')
axs[2].set_title('Negative')
plt.tight_layout()
plt.show()
Roberta Pre-trained Model: Use RoBERTa for advanced sentiment analysis:

python
Always show details

Copy code
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from scipy.special import softmax

MODEL = "cardiffnlp/twitter-roberta-base-sentiment-latest"
tokenizer = AutoTokenizer.from_pretrained(MODEL)
model = AutoModelForSequenceClassification.from_pretrained(MODEL)

encoded_text = tokenizer(example, return_tensors='pt')
output = model(**encoded_text)
scores = output[0][0].detach().numpy()
scores = softmax(scores)
print({"roberta_neg": scores[0], "roberta_neu": scores[1], "roberta_pos": scores[2]})
Combine and Compare: Compare the results of VADER and RoBERTa:

python
Always show details

Copy code
sns.pairplot(data=results_df,
             vars=['vader_neg', 'vader_neu', 'vader_pos',
                   'roberta_neg', 'roberta_neu', 'roberta_pos'],
             hue='Score',
             palette='tab10')
plt.show()
Review Examples: Positive 1-Star and Negative 5-Star Reviews: Let's look at some examples where the model scoring and review score differ the most:

python
Always show details

Copy code
positive_1_star_roberta = results_df.query('Score == 1') \\
    .sort_values('roberta_pos', ascending=False)['Text'].values[0]

negative_5_star_roberta = results_df.query('Score == 5') \\
    .sort_values('roberta_neg', ascending=False)['Text'].values[0]

print("Positive 1-Star Review (RoBERTa):", positive_1_star_roberta)
print("Negative 5-Star Review (RoBERTa):", negative_5_star_roberta)
Dataset
The dataset contains Amazon reviews with various ratings, providing a rich source for sentiment analysis. The reviews are assessed based on their star ratings and text content, enabling us to explore customer sentiments.
=======
# NLP-Sentiment-Analysis-
>>>>>>> parent of 78d7661 (f1)
