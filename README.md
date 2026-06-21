Mystery Friend NLP Classifier
Overview
This project uses a bag-of-words model and a Naive Bayes classifier to identify which of three friends wrote a mystery postcard. It compares writing samples from Emma Goldman, Matthew Henson, and TingFang Wu.

What the Project Does
Loads writing samples from three authors.

Converts the text into feature vectors with CountVectorizer.

Trains a MultinomialNB classifier on the labeled samples.

Predicts which friend likely wrote the mystery postcard.

Displays prediction probabilities with predict_proba().

Setup
To run this project, make sure the provided data modules are available:

goldman_emma_raw

henson_matthew_raw

wu_tingfang_raw

You also need scikit-learn installed in your Python environment.

How It Works
Combine all friend documents into one list.

Vectorize the text using a bag-of-words approach.

Train a Naive Bayes classifier on the vectors and labels.

Transform the mystery postcard with the same vectorizer.

Predict the most likely author.

Files
script.py: Main Python script for vectorizing text, training the model, and making predictions.

README.md: Project overview and usage notes.

Results
The mystery postcard is likely written by Matthew Henson because it mentions the Roosevelt and Arctic ice, which matches expedition-related writing.

Libraries Used
sklearn.feature_extraction.text.CountVectorizer

sklearn.naive_bayes.MultinomialNB

Notes
You can test the classifier further by replacing mystery_postcard with excerpts from Project Gutenberg or with your own sample texts.
