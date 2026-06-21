from goldman_emma_raw import goldman_docs
from henson_matthew_raw import henson_docs
from wu_tingfang_raw import wu_docs
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB

# Setting up the combined list of friends' writing samples
friends_docs = goldman_docs + henson_docs + wu_docs

# Setting up labels for your three friends
friends_labels = [1] * 154 + [2] * 141 + [3] * 166

# Task 5: Print out a document from each friend
print("Emma Goldman document (index 70):")
print(goldman_docs[70])
print("\nMatthew Henson document (index 70):")
print(henson_docs[70])
print("\nTingFang Wu document (index 70):")
print(wu_docs[70])

mystery_postcard = """
My friend,
From the 10th of July to the 13th, a fierce storm raged, clouds of
freeing spray broke over the ship, incasing her in a coat of icy mail,
and the tempest forced all of the ice out of the lower end of the
channel and beyond as far as the eye could see, but the _Roosevelt_
still remained surrounded by ice.
Hope to see you soon.
"""

# Task 2: Create bow_vectorizer
bow_vectorizer = CountVectorizer()

# Task 3: Define friends_vectors (fit + transform on all friends' docs)
friends_vectors = bow_vectorizer.fit_transform(friends_docs)

# Task 4: Define mystery_vector (transform only, using existing vocabulary)
mystery_vector = bow_vectorizer.transform([mystery_postcard])

# Task 6: Define friends_classifier
friends_classifier = MultinomialNB()

# Task 7: Train the classifier
friends_classifier.fit(friends_vectors, friends_labels)

# Task 8: Get predictions
predictions = friends_classifier.predict(mystery_vector)

# Map label to friend name
label_to_name = {1: "Emma Goldman", 2: "Matthew Henson", 3: "TingFang Wu"}
mystery_friend_name = label_to_name.get(predictions[0], "someone else")

# Task 9: Print result
print("The postcard was from {}!".format(mystery_friend_name))

# Task 10: Show probabilities using predict_proba()
probabilities = friends_classifier.predict_proba(mystery_vector)
print("\nEstimated probabilities for each friend:")
for i, prob in zip(friends_classifier.classes_, probabilities[0]):
    name = label_to_name.get(i, "Unknown")
    print(f"{name}: {prob:.2%}")
