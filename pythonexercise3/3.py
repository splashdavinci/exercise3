import nltk
from nltk.corpus import gutenberg
import matplotlib.pyplot as plt

nltk.download('gutenberg')
# Reading the Beluga whale archive from the Gutenberg dataset
whale_text = gutenberg.raw('melville-moby_dick.txt')

nltk.download('punkt')
tokens = nltk.word_tokenize(whale_text)

nltk.download('stopwords')
stop_words = set(nltk.corpus.stopwords.words('english'))
filtered_tokens = [token for token in tokens if token.lower() not in stop_words]

nltk.download('averaged_perceptron_tagger')
pos_tags = nltk.pos_tag(filtered_tokens)

pos_freq = nltk.FreqDist(tag for (word, tag) in pos_tags)
top_pos = pos_freq.most_common(5)

print("POSfrequency：")
for pos, freq in top_pos:
    print(f"{pos}: {freq}")
    
nltk.download('wordnet')
lemmatizer = nltk.WordNetLemmatizer()
lemmatized_tokens = [lemmatizer.lemmatize(token) for token in filtered_tokens]


lemmatized_freq = nltk.FreqDist(lemmatized_tokens)
lemmatized_freq.plot(20, cumulative=False)

plt.xlabel('root')
plt.ylabel('frequency')
plt.title('Root frequency distribution')
plt.show()
