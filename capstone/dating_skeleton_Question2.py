import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from matplotlib import pyplot as plt
from sklearn.linear_model import LinearRegression


#1.) Import Data to Dataframe:
df = pd.read_csv("profiles.csv")

essay_cols = ["essay0", "essay1", "essay2", "essay3", "essay4", "essay5", "essay6", "essay7", "essay8", "essay9"]

# Format Text Data, remove na, lowercase, remove non-text values.
all_essays = df[essay_cols].fillna('')
all_essays = all_essays[essay_cols].apply(lambda x: ' '.join(x), axis=1)
all_essays = all_essays.replace(regex=r'[\W]', value=' ').replace(regex=r'[\s_]+', value=' ')

list_young = ['party', 'parting', 'school', 'beach', 'shit', 'fuck', 'internet', 'weekend', 'workout', 'youtube', 'exams', 'hungover', 'lol']
list_middle = ['bills', 'work', 'coworkers', 'drinks', 'friends', 'office', 'company', 'apartment', 'money', 'relax', 'wedding', 'manager', 'boss']
list_old = ['yard', 'children', 'child', 'family', 'blessed', 'loving', 'country', 'prayers', 'thankful', 'home', 'house', 'employee', 'employees']

vectorizer = CountVectorizer()
vectorizer.fit(list_young)
word_array = vectorizer.transform(all_essays).toarray()
cnts_young = pd.DataFrame(word_array, columns=vectorizer.get_feature_names()).sum(axis=1)

vectorizer.fit(list_middle)
word_array = vectorizer.transform(all_essays).toarray()
cnts_middle = pd.DataFrame(word_array, columns=vectorizer.get_feature_names()).sum(axis=1)

vectorizer.fit(list_old)
word_array = vectorizer.transform(all_essays).toarray()
cnts_old = pd.DataFrame(word_array, columns=vectorizer.get_feature_names()).sum(axis=1)
#print(word_cnts)

religious_level = df['religion'].replace(r'^(\w+)', '', regex=True).replace(r'^(\s)', '', regex=True)
data = pd.DataFrame()
data['young_words'] = cnts_young
data['middle_words'] = cnts_middle
data['old_words'] = cnts_old
data['age'] = df['age']
data = data.dropna()

#Data Exploration Plots
plt.subplot(221)
plt.scatter(data['age'], data['young_words'])
plt.xlabel('Age')
plt.ylabel('Word Cnt (Young)')

plt.subplot(222)
plt.scatter(data['age'], data['middle_words'])
plt.xlabel('Age')
plt.ylabel('Word Cnt (Middle)')

plt.subplot(223)
plt.scatter(data['age'], data['old_words'])
plt.xlabel('Age')
plt.ylabel('Word Cnt (Old)')
plt.show()

#Linear Regression
X = data[['young_words', 'middle_words', 'old_words']]
y = data['age']
print(X)
print(y)

lm = LinearRegression()
model = lm.fit(X, y)
print(lm.score(X, y))
