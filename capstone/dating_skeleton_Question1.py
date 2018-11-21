import pandas as pd
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import MultinomialNB

#1.) Import Data to Dataframe:
df = pd.read_csv("profiles.csv")

'''Extract Ethinicty/Religion Name from first word of the string, create map based on religion_mapping and insert this 
	back into the dataframe as a new column.
*Note ethnicity can be a list, we are only using the first listed ethnicity assuming users listed them in order of 
	importance to them.
'''

religions = df['religion'].str.extract(r'^(\w+)')
ethnicity = df['ethnicity'].str.extract(r'([\w ]+)')

# Mappings
religion_mapping = {'atheism': 0, 'agnosticism': 1, 'christianity': 2, 'catholicism': 3, 'judaism': 4, 'buddhism': 5,
					'hinduism': 6, 'islam': 7, 'other': 8}

ethnicity_mapping = {'asian': 0, 'black': 1, 'hispanic ': 2, 'indian': 3, 'middle eastern': 4, 'native american': 5,
					'pacific islander': 6, 'white': 7, 'other': 8}

drug_mapping = {"never": 0, "sometimes": 1, "often": 2}

alcohol_mapping = {'not at all': 0, 'rarely': 1, 'socially': 2, 'often': 3, 'very often': 4, 'desperately': 5}

# Pull used data/mappings into a centralized dataframe.
data = pd.DataFrame()
data['religion_map'] = religions[0].map(religion_mapping)
data['ethnicity_map'] = ethnicity[0].map(ethnicity_mapping)
data['drug_map'] = df.drugs.map(drug_mapping)
data['alcohol_map'] = df.drinks.map(alcohol_mapping)
data = data.dropna()


# Look at Freqency of each religion based on Ethnicity
relg_freq = data.groupby(['ethnicity_map','religion_map']).size().reset_index(name='relg_frequency')
drug_freq = data.groupby(['drug_map','religion_map']).size().reset_index(name='relg_frequency')
alc_freq = data.groupby(['alcohol_map','religion_map']).size().reset_index(name='relg_frequency')
drug_counts = data.drug_map.value_counts()
relg_counts = data.ethnicity_map.value_counts()
alc_counts = data.alcohol_map.value_counts()

# Rather than frequency look at the percentage of religion occurrence in each subset.
p_list = []
for i in range(len(relg_freq)):
	percentage = round(100 * (relg_freq['relg_frequency'][i] / relg_counts[relg_freq['ethnicity_map'][i]]), 2)
	p_list.append(percentage)
relg_freq['relg_percentage'] = p_list

p_list = []
for i in range(len(drug_freq)):
	percentage = round(100 * (drug_freq['relg_frequency'][i] / drug_counts[drug_freq['drug_map'][i]]), 2)
	p_list.append(percentage)
drug_freq['relg_percentage'] = p_list

p_list = []
for i in range(len(alc_freq)):
	percentage = round(100 * (alc_freq['relg_frequency'][i] / alc_counts[alc_freq['alcohol_map'][i]]), 2)
	p_list.append(percentage)
alc_freq['relg_percentage'] = p_list


# Plot graphs showing the religion vs mapped field where size of the data point is based on percentage of occurrence.
plt.subplot(221)
plt.scatter(relg_freq['ethnicity_map'], relg_freq['religion_map'], s=relg_freq['relg_percentage']**2)
plt.xlabel("Ethnicity")
plt.ylabel("Religion")

plt.subplot(222)
plt.scatter(drug_freq['drug_map'], drug_freq['religion_map'], s=drug_freq['relg_percentage']**2)
plt.xlabel("drug_usage")
plt.ylabel("Religion")

plt.subplot(223)
plt.scatter(alc_freq['alcohol_map'], alc_freq['religion_map'], s=alc_freq['relg_percentage']**2)
plt.xlabel("alcohol_usage")
plt.ylabel("Religion")


X_train, X_test, y_train, y_test = train_test_split(data[data.columns[-3:]], data['religion_map'], random_state=42)

#KNN Classification
accuracies = []
for k in range(1, 50):
	Classifier = KNeighborsClassifier(n_neighbors=k)
	Classifier.fit(X_train, y_train)
	accuracies.append(Classifier.score(X_test, y_test))

plt.subplot(224)
plt.plot(range(1, 50), accuracies)
plt.xlabel('K')
plt.ylabel('Validation Accuracy')
plt.title('Religion Classifier Accuracy')
plt.show()

#Naive Bayes Classification
Classifier = MultinomialNB()
Classifier.fit(X_train,y_train)
score = Classifier.score(X_test, y_test)
print(score)