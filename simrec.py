import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

g1 = pd.read_csv('BX-Books.csv', encoding='latin-1', sep=';', low_memory=False, on_bad_lines='skip') # reads in book csv and skips erorrs
#
g = g1.drop_duplicates(subset="Book-Title")
 # drop duplicates of book titels 
g2 =g.drop(["ISBN","Image-URL-L","ISBN","Image-URL-M","ISBN","Image-URL-S"],axis=1)
#drop the rest of te uneeded columns

df = g2.sample(n=200 , replace=False,random_state=490) # take a sample of 200 with no replacement 490 seed 
df =df.reset_index() # reset index 
df=df.drop("index",axis=1) # get rid of old index axis-1 refers to column

def cleantext(te):
    ''' a simple function to clean up text by gettong rid of stop words and whitespaces
    will be used to get data ready '''
    te = str(te).lower()
    stopwords = ['a', 'the', 'and', 'i', ' ']
    for i in stopwords:
        te = te.replace(i, '')  # Assign the result back to 'te'
    return te

df['Book-Author'] = df['Book-Author'].apply(cleantext)
df['Publisher'] = df['Publisher'].apply(cleantext)

df['Book-Title'] = df['Book-Title'].str.lower()


df['Data'] = df['Book-Author'] + ' ' + df['Book-Title'] + ' ' + df['Publisher']

#create a new column called data that contains the author title and publisher

vectorizer = CountVectorizer()
vectorized = vectorizer.fit_transform(df['Data'])
#we vecotrize this new data column
similarities = cosine_similarity(vectorized)
df = pd.DataFrame(similarities, columns=df['Book-Title'], index=df['Book-Title'])
#creates a dtad frame form the similarity matrix using book title as both column and index
input_book = 'Thank You, Jeeves'.lower() # the book we use as example to check 
if input_book in df.columns:
    recommendations = pd.DataFrame(df[input_book].nlargest(11).index)
    #will give 10 reccomendations
    recommendations = recommendations[recommendations['Book-Title'] != input_book]
    #doesnt include the book itself

    print(recommendations)
else:
    print("Input book not found in the DataFrame.")
