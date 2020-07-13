from nltk.tokenize import word_tokenize
import gensim
from gensim.models import Word2Vec
import pandas as pd

data = pd.read_csv('Derlem.csv')
data.head()

text = []
for i in range(len(data)):
    text.append(data.iloc[i, 3])


words = []
for i in range(len(text)):
    words.append(word_tokenize(str(text[i])))


# CBOW modeli oluştur
model1 = gensim.models.Word2Vec(words, min_count=1, size = 100, window = 5)

# sonuclar
print("'yaşlı' " + "ve 'vatandaş' arasındaki benzerlik - CBOW : ",
      model1.similarity('yaşlı', 'vatandaş'))

print("'bebek' " + "ve 'kritik' arasındaki benzerlik - CBOW : ",
      model1.similarity('bebek', 'kritik'))

print("'kuş' " + "ve 'mahalle' arasındaki benzerlik - CBOW : ",
      model1.similarity('kuşlara', 'mahalle'))

print("'ev' " + "ve 'karantina' arasındaki benzerlik - CBOW : ",
      model1.similarity('ev', 'karantina'))

print("'borsa' " +"ve 'kayıp' arasındaki benzerlik - CBOW : ",
      model1.similarity('Borsa', 'kayıplar'))


# Skip Gram modelini oluştur
model2 = gensim.models.Word2Vec(words, min_count=1, size=100, window=5, sg=1)

# sonuclar
print("'yaşlı' " + "ve 'vatandaş' arasındaki benzerlik - Skip Gram : ",
      model2.similarity('yaşlı', 'vatandaş'))

print("'bebek' " + "ve 'kritik' arasındaki benzerlik - Skip Gram : ",
      model2.similarity('bebek', 'kritik'))

print("'kuş' " + "ve 'mahalle' arasındaki benzerlik - Skip Gram : ",
      model2.similarity('kuşlara', 'mahalle'))

print("'ev' " + "ve 'karantina' arasındaki benzerlik - Skip Gram : ",
      model2.similarity('ev', 'karantina'))

print("'borsa' " + "ve 'kayıp' arasındaki benzerlik - Skip Gram : ",
      model2.similarity('Borsa', 'kayıplar'))