import pandas as pd

df=pd.read_csv('Restaurant_Reviews.tsv',delimiter='\t',quoting=3)

#cleaning the data
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
nltk.download('stopwords')
corpus=[]
for i in range(0,1000):
    reviews=re.sub('[^a-zA-Z]',' ',df['Review'][i])
    reviews=reviews.lower()
    reviews=reviews.split()
    ps=PorterStemmer()
    reviews=[ps.stem(word) for word in reviews if not word in set(stopwords.words('english'))]
    reviews=' '.join(reviews)
    corpus.append(reviews)
#Creating thr bag of words model
from sklearn.feature_extraction.text import CountVectorizer
cv=CountVectorizer(max_features=1500)
X=cv.fit_transform(corpus).toarray()
y=df.iloc[:,1].values
#building the classification model
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(X,y,test_size=0.2)

from sklearn.naive_bayes import GaussianNB
gn=GaussianNB()
gn.fit(x_train,y_train)

y_pred=gn.predict(x_test)

from sklearn.metrics import confusion_matrix
cm=confusion_matrix(y_test,y_pred)

