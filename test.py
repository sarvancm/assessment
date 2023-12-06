import pandas as pd

df=pd.read_csv('articles.csv')

from sklearn import preprocessing 
label_encoder = preprocessing.LabelEncoder() 
df["Article_Type"] = label_encoder.fit_transform(df["Article_Type"])

df.drop(['Id','Article.Banner.Image'],axis=1,inplace=True)

df['Combined'] = pd.concat([df['Heading'], df['Full_Article']], axis=1).apply(lambda x: ''.join(x), axis=1)


x=df['Combined']
y=df['Article_Type']

from sentence_transformers import SentenceTransformer

model=SentenceTransformer('all-MiniLM-L6-v2')