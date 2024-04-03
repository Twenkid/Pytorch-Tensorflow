import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.corpus import stopwords
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import SelectKBest, chi2

KEYWORD_LIST = 'keywords.csv'
TEST_KEYWORD_LIST = 'test_keyword.cs'
RESULT_LIST = 'result_list.cs'

#df = pd.read_csv('[KEYWORD_LIST].csv')
df = pd.read_csv(KEYWORD_LIST)
data = pd.DataFrame(df)

words = stopwords.words('english_adjusted')

pipeline = Pipeline([('vect', TfidfVectorizer(ngram_range=(1, 3), stop_words=words)),
                    ('chi', SelectKBest(chi2, k='all')),
                    ('clf', LogisticRegression(C=1.0, penalty='l2', max_iter=1000))])

model = pipeline.fit(data.Keyword, data.Type)
chi = model.named_steps['chi']
clf = model.named_steps['clf']

doutput = pd.read_csv(TEST_KEYWORD_LIST)

doutput['Type'] = model.predict(doutput['Keyword'])

doutput.to_csv(RESULT_LIST)
##print('accuracy score ' + str(model.score(x_test, y_test)))
