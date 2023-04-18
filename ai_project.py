import os 
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

import sys
!{sys.executable} -m pip install tensorflow-addons
import tensorflow_addons as tfa

#general purpose packages
import os
import numpy as np
import pandas as pd
import tensorflow as tf

from sklearn import preprocessing
from imblearn.over_sampling import RandomOverSampler
from sklearn.model_selection import train_test_split

#transformers
from transformers import BertTokenizerFast
from transformers import TFBertModel
from transformers import RobertaTokenizerFast
from transformers import TFRobertaModel

#keras
import tensorflow as tf
from tensorflow import keras

#set seed for reproducibility
seed=42

df = pd.read_csv('/kaggle/input/prometeo23-kaggle/train_absa.csv')
df_test = pd.read_csv('/kaggle/input/prometeo23-kaggle/test_absa.csv')

tokenizer_roberta = RobertaTokenizerFast.from_pretrained("roberta-base")

X=df['text'].values
y=df['label'].values

x=df_test['text'].values

X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.2, stratify=y, random_state=seed)

y_train_le = y_train.copy()
y_valid_le = y_valid.copy()

ohe = preprocessing.OneHotEncoder()
y_train = ohe.fit_transform(np.array(y_train).reshape(-1, 1)).toarray()
y_valid = ohe.fit_transform(np.array(y_valid).reshape(-1, 1)).toarray()

token_lens = []
for txt in X_train:
    tokens = tokenizer_roberta.encode(txt, max_length=333, truncation=True)
    token_lens.append(len(tokens))
max_length=np.max(token_lens)

MAX_LEN=333

MAX_LEN

def tokenize_roberta(data,max_len=MAX_LEN) :
    input_ids = []
    attention_masks = []
    for i in range(len(data)):
        encoded = tokenizer_roberta.encode_plus(
            data[i],
            add_special_tokens=True,
            max_length=max_len,
            padding='max_length',
            return_attention_mask=True
        )
        input_ids.append(encoded['input_ids'])
        attention_masks.append(encoded['attention_mask'])
    return np.array(input_ids),np.array(attention_masks)

train_input_ids, train_attention_masks = tokenize_roberta(X_train, MAX_LEN)
val_input_ids, val_attention_masks = tokenize_roberta(X_valid, MAX_LEN)
test_input_ids, test_attention_masks = tokenize_roberta(df_test['text'].values, MAX_LEN)

test_input_ids, test_attention_masks = tokenize_roberta(x, MAX_LEN)

metric1=tfa.metrics.F1Score(num_classes=3,threshold=0.5)
metric2=tfa.metrics.FBetaScore(num_classes=3,threshold=0.5,beta=2.0)

def create_model(bert_model, max_len=MAX_LEN):
    
    opt = tf.keras.optimizers.legacy.Adam(learning_rate=2e-5, decay=0.07)
    loss = tf.keras.losses.CategoricalCrossentropy()
    accuracy = tf.keras.metrics.CategoricalAccuracy()

    input_ids = tf.keras.Input(shape=(max_len,),dtype='int32')
    attention_masks = tf.keras.Input(shape=(max_len,),dtype='int32')
    output = bert_model([input_ids,attention_masks])
    output = output[1]
    output = tf.keras.layers.Dense(3, activation=tf.nn.softmax)(output)
    model = tf.keras.models.Model(inputs = [input_ids,attention_masks],outputs = output)
    model.compile(opt, loss=loss, metrics = [metric1,metric2])
    return model

roberta_model = TFRobertaModel.from_pretrained('roberta-base')

model = create_model(roberta_model, MAX_LEN)
print(model.summary())

history_2 = model.fit([train_input_ids,train_attention_masks], y_train, validation_data=([val_input_ids,val_attention_masks], y_valid), epochs=3, batch_size=4)

model.save('roberta.h5')

results=list()
sentiment = model.predict([test_input_ids,test_attention_masks],batch_size=1,verbose = 1)

sentiment[0]

for i in range(0,len(sentiment)):
    a=sentiment[i]
    results.append(np.argmax(a))

results[0]

sub = pd.read_csv('/kaggle/input/prometeo23-kaggle/sample.csv')

sub_dict = {0:'Negative' , 1:'Neutral' , 2:'Positive'}
results = sub['Predicted'].map(sub_dict)

sub['Predicted'] = results

sub

sub.to_csv('submission.csv',index=False)
