import streamlit as st
import pickle
import pandas as pd
import numpy as np
import re
from transformers import AutoModelForSequenceClassification
from transformers import AutoTokenizer, AutoConfig
import numpy as np
from tensorflow import keras
from scipy.special import softmax
from keras_preprocessing.sequence import pad_sequences
import time

#title
st.title('Web Aplication Perbandingan Hasil Analisa Sentimen Pada Tweets Twitter Menggunakan Model SVM, LSTM, CNN dan BERT ')
st.markdown("Pascal Adriantha - 2301922751")
st.markdown("""---""")


st.header("Sentiment Analysis 😃 😐 😡")
##svm
SVM_model_loaded = pickle.load(open('svm_model_saved.pkl', 'rb'))
 
#input
st.session_state.disabled = False
input = st.text_input('Input your text below👇', 
                      placeholder="Input text!",
                      max_chars=100)
if input:

    progress_text = "Analysis in progress. Please wait."
    my_bar = st.progress(0, text=progress_text)

    for percent_complete in range(100):
        time.sleep(0.08)
        my_bar.progress(percent_complete + 1, text=progress_text)
        
    my_bar.empty()

    st.success('Sentiment Analysis Generated!', icon="✅")

    def preprocess(text):
        new_text = []
        for t in text.split(" "):
            t = '@user' if t.startswith('@') and len(t) > 1 else t
            t = 'http' if t.startswith('http') else t
            new_text.append(t)
        return " ".join(new_text)

    input = preprocess(input)

    SVM_fe_loaded = pickle.load(open('svm_fe_saved.pkl', 'rb'))

    df_test20 = pd.DataFrame([input], columns=['Lemmatized and Stopword'])

    test_X_tfidf_8020 = SVM_fe_loaded.transform(df_test20['Lemmatized and Stopword'])

    ##prediction
    predictions_SVM_8020 = SVM_model_loaded.predict(test_X_tfidf_8020)

    predictions_SVM_8020 = (" ".join(predictions_SVM_8020))
    

    if predictions_SVM_8020 == "positive":
        st.subheader("SVM Model Result = :green[Positive] 😃")
    if predictions_SVM_8020 == "neutral":
        st.subheader("SVM Model Result = Neutral 😐")
    if predictions_SVM_8020 == "negative":
        st.subheader("SVM Model Result = :red[Negative] 😡")

    ##LSTM
    

    modelLSTM = keras.models.load_model('lstm_model.h5')

    input = preprocess(input)
    df_test20 = pd.DataFrame([input], columns=['Lemmatized and Stopword'])

    LSTM_fe_loaded = pickle.load(open('lstm_fe_saved.pkl', 'rb'))

    X_LSTM = LSTM_fe_loaded.texts_to_sequences(df_test20['Lemmatized and Stopword'].values)
    X_LSTM = pad_sequences(X_LSTM, maxlen = 100)

    Y_pred_LSTM = np.argmax(modelLSTM.predict(X_LSTM), axis=-1)

    if Y_pred_LSTM == 2:
        st.subheader("LSTM Model Result = :green[Positive] 😃")
    if Y_pred_LSTM == 1:
        st.subheader("LSTM Model Result = Neutral 😐")
    if Y_pred_LSTM == 0:
        st.subheader("LSTM Model Result = :red[Negative] 😡")
        
    ##CNN
    modelLSTM = keras.models.load_model('cnn_model.h5')

    input = preprocess(input)
    df_test20 = pd.DataFrame([input], columns=['Lemmatized and Stopword'])

    CNN_fe_loaded = pickle.load(open('cnn_fe_saved.pkl', 'rb'))

    X_CNN = CNN_fe_loaded.texts_to_sequences(df_test20)
    X_CNN = pad_sequences(X_CNN, maxlen = 100,
                  padding="post"
                  )

    Y_pred_CNN = np.argmax(modelLSTM.predict(X_CNN), axis=-1)

    if Y_pred_CNN == 2:
        st.subheader("CNN Model Result = :green[Positive] 😃")
    if Y_pred_CNN == 1:
        st.subheader("CNN Model Result = Neutral 😐")
    if Y_pred_CNN == 0:
        st.subheader("CNN Model Result = :red[Negative] 😡")   
       
    #BERT
    MODEL = f"cardiffnlp/twitter-roberta-base-sentiment-latest"
    tokenizer = AutoTokenizer.from_pretrained(MODEL)
    config = AutoConfig.from_pretrained(MODEL)

    modelBERT = AutoModelForSequenceClassification.from_pretrained(MODEL)

    text = input
    text = preprocess(text)
    encoded_input = tokenizer(text, return_tensors='pt')
    output = modelBERT(**encoded_input)
    scores = output[0][0].detach().numpy()
    scores = softmax(scores)

    ranking = np.argsort(scores)
    ranking = ranking[::-1]

    predicted_class_id = config.id2label[ranking[0]]
        
    if predicted_class_id == "positive":
        st.subheader("BERT Model Result = :green[Positive] 😃")
    if predicted_class_id == "neutral":
        st.subheader("BERT Model Result = Neutral 😐")
    if predicted_class_id == "negative":
        st.subheader("BERT Model Result = :red[Negative] 😡")

