import streamlit as st
import tensorflow as tf
import pandas as pd
import numpy as np

#Reading the model
model = tf.keras.models.load_model('toxicity.h5')



data = pd.read_csv("./data/train.csv")

data.head()
X = data['comment_text']
y = data.iloc[:,2:].values
MAX_FEATURES = 200000 # Number of words in the vocabulary
vectorizer = tf.keras.layers.TextVectorization(
    max_tokens=MAX_FEATURES, output_sequence_length=1800, output_mode="int"
)
vectorizer.adapt(X.values)



def main():
    st.title("Toxicity Detection")



    st.subheader("Enter the text to check for toxicity")
    text = st.text_input("Enter the text here")
    vtext = vectorizer([text])
    if st.button("Predict"):
        prediction = model.predict(vtext)
        if prediction[0][0] > 0.5:
            st.write("Toxic")
        else:
            st.write("Not Toxic") 

# def score_comment(comment):
#     vectorized_comment = vectorizer([comment])
#     results = model.predict(vectorized_comment)

#     text=''
#     for idx, col in enumerate(data.columns[2:-1]):
#         text += f'{col}: {results[0][idx]>0.5}|| '
#     return text

if __name__ == '__main__':
    main()

# print(score_comment("You are a bad person"))

