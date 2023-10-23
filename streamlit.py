import streamlit as st
import tensorflow as tf

#Reading the model
model = tf.keras.models.load_model('toxicity.h5')

def main():
    st.title("Toxicity Detection")
    st.subheader("Enter the text to check for toxicity")
    text = st.text_input("Enter the text here")
    if st.button("Predict"):
        prediction = model.predict([text])
        if prediction[0][0] > 0.5:
            st.write("Toxic")
        else:
            st.write("Not Toxic") 

if __name__ == '__main__':
    main()