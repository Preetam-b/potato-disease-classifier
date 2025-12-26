import streamlit as st
from PIL import Image
from api import predict_image

st.set_page_config(page_title="Potato Disease Classifier", layout="centered")

st.title("🥔 Potato Disease Classification")
st.write("Upload a potato leaf image. The model predicts disease with confidence.")

uploaded_file = st.file_uploader("Choose an image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    if st.button("Predict"):
        result = predict_image(image)

        st.subheader("Prediction")
        st.success(result["prediction"])

        st.write(f"**Confidence:** {result['confidence']}%")

        st.subheader("Class Probabilities")
        st.json(result["probabilities"])

        if result["prediction"] == "Uncertain":
            st.warning("⚠️ Model is unsure. Image may be outside training distribution.")
