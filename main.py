import streamlit as st
import os
from PIL import Image
import numpy as np
import pickle
import tensorflow
from tensorflow.keras.preprocessing import image
from tensorflow.keras.layers import GlobalMaxPooling2D
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
from annoy import AnnoyIndex
from numpy.linalg import norm

# Load precomputed embeddings and filenames
feature_list = np.array(pickle.load(open('embeddings.pkl', 'rb')))
filenames = pickle.load(open('filenames.pkl', 'rb'))

# Define the dimension of the feature vectors
feature_dim = feature_list.shape[1]  # Dimension of the feature vectors

# Initialize Annoy index
index = AnnoyIndex(feature_dim, 'euclidean')  # 'euclidean' for distance metric
for i, feature in enumerate(feature_list):
    index.add_item(i, feature)
index.build(10)  # Number of trees to build

# Initialize the model
model = ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
model.trainable = False
model = tensorflow.keras.Sequential([
    model,
    GlobalMaxPooling2D()
])

# Create upload directory if it doesn't exist
if not os.path.exists('uploads'):
    os.makedirs('uploads')

st.title('Products Recommendation System')

def save_uploaded_file(uploaded_file):
    try:
        file_path = os.path.join('uploads', uploaded_file.name)
        with open(file_path, 'wb') as f:
            f.write(uploaded_file.getbuffer())
        return file_path
    except Exception as e:
        st.error(f"Error occurred during file upload: {e}")
        return None

def feature_extraction(img_path, model):
    try:
        img = image.load_img(img_path, target_size=(224, 224))
        img_array = image.img_to_array(img)
        expanded_img_array = np.expand_dims(img_array, axis=0)
        preprocessed_img = preprocess_input(expanded_img_array)
        result = model.predict(preprocessed_img).flatten()
        normalized_result = result / norm(result)
        return normalized_result
    except Exception as e:
        st.error(f"Error occurred during feature extraction: {e}")
        return None

def recommend(features, index):
    indices = index.get_nns_by_vector(features, 6)  # Get 6 nearest neighbors
    return indices

def search_product(query, filenames):
    # Simple search based on file names
    search_results = [filename for filename in filenames if query.lower() in filename.lower()]
    return search_results

def display_recommendations(indices):
    num_recommendations = min(len(indices), 5)
    cols = st.columns(num_recommendations)
    for i, col in enumerate(cols):
        col.image(filenames[indices[i]])

# File upload and search option
option = st.radio("Choose Input Method", ("Upload Image", "Search Product"))

if option == "Upload Image":
    uploaded_file = st.file_uploader("Choose an image")
    if uploaded_file is not None:
        file_path = save_uploaded_file(uploaded_file)
        if file_path:
            display_image = Image.open(file_path)
            st.image(display_image)
            features = feature_extraction(file_path, model)
            if features is not None:
                indices = recommend(features, index)
                display_recommendations(indices)
        else:
            st.error("Failed to upload file.")

elif option == "Search Product":
    search_query = st.text_input("Search for a product:")
    if search_query:
        search_results = search_product(search_query, filenames)
        if search_results:
            st.success(f"Found {len(search_results)} matching products.")
            for result in search_results:
                st.image(result)
                features = feature_extraction(result, model)
                if features is not None:
                    indices = recommend(features, index)
                    display_recommendations(indices)
        else:
            st.warning("No products found matching the search query.")
