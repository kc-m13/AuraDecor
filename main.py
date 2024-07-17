# import streamlit as st
# import os
# from PIL import Image
# import numpy as np
# import pickle
# import tensorflow
# from tensorflow.keras.preprocessing import image
# from tensorflow.keras.layers import GlobalMaxPooling2D
# from tensorflow.keras.applications.resnet50 import ResNet50,preprocess_input
# from sklearn.neighbors import NearestNeighbors
# from numpy.linalg import norm

# # # app.py

# # import streamlit as st

# # st.title('Hello, Streamlit!')
# # st.write('This is a test Streamlit app.')


# feature_list = np.array(pickle.load(open('embeddings.pkl','rb')))
# filenames = pickle.load(open('filenames.pkl','rb'))

# model = ResNet50(weights='imagenet',include_top=False,input_shape=(224,224,3))
# model.trainable = False

# model = tensorflow.keras.Sequential([
#     model,
#     GlobalMaxPooling2D()
# ])

# st.title('Fashion Recommender System')

# def save_uploaded_file(uploaded_file):
#     try:
#         with open(os.path.join('uploads',uploaded_file.name),'wb') as f:
#             f.write(uploaded_file.getbuffer())
#         return 1
#     except:
#         return 0

# def feature_extraction(img_path,model):
#     img = image.load_img(img_path, target_size=(224, 224))
#     img_array = image.img_to_array(img)
#     expanded_img_array = np.expand_dims(img_array, axis=0)
#     preprocessed_img = preprocess_input(expanded_img_array)
#     result = model.predict(preprocessed_img).flatten()
#     normalized_result = result / norm(result)

#     return normalized_result

# def recommend(features,feature_list):
#     neighbors = NearestNeighbors(n_neighbors=6, algorithm='brute', metric='euclidean')
#     neighbors.fit(feature_list)

#     distances, indices = neighbors.kneighbors([features])

#     return indices

# # steps
# # file upload -> save
# uploaded_file = st.file_uploader("Choose an image")
# if uploaded_file is not None:
#     if save_uploaded_file(uploaded_file):
#         # display the file
#         display_image = Image.open(uploaded_file)
#         st.image(display_image)
#         # feature extract
#         features = feature_extraction(os.path.join("uploads",uploaded_file.name),model)
#         #st.text(features)
#         # recommendention
#         indices = recommend(features,feature_list)
#         # show
#         col1,col2,col3,col4,col5 = st.columns(5)

#         with col1:
#             st.image(filenames[indices[0][0]])
#         with col2:
#             st.image(filenames[indices[0][1]])
#         with col3:
#             st.image(filenames[indices[0][2]])
#         with col4:
#             st.image(filenames[indices[0][3]])
#         with col5:
#             st.image(filenames[indices[0][4]])
#     else:
#         st.header("Some error occured in file upload")

import streamlit as st
import os
from PIL import Image
import numpy as np
import pickle
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tensorflow.keras.layers import GlobalMaxPooling2D
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
from sklearn.neighbors import NearestNeighbors
from numpy.linalg import norm

# Ensure the uploads directory exists
if not os.path.exists('uploads'):
    os.makedirs('uploads')

# Load embeddings and filenames
feature_list = np.array(pickle.load(open('embeddings.pkl', 'rb')))
filenames = pickle.load(open('filenames.pkl', 'rb'))

# Load model
model = ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
model.trainable = False

model = tf.keras.Sequential([
    model,
    GlobalMaxPooling2D()
])

# Sidebar for file upload
st.sidebar.title('Upload an Image')
uploaded_file = st.sidebar.file_uploader("Choose an image")

st.title('Fashion Recommender System')
st.markdown('### Upload an image to get similar fashion recommendations')

def save_uploaded_file(uploaded_file):
    try:
        with open(os.path.join('uploads', uploaded_file.name), 'wb') as f:
            f.write(uploaded_file.getbuffer())
        return 1
    except Exception as e:
        st.error(f"Error: {e}")
        return 0

def feature_extraction(img_path, model):
    img = image.load_img(img_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    expanded_img_array = np.expand_dims(img_array, axis=0)
    preprocessed_img = preprocess_input(expanded_img_array)
    result = model.predict(preprocessed_img).flatten()
    normalized_result = result / norm(result)

    return normalized_result

def recommend(features, feature_list):
    neighbors = NearestNeighbors(n_neighbors=6, algorithm='brute', metric='euclidean')
    neighbors.fit(feature_list)

    distances, indices = neighbors.kneighbors([features])

    return indices

if uploaded_file is not None:
    if save_uploaded_file(uploaded_file):
        # Display the file
        display_image = Image.open(uploaded_file)
        st.image(display_image, caption='Uploaded Image', use_column_width=True)
        
        with st.spinner('Extracting features and finding recommendations...'):
            # Feature extraction
            features = feature_extraction(os.path.join("uploads", uploaded_file.name), model)
            # Recommendation
            indices = recommend(features, feature_list)
        
        st.success('Recommendations found!')
        st.markdown('### Recommended Items')
        
        col1, col2, col3, col4, col5 = st.columns(5)

        with col1:
            st.image(filenames[indices[0][0]], use_column_width=True)
        with col2:
            st.image(filenames[indices[0][1]], use_column_width=True)
        with col3:
            st.image(filenames[indices[0][2]], use_column_width=True)
        with col4:
            st.image(filenames[indices[0][3]], use_column_width=True)
        with col5:
            st.image(filenames[indices[0][4]], use_column_width=True)
    else:
        st.error("Some error occurred in file upload")
