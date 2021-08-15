import streamlit as st
from PIL import Image
import matplotlib.pyplot as plt
import tensorflow_hub as hub
import tensorflow as tf
import numpy as np
from tensorflow import keras
from tensorflow.keras.models import load_model
from tensorflow.keras import preprocessing
import time

fig = plt.figure()

st.sidebar.subheader('About the Creator:')
st.sidebar.markdown('Ritika Chendvenkar')
st.sidebar.markdown('National College of Ireland, Dublin')


st.title('Mango Pest Classifier WebApp')

st.markdown("Welcome to this simple web application that classifies the type of pest that has infected the mango crop. The images are classified into sixteen different classes namely:")

st.markdown('''
	- apoderus_javanicus
	- aulacaspis_tubercularis
	- ceroplastes_rubens
	- cisaberoptus_kenyae
	- dappula_tertia
	- dialeuropora_decempuncta
	- erosomyia_sp
	- icerya_seychellarum
	- ischnaspis_longirostris
	- mictis_longicornis
	- neomelicharia_sparsa
	- normal
	- orthaga_euadrusalis
	- procontarinia_matteiana
	- procontarinia_rubus
	- valanga_nigricornis
     ''')

def main():
    file_uploaded = st.file_uploader("Upload Image File", type=["png","jpg","jpeg"])
    class_btn = st.button("Classify")
    if file_uploaded is not None:    
        image = Image.open(file_uploaded)
        st.image(image, caption='Uploaded Image', use_column_width=True)
        
    if class_btn:
        if file_uploaded is None:
            st.write("Invalid command, please upload an image")
        else:
            with st.spinner('Model working....'):
                plt.imshow(image)
                plt.axis("off")
                predictions = predict(image)
                time.sleep(1)
                st.success('Classified')
                st.write(predictions)
                #st.pyplot(fig)


def predict(image):
    classifier_model = "cnn_with_weights_dir.h5"
    IMAGE_SHAPE = (224, 224,3)
    model = keras.models.load_model(classifier_model)
    test_image = image.resize((224,224))
    test_image = preprocessing.image.img_to_array(test_image)
    test_image = np.expand_dims(test_image, axis=0)

    class_names_with_indices =['apoderus_javanicus',
								 'aulacaspis_tubercularis',
								 'ceroplastes_rubens',
								 'cisaberoptus_kenyae',
								 'dappula_tertia',
								 'dialeuropora_decempuncta',
								 'erosomyia_sp',
								 'icerya_seychellarum',
								 'ischnaspis_longirostris',
								 'mictis_longicornis',
								 'neomelicharia_sparsa',
								 'normal',
								 'orthaga_euadrusalis',
								 'procontarinia_matteiana',
								 'procontarinia_rubus',
								 'valanga_nigricornis'
								 ]


    predictions = model.predict(test_image)
    scores = tf.nn.softmax(predictions)
    scores = scores.numpy()

    
    result = f"{class_names_with_indices[np.argmax(scores)]} with a { (100 * np.max(scores)).round(2) } % confidence." 
    return result




if __name__ == "__main__":
    main()
