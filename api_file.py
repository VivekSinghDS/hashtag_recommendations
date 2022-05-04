import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial.distance import cosine
import pickle
#fastapi imports

import uvicorn
from fastapi import FastAPI, status
from pydantic import BaseModel
from fastapi.responses import FileResponse
import pandas

img_shape = (160, 160, 3)

# Create the base model from the pre-trained model MobileNet V2
base_model = MobileNetV2(input_shape=img_shape, include_top=False, weights='imagenet')

global_average_layer = tf.keras.layers.GlobalAveragePooling2D()

neural_network = tf.keras.Sequential([
    base_model,
    global_average_layer,
])


def prepare_image(img_path, height=160, width=160, where='local'):
    img = tf.io.read_file(img_path)

    img = tf.image.decode_image(img)

    img = tf.cast(img, tf.float32)

    img = (img / 127.5) - 1

    img = tf.image.resize(img, (height, width))

    if img.shape != (160, 160, 3):
        img = tf.concat([img, img, img], axis=2)

    return img


def extract_features(image, neural_network):
    image_np = image.numpy()
    image_np = np.expand_dims(image_np, axis=0)
    deep_features = neural_network.predict(image_np)[0]
    return deep_features




with open('recommender_df.pickle', 'rb') as file:
    recommender_df = pickle.load(file)
    file.close()

with open('hashtag_features.pickle', 'rb') as file:
    hashtag_features = pickle.load(file)
    file.close()

with open('hashtags_df.pickle', 'rb') as file:
    hashtags_df = pickle.load(file)
    file.close()



def find_neighbor_vectors(image_path, k=5, recommender_df=recommender_df):
    """Find image features (user vectors) for similar images."""
    prep_image = prepare_image(image_path, where='local')
    pics = extract_features(prep_image, neural_network)
    rdf = recommender_df.copy()
    rdf['dist'] = rdf['deep_features'].apply(lambda x: cosine(x, pics))
    rdf = rdf.sort_values(by='dist')
    return rdf.head(k)


def generate_hashtags(image_path):
    fnv = find_neighbor_vectors(image_path, k=5, recommender_df=recommender_df)
    # Find the average of the 5 user features found based on cosine similarity.
    features = []
    for item in fnv.features.values:
        features.append(item)

    avg_features = np.mean(np.asarray(features), axis=0)

    # Add new column to the hashtag features which will be the dot product with the average image(user) features
    hashtag_features['dot_product'] = hashtag_features['features'].apply(lambda x: np.asarray(x).dot(avg_features))

    # Find the 10 hashtags with the highest feature dot products
    final_recs = hashtag_features.sort_values(by='dot_product', ascending=False).head(5)
    # Look up hashtags by their numeric IDs
    output = []
    for hashtag_id in final_recs.id.values:
        output.append(hashtags_df.iloc[hashtag_id]['hashtag'])
    return output


def show_results(test_image):
    img = mpimg.imread(test_image)


    recommended_hashtags = generate_hashtags(test_image)


    return recommended_hashtags



app = FastAPI()
class Item(BaseModel):
    image_path:str
@app.post('/get_recommedations/')
async def recommend(item:Item):
    item_dict = item.dict()

    hashtags_output = show_results(item_dict['image_path'])
    return hashtags_output

@app.get('/')
async def recommend(item:Item):
    return 'Sanity check done '

if __name__ == '__main__':
    uvicorn.run(app, host = "0.0.0.0", port = 7000)