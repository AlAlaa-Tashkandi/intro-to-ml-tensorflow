"""
Created on Wed May 27 21:23:51 2020

@author: AlAlaa Tashkandi
"""

import argparse
import tensorflow as tf
import tensorflow_hub as hub
import numpy as np
import json
from PIL import Image

    
def process_image(image):
    image_size = 224
    image = tf.cast(image, tf.float32)
    image = tf.image.resize(image, (image_size, image_size))
    image /= 255
    numpy_image = image.numpy()
    return numpy_image


def predict(image_path, model, top_k):
    im = Image.open(image_path)
    numpy_image = np.asarray(im)
    processed_np_image = process_image(numpy_image)
    fourd_processed_np_image = np.expand_dims(processed_np_image,axis=0)
    ps = model.predict(fourd_processed_np_image)
    #print(ps)
    #https://stackoverflow.com/questions/6910641/how-do-i-get-indices-of-n-maximum-values-in-a-numpy-array
    probes = np.sort(ps[0])[-top_k:][::-1]
    classes = np.argsort(ps[0])[-top_k:][::-1]
    

    return probes,classes+1

def main():
    check_model = tf.keras.models.load_model(input_model,custom_objects={'KerasLayer':hub.KerasLayer})
    
    probs, classes =predict(input_image,check_model,input_top_k)
    
    class_names_list = [class_names.get(str(key)) for key in classes]
    
    print(probs)
    #print(classes)
    print(class_names_list)
    for i in range(input_top_k):
        print("The image is predicted to be {} with probability of {:.2f}%".format(class_names_list[i],probs[i]*100))
    
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Predict image class.')
    parser.add_argument('input_image', 
                        action="store",
                        help='provide image file path/name')
    parser.add_argument('input_model', 
                        action="store",
                        help='provide trained model path/name')
    parser.add_argument('--top_k', 
                        action='store',
                        dest='input_top_k',
                        type=int,
                        default=1,
                        help='provide the top k classes as int')
    parser.add_argument('--category_names', 
                        action='store',
                        dest='input_category_names',
                        default='./label_map.json',
                        help='provide the top k classes as int')
    parser.add_argument('--version', action='version',
                        version='%(prog)s 1.0')
    
    results = parser.parse_args()
    
    
    input_image = results.input_image
    input_model = results.input_model
    input_top_k = results.input_top_k
    input_category_names = results.input_category_names
    
    with open(input_category_names, 'r') as f:
        class_names = json.load(f)
    main()
    



