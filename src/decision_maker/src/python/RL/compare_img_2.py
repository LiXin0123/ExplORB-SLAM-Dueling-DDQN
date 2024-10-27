import numpy as np
import os
import glob
import csv
from tensorflow.keras.applications.inception_v3 import InceptionV3, preprocess_input as inception_preprocess_input
from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input as vgg_preprocess_input
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input as resnet_preprocess_input
from tensorflow.keras.preprocessing import image
from sklearn.metrics.pairwise import cosine_similarity

def calculate_image_similarity(model_name, img_path1, img_path2):
    # Load model based on the specified model name
    if model_name == 'inceptionV3':
        model = InceptionV3(weights='imagenet', include_top=False)
        preprocess_input = inception_preprocess_input
        target_size = (299, 299)
    elif model_name == 'vgg16':
        model = VGG16(weights='imagenet', include_top=False)
        preprocess_input = vgg_preprocess_input
        target_size = (224, 224)
    elif model_name == 'resnet50':
        model = ResNet50(weights='imagenet', include_top=False)
        preprocess_input = resnet_preprocess_input
        target_size = (224, 224)
    else:
        raise ValueError(f"Unsupported model: {model_name}")

    # Load and preprocess the images
    img1 = image.load_img(img_path1, target_size=target_size)
    img2 = image.load_img(img_path2, target_size=target_size)
    img1 = image.img_to_array(img1)
    img2 = image.img_to_array(img2)
    img1 = np.expand_dims(img1, axis=0)
    img2 = np.expand_dims(img2, axis=0)
    img1 = preprocess_input(img1)
    img2 = preprocess_input(img2)

    # Extract features using the model
    features1 = model.predict(img1)
    features2 = model.predict(img2)

    # Calculate cosine similarity
    similarity = cosine_similarity(features1.reshape(1, -1), features2.reshape(1, -1))[0][0]
    return similarity

def compare_folder_images_to_img(folder_path, img1_path, model_name, output_csv_path):
    # Open the CSV file to write the similarity data
    with open(output_csv_path, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["Image Path", "Similarity (%)"])  # Header row

        # Iterate over all images in the folder
        for img2_path in glob.glob(os.path.join(folder_path, '*.png')):
            # Calculate similarity
            similarity = calculate_image_similarity(model_name, img1_path, img2_path)

            # Convert similarity to percentage and round
            similarity_percentage = round(similarity * 100, 2)

            # Write to CSV
            writer.writerow([img2_path, similarity_percentage])
            print(f"Similarity between {img1_path} and {img2_path}: {similarity_percentage}%")

# Example usage
img1_path = '/home/lx23/Downloads/explORB-SLAM-RL/src/decision_maker/src/python/RL/similar_images/base/train_result_3.png'
folder_path = '/home/lx23/Downloads/explORB-SLAM-RL/src/decision_maker/src/python/RL/rviz_results/aws_bookstore/dueling_ddqn/20/all/new'
model_name = 'inceptionV3'  # Choose between 'inceptionV3', 'vgg16', or 'resnet50'
output_csv_path = '/home/lx23/Downloads/explORB-SLAM-RL/src/decision_maker/src/python/RL/csv/similarity/AWS_bookstore_Explorb_Dueling_DDQN.csv'

compare_folder_images_to_img(folder_path, img1_path, model_name, output_csv_path)
