from PIL import Image
import os
import glob

def crop_images_in_folder(folder_path, output_folder, crop_width=1300, crop_height=1100):
    # Create the output folder if it does not exist
    os.makedirs(output_folder, exist_ok=True)
    
    # Get all image files in the folder
    for img_path in glob.glob(os.path.join(folder_path, '*.png')):
        with Image.open(img_path) as img:
            # Get the dimensions of the image
            width, height = img.size
            
            # Calculate the coordinates for cropping from the bottom-right corner
            left = width - crop_width
            top = height - crop_height
            right = width
            bottom = height - 50
            
            # Perform the cropping
            cropped_img = img.crop((left, top, right, bottom))
            
            # Save the cropped image to the output folder
            img_name = os.path.basename(img_path)
            cropped_img.save(os.path.join(output_folder, img_name))
            print(f'Cropped and saved {img_name}')

# Example usage
folder_path = '/home/lx23/Downloads/explORB-SLAM-RL/src/decision_maker/src/python/RL/rviz_results/aws_bookstore/dueling_ddqn/20/all'
output_folder = '/home/lx23/Downloads/explORB-SLAM-RL/src/decision_maker/src/python/RL/rviz_results/aws_bookstore/dueling_ddqn/20/all/new'
crop_images_in_folder(folder_path, output_folder)
