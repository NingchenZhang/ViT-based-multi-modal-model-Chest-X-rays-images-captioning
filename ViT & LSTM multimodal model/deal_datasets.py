
import os
from PIL import Image
def preprocess_and_save_images(img_dir, save_dir, img_size=(224, 224)):

    os.makedirs(save_dir, exist_ok=True)

    # Get a list of all image files in img_dir
    img_files = os.listdir(img_dir)
    img_files = [f for f in img_files if f.endswith('.png') or f.endswith('.jpg')]

    for img_file in img_files:
        # Open the image file
        img = Image.open(os.path.join(img_dir, img_file))

        # Convert the image to RGB if it is not already
        if img.mode != 'RGB':
            img = img.convert('RGB')

        # Resize the image
        img = img.resize(img_size)

        # Save the processed image
        img.save(os.path.join(save_dir, img_file))


#Test the function of this py.file
if __name__ == '__main__':
    preprocess_and_save_images('textmodel/images', 'textmodel/images_training')


