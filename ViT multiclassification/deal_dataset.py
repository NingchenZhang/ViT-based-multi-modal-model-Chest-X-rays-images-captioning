
import os
import glob
import tarfile
import pandas as pd
from PIL import Image

def preprocess_and_save_images(tar_dir, save_dir, img_size=(224, 224)):
    """
    Unpack images from tar.gz files, convert them to RGB mode, resize them, and save them to a directory.

    Args:
    tar_dir (str): The directory containing tar.gz files.
    save_dir (str): The directory where the processed images will be saved.
    img_size (tuple): The size to which images will be resized.

    Returns:
    None
    """

    os.makedirs(save_dir, exist_ok=True)

    # get the context of all tar.gz file
    tar_files = glob.glob(os.path.join(tar_dir, '*.tar.gz'))

    for tar_file in tar_files:
        # open the tar.gz files
        with tarfile.open(tar_file, 'r:gz') as tar:
            for member in tar.getmembers():
                if member.name.endswith('.png') or member.name.endswith('.jpg'):
                    # get the images from files
                    f = tar.extractfile(member)
                    img = Image.open(f)

                    # change the images format from L to RGB
                    if img.mode != 'RGB':
                        img = img.convert('RGB')

                    # resize into 224x224
                    img = img.resize(img_size)

                    # save the images to dir
                    img.save(os.path.join(save_dir, os.path.basename(member.name)))

def process_class():
    df = pd.read_csv('Data_Entry_2017_v2020.csv')

    # get all categories
    categories = df['Finding Labels'].unique()
    all_categories = set()  # create empty set
    for labels in categories:
        all_categories.update(labels.split('|'))
    all_categorie = list(all_categories)
    return all_categorie

#Test the function of this py.file
if __name__ == '__main__':
    preprocess_and_save_images('images', 'image_training')
    process_class()

