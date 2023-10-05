from PIL import Image
import torch
from torch.utils.data import Dataset
from deal_dataset import process_class

def process_labels(labels, classes):
    # Initialize the label vector, setting all elements to 0
    label_vector = torch.zeros(len(classes))

    # For each class that appears in this sample, set the corresponding element to 1
    for label in labels.split('|'):
        if label in classes:  # Only update the vector for classes that are still in the class list
            label_index = classes.index(label)
            label_vector[label_index] = 1

    return label_vector

class MyDataSet(Dataset):
    """Customised datasets"""

    def __init__(self, images_path: list, images_class: list, classes:list,transform=None):
        self.images_path = images_path#[:100]
        self.images_class = images_class
        self.classes = classes
        self.transform = transform

    def __len__(self):
        return len(self.images_path)

    def __getitem__(self, item):
        img = Image.open(self.images_path[item])
        # RGB，L
        if img.mode != 'RGB':
            raise ValueError("image: {} isn't RGB mode.".format(self.images_path[item]))
        #test_index = 0

        #print(type(self.images_class[test_index]))
        #print(self.images_class[test_index])
        #label = process_labels(self.images_class[item], self.classes)
        label = self.images_class[item]

        if self.transform is not None:
            img = self.transform(img)

        return img, label

    @staticmethod
    def collate_fn(batch):

        # https://github.com/pytorch/pytorch/blob/67b7e751e6b5931a9f45274653f4f653a4e6cdf6/torch/utils/data/_utils/collate.py
        images, labels = tuple(zip(*batch))

        images = torch.stack(images, dim=0)
        labels = torch.as_tensor(labels)
        return images, labels



