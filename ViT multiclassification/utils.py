import os
import sys
import json
import pickle
import random
import pandas as pd
import torch
from tqdm import tqdm
from typing import List
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter
##---------just for test--------------
from deal_dataset import process_class
from my_dataset import process_labels

from sklearn.model_selection import train_test_split
from sklearn.utils import resample
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

##---------just for test--------------

import random
import numpy as np
import torch

def get_class_counts(df, all_classes):
    counts = {class_name: 0 for class_name in all_classes}
    for class_name in all_classes:
        counts[class_name] = len(df[df['Finding Labels'] == class_name])
    return counts


def plot_detailed_class_distribution(train_df, test_df, all_classes):
    train_counts = get_class_counts(train_df, all_classes)
    test_counts = get_class_counts(test_df, all_classes)

    labels = np.arange(len(all_classes))
    train_values = [train_counts[class_name] for class_name in all_classes]
    test_values = [test_counts[class_name] for class_name in all_classes]
    width = 0.35

    fig, ax = plt.subplots(figsize=(15, 8))

    rects1 = ax.bar(labels - width / 2, train_values, width, label='Train', color='blue')
    rects2 = ax.bar(labels + width / 2, test_values, width, label='Test', color='orange')

    ax.set_xlabel('Class Name')
    ax.set_ylabel('Number of Images')
    ax.set_title('Number of images per class in Train and Test sets')
    ax.set_xticks(labels)
    ax.set_xticklabels(all_classes, rotation=45, ha='right')
    ax.legend()

    plt.tight_layout()
    plt.savefig('class_distribution.png', dpi=300, bbox_inches='tight')
    plt.show()




def read_split_data(root: str, all_c: List[str], csv_file: str, test_rate: float = 0.2):
    seed = 98
    random.seed(seed)  # Python
    np.random.seed(seed)  # NumPy
    torch.manual_seed(seed)  # PyTorch

    assert os.path.exists(root), "dataset root: {} does not exist.".format(root)
    data_entry = pd.read_csv(csv_file)

    unwanted_values = ['Emphysema','Cardinomegaly','Edema','Fibrosis','Hernia','Pneumonia','Pleural_Thickening','Consolidation','Cardiomegaly']
    data_entry = data_entry[~data_entry['Finding Labels'].str.contains('|'.join(unwanted_values))]
    unique_classes = (data_entry['Finding Labels'].unique()).tolist()

    target_count = 1000

    resampled_df = pd.DataFrame(columns=data_entry.columns)
    for class_name in unique_classes:
        class_df = data_entry[data_entry['Finding Labels'] == class_name]

        # Calculate the current class count
        class_count = len(class_df)

        if class_count < target_count:
            # Upsample the class
            upsample_factor = target_count // class_count
            resampled_class_df = resample(class_df, replace=True, n_samples=class_count * upsample_factor,
                                          random_state=42)
            resampled_df = resampled_df.append(resampled_class_df)
        elif class_count > target_count:
            # Downsample the class
            downsample_factor = class_count // target_count
            resampled_class_df = resample(class_df, replace=False, n_samples=class_count // downsample_factor,
                                          random_state=42)
            resampled_df = resampled_df.append(resampled_class_df)
        else:
            # No need for resampling, class_count equals target_count
            resampled_df = resampled_df.append(class_df)

    data_entry = resampled_df
    file_class_mapping = {row[0]: row[1] for _, row in data_entry.iterrows()}
    y = data_entry['Finding Labels']

    X_train, X_test, _, _ = train_test_split(data_entry, y, test_size=0.2, random_state=42)


    # Filter out unwanted classes
    unwanted_values = ['Emphysema','Cardinomegaly','Edema','Fibrosis','Hernia','Pneumonia','Pleural_Thickening','Consolidation','Cardiomegaly']
    all_c = list(filter(lambda x: x not in unwanted_values, all_c))

    # Create a mapping from class name to index
    class_indices = dict((k, v) for v, k in enumerate(all_c))
    json_str = json.dumps(dict((val, key) for key, val in class_indices.items()), indent=4)
    with open('class_indices.json', 'w') as json_file:
        json_file.write(json_str)

    supported = [".jpg", ".JPG", ".png", ".PNG"]

    train_images_path = X_train.iloc[:, 0].tolist()
    train_images_path = [os.path.join(root, item) for item in train_images_path]
    train_images_label = X_train.iloc[:, 1].tolist()
    train_images_label = [process_labels(item, all_c).tolist() for item in train_images_label]

    test_images_path = X_test.iloc[:, 0].tolist()
    test_images_path = [os.path.join(root, item) for item in test_images_path]
    test_images_label = X_test.iloc[:, 1].tolist()
    test_images_label = [process_labels(item, all_c).tolist() for item in test_images_label]

    # Count the label distribution for the train and test images
    train_label_count = [0]*len(all_c)
    test_label_count = [0]*len(all_c)

    for label in train_images_label:
        for i, x in enumerate(label):
            if x == 1:
                train_label_count[i] += 1

    for label in test_images_label:
        for i, x in enumerate(label):
            if x == 1:
                test_label_count[i] += 1

    # Print the count of each label for train and test sets
    for i in unique_classes:
        print(f'Count of label {i} in train set: {target_count*0.8}, in test set: {target_count*0.2}')

    print("{} images were found in the dataset.".format(sum(train_label_count) + sum(test_label_count)))
    assert len(train_images_path) > 0, "number of training images must greater than 0."
    assert len(test_images_path) > 0, "number of testing images must greater than 0."
    plot_detailed_class_distribution(X_train, X_test, unique_classes)
    return train_images_path, train_images_label, test_images_path, test_images_label, all_c




##-----just for test in coding------
if __name__ == '__main__':
    all_categorie = process_class()
    train_images_path, train_images_label, val_images_path, val_images_label,all_categorie = read_split_data("image_training",
                                                                                       all_categorie,
                                                                                               'Data_Entry_2017_v2020.csv')




def plot_data_loader_image(data_loader):
    batch_size = data_loader.batch_size
    plot_num = min(batch_size, 4)

    json_path = './class_indices.json'
    assert os.path.exists(json_path), json_path + " does not exist."
    json_file = open(json_path, 'r')
    class_indices = json.load(json_file)

    for data in data_loader:
        images, labels = data
        for i in range(plot_num):
            # [C, H, W] -> [H, W, C]
            img = images[i].numpy().transpose(1, 2, 0)
            # 反Normalize操作
            img = (img * [0.229, 0.224, 0.225] + [0.485, 0.456, 0.406]) * 255
            label = labels[i].item()
            plt.subplot(1, plot_num, i + 1)
            plt.xlabel(class_indices[str(label)])
            plt.xticks([])  # 去掉x轴的刻度
            plt.yticks([])  # 去掉y轴的刻度
            plt.imshow(img.astype('uint8'))
        plt.show()


def write_pickle(list_info: list, file_name: str):
    with open(file_name, 'wb') as f:
        pickle.dump(list_info, f)


def read_pickle(file_name: str) -> list:
    with open(file_name, 'rb') as f:
        info_list = pickle.load(f)
        return info_list


# ---modify the code of training and evaluation beacuse should suit for mulitiple-lable
def train_one_epoch(model, optimizer, data_loader, device, epoch,class_number):
    model.train()
    #可以尝试修改一下softmax的改进版本的函数会不会更好一些
    loss_function = torch.nn.BCEWithLogitsLoss()
    # loss_function = torch.nn.CrossEntropyLoss()
    accu_loss = torch.zeros(1).to(device)  # Cumulative losses
    accu_num = torch.zeros(1).to(device)  # Cumulative predict correct
    accu_num_per_class = torch.zeros(7).to(device)
    num_per_class = torch.zeros(7).to(device)
    optimizer.zero_grad()

    sample_num = 0
    data_loader = tqdm(data_loader, file=sys.stdout)
    for step, data in enumerate(data_loader):
        images, labels = data
        sample_num += images.shape[0]

        pred = model(images.to(device))
        pred_classes = (torch.sigmoid(pred) > 0.5).float()
        #can not use eq function beacuse this is the mutiple label classification.
        accu_num += (pred_classes == labels.to(device)).float().sum().item()

        accu_num_per_class += (pred_classes == labels.to(device)).sum(0).float()
        num_per_class += labels.to(device).sum(0).float()

        loss = loss_function(pred, labels.to(device))
        loss.backward()
        accu_loss += loss.detach()

        data_loader.desc = "[train epoch {}] loss: {:.3f}, acc: {:.3f}".format(epoch,
                                                                               accu_loss.item() / (step + 1),
                                                                               accu_num.item() / (class_number*sample_num))

        if not torch.isfinite(loss):
            print('WARNING: non-finite loss, ending training ', loss)
            sys.exit(1)

        optimizer.step()
        optimizer.zero_grad()
    print('acc per class', accu_num_per_class/sample_num)

    return accu_loss.item() / (step + 1), accu_num.item() / (class_number*sample_num)


# ---modify the code of training and evaluation beacuse should suit for mulitiple-lable
@torch.no_grad()
def evaluate(model, data_loader, device, epoch, class_number,class_names):
    loss_function = torch.nn.BCEWithLogitsLoss()  # Multiple-choice
    # loss_function = torch.nn.CrossEntropyLoss()

    model.eval()

    accu_num = torch.zeros(1).to(device)   # Cumulative losses
    accu_loss = torch.zeros(1).to(device)  # Cumulative losses
    accu_num_per_class = torch.zeros(7).to(device)
    num_per_class = torch.zeros(7).to(device)

    sample_num = 0
    data_loader = tqdm(data_loader, file=sys.stdout)
    for step, data in enumerate(data_loader):
        images, labels = data
        sample_num += images.shape[0]

        pred = model(images.to(device))
        # Change the calculation of accuracy
        pred_classes = (torch.sigmoid(pred) > 0.5).float()
        accu_num += (pred_classes == labels.to(device)).float().sum().item()

        accu_num_per_class += (pred_classes == labels.to(device)).sum(0).float()
        num_per_class += labels.to(device).sum(0).float()

        loss = loss_function(pred, labels.to(device))
        accu_loss += loss

        data_loader.desc = "[valid epoch {}] loss: {:.3f}, acc: {:.3f}".format(epoch,
                                                                               accu_loss.item() / (step + 1),
                                                                               accu_num.item() / (class_number*sample_num))

    for i, class_name in enumerate(class_names):
        print('Accuracy for {}: {:.4f}'.format(class_name, accu_num_per_class[i].item() / sample_num))

    return accu_loss.item() / (step + 1), accu_num.item() / (class_number*sample_num)




def plot_and_save_metrics(train_loss, train_acc, val_loss, val_acc, save_path="metrics.png"):
    epochs = range(1, len(train_loss) + 1)

    formatter = FormatStrFormatter('%.3f')

    plt.figure(figsize=(12, 6))

    plt.subplot(1, 2, 1)
    plt.plot(epochs, train_loss, 'bo', label='Training loss')
    plt.plot(epochs, val_loss, 'b', label='Validation loss')
    plt.title('Training and validation loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.gca().yaxis.set_major_formatter(formatter)

    plt.subplot(1, 2, 2)
    plt.plot(epochs, train_acc, 'bo', label='Training acc')
    plt.plot(epochs, val_acc, 'b', label='Validation acc')
    plt.title('Training and validation accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.gca().yaxis.set_major_formatter(formatter)

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()