from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from vit_model import VisionTransformer, vit_base_patch16_224_in21k
from torchvision.models import resnet34
from utils import read_split_data
from deal_dataset import process_class
from torchvision import transforms
from my_dataset import MyDataSet
import numpy as np
from tqdm import tqdm
import torch

# load datasets
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
all_categorie = process_class()
train_images_path, train_images_label, val_images_path, val_images_label, _ = read_split_data('image_training',
                                                                                               all_categorie,
                                                                                               'Data_Entry_2017_v2020.csv')
transform = transforms.Compose([transforms.ToTensor(),
                                transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])])
val_dataset = MyDataSet(images_path=val_images_path,
                        images_class=val_images_label,
                        classes=all_categorie,
                        transform=transform)  # Use the defined transform

val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=8, shuffle=False)


# Load the weights, keeping only the keys that match the model
resnet_model = resnet34(pretrained=False)
resnet_model.to(device)
pretrained_dict = torch.load('weight_result/model_baseline-100.pth')
model_dict = resnet_model.state_dict()
pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict and "fc" not in k}
model_dict.update(pretrained_dict)
resnet_model.load_state_dict(model_dict)
resnet_feature_model = torch.nn.Sequential(*(list(resnet_model.children())[:-2]))


# Load the trained ViT model
vit_model = vit_base_patch16_224_in21k(num_classes=7, has_logits=False)
vit_model.to(device)
vit_model.load_state_dict(torch.load('weight_result/model-500.pth'))


# feature extract extraction
resnet_features = []
vit_features = []
with torch.no_grad():
    for images, _ in tqdm(val_loader, desc="Extracting features"):
        images = images.to(device)
        resnet_feature = resnet_feature_model(images).mean([2, 3]).cpu().numpy()
        vit_feature = vit_model.forward_features(images).cpu().numpy()  # Transformer层的输出，形状为 (8, 768)
        resnet_features.extend(resnet_feature)
        vit_features.extend(vit_feature)


resnet_features = np.array(resnet_features)
vit_features = np.array(vit_features)

pca_resnet_3d = PCA(n_components=3)
resnet_pca_features_3d = pca_resnet_3d.fit_transform(resnet_features)

# Perform PCA on ViT features and reduce to 3 dimensions
pca_vit_3d = PCA(n_components=3)
vit_pca_features_3d = pca_vit_3d.fit_transform(vit_features)

# 3D visualization
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(resnet_pca_features_3d[:, 0], resnet_pca_features_3d[:, 1], resnet_pca_features_3d[:, 2], c='b', label='ResNet', alpha=0.5, s=10)
ax.scatter(vit_pca_features_3d[:, 0], vit_pca_features_3d[:, 1], vit_pca_features_3d[:, 2], c='r', label='ViT', alpha=0.5, s=10)
ax.set_xlabel("First Principal Component")
ax.set_ylabel("Second Principal Component")
ax.set_zlabel("Third Principal Component")
plt.legend()
plt.show()
# View from the side
fig = plt.figure()  # Create a new figure for the side view
ax2 = fig.add_subplot(111, projection='3d')  # Change 122 to 111 since this is the only plot in this figure
ax2.scatter(resnet_pca_features_3d[:, 0], resnet_pca_features_3d[:, 1], resnet_pca_features_3d[:, 2], c='b', label='ResNet', alpha=0.5, s=10)
ax2.scatter(vit_pca_features_3d[:, 0], vit_pca_features_3d[:, 1], vit_pca_features_3d[:, 2], c='r', label='ViT', alpha=0.5, s=10)
ax2.set_xlabel("First Principal Component")
ax2.set_ylabel("Second Principal Component")
ax2.set_zlabel("Third Principal Component")
ax2.view_init(10, 185)  # Adjust the viewi`ng angle for better visualization (side view)
plt.legend()
plt.show()

# Perform PCA on ResNet and ViT features
pca_resnet = PCA(n_components=2)
resnet_pca_features = pca_resnet.fit_transform(resnet_features)

pca_vit = PCA(n_components=2)
vit_pca_features = pca_vit.fit_transform(vit_features)
# 2D visualization
plt.figure()
plt.scatter(resnet_pca_features[:, 0], resnet_pca_features[:, 1], c='b', label='ResNet', alpha=0.5, s=10)
plt.scatter(vit_pca_features[:, 0], vit_pca_features[:, 1], c='r', label='ViT', alpha=0.5, s=10)
plt.xlabel("First Principal Component")
plt.ylabel("Second Principal Component")
plt.legend()
plt.show()