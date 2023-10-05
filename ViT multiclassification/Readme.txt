-----Environment-----
torch version: 2.0.1+cu117
numpy version: 1.25.2
pandas version: 1.5.3
torchtext version: 0.15.2
tensorboardX version: 2.6.2
My computer GPU is RTX2060 and this model training on the RTX 3090 each epoch spends 9 min.
--------------------------------
------Dataset----------
The datasets after reshape and rename process is store in the Google drive. 
The link is:
https://drive.google.com/file/d/1-0QHHRWwCd7srH7x_STQTW1LsLlgc9o5/view?usp=drive_link
The original dataset link is:
https://nihcc.app.box.com/v/ChestXray-NIHCC/folder/36938765345
-------------------------------
------Running----------------
The training process is in the training.py.
You can running the training program after download the datasets.
The utils.py is the training process loop and Vaild process loop and some data pre-process  method.
show_feature.py is the PCA function to show the feature of baseline and ViT.
if you want to get the project weight after training is in the multimodal model/weights folder.