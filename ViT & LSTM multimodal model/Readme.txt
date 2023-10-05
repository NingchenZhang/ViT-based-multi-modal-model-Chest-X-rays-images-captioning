-----Environment-----
torch version: 2.0.1+cu117
numpy version: 1.25.2
pandas version: 1.5.3
torchtext version: 0.15.2
tensorboardX version: 2.6.2
My computer GPU is RTX2060 and this model training on the RTX AT5000 each epoch spends 1 min.
--------------------------------
----------Datasets-------------
The datasets after reshape and rename process is store in the Google drive. 
The link is:
https://drive.google.com/file/d/1MWYOsSccAY5OVgW6nkJidwU8f3nmf3CH/view?usp=drive_link
---------------------------------
---------weights-----------------
The model_vitfinal.pth file is the ViT weight after training on the CXR8 dataset which as the input weight for the encoder 
in the multimodal model.
The model_multimodel_final.pth file is the ViT & LSTM multimodal model weight which used in the Greedy search.
The vit_base_patch16_224_in21k.pth is the original ViT model weight.
---------------------------------

----Running----------
The Training python file is new_model.py file
The Greedy search which can input the X-rays images and predict report. You can just find all test images in my main function in this file.
But it can only run a few of the provided test files, if you want to try other test images of your own 
you need to re-modify the main function part of the file, greedy search algorithm is used in general.
--------------------------
The Tokenizer method and some pre-process for dataset function methods are in the utils.py file.
The python files of encoder and decoder correspond to the encoder and decoder parts of the multimodal model.

