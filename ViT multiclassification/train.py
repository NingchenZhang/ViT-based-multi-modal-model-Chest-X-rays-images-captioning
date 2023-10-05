import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'
import math
import argparse
import torch
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms 
from torchvision.models import resnet34
import torch.nn as nn
from my_dataset import MyDataSet
from vit_model import vit_base_patch16_224_in21k as create_model #get the model want to use from vit_model.py
from utils import read_split_data, train_one_epoch, evaluate, plot_and_save_metrics
from deal_dataset import process_class
from vit_model import DropPath
#if torch.cuda.is_available() else "cpu"

def main(args):
    device = torch.device(args.device )
    print('Training on device:', device)
    if os.path.exists("./weights") is False:
        os.makedirs("./weights")

    all_categorie = process_class()

    tb_writer = SummaryWriter()# visulazation for the training process
#----need to change the path here---------------
    train_images_path, train_images_label, val_images_path, val_images_label,all_cfinal = read_split_data(args.data_path,
                                                                                               all_categorie,
                                                                                               'Data_Entry_2017_v2020.csv')
# ----need to change the path here---------------
    all_categorie = all_cfinal
    print(all_categorie)
    #Data Enhancement
    data_transform = {
        "train": transforms.Compose([transforms.Resize(224),
                                     transforms.RandomHorizontalFlip(),
                                     transforms.RandomVerticalFlip(),
                                     transforms.RandomRotation(90),
                                     transforms.RandomChoice([
                                        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2),
                                        transforms.RandomAffine(degrees=15, translate=(0.1, 0.1), scale=(0.8, 1.2), shear=15),
                                        transforms.GaussianBlur(kernel_size=5),
                                     ]),
                                     transforms.ToTensor(),
                                     transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])]),
        "val": transforms.Compose([transforms.Resize(224),
                                   transforms.ToTensor(),
                                   transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])])}

    # Instantiating the training dataset
    train_dataset = MyDataSet(images_path=train_images_path,
                              images_class=train_images_label,
                              classes=all_categorie,
                              transform= data_transform['train'])  # Use the defined transform

    # Instantiating the validation dataset
    val_dataset = MyDataSet(images_path=val_images_path,
                            images_class=val_images_label,
                            classes=all_categorie,
                           transform=data_transform['val'])  # Use the defined transform

    batch_size = args.batch_size
    nw = min([os.cpu_count(), batch_size if batch_size > 1 else 0, 8])  # number of process
    print('Using {} dataloader workers every process'.format(nw))
    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size=batch_size,
                                               shuffle=True,
                                               pin_memory=True,
                                               num_workers=nw,
                                               collate_fn=train_dataset.collate_fn)

    val_loader = torch.utils.data.DataLoader(val_dataset,
                                             batch_size=batch_size,
                                             shuffle=False,
                                             pin_memory=True,
                                             num_workers=nw,
                                             collate_fn=val_dataset.collate_fn)
    model = create_model(num_classes=args.num_classes, has_logits=False).to(device)

    x = torch.randn(1, 3, 224, 224).to(device)  # Assuming input size is (3, 224, 224)
    outputs = model(x)

    #if donnot have the pre-weight drop
    if args.weights != "":
        assert os.path.exists(args.weights), "weights file: '{}' not exist.".format(args.weights)
        weights_dict = torch.load(args.weights, map_location=device)#["state_dict"]
        # delete the weight donnot use
        del_keys = ['head.weight', 'head.bias'] if model.has_logits \
            else ['pre_logits.fc.weight', 'pre_logits.fc.bias', 'head.weight', 'head.bias']
        for k in del_keys:
            del weights_dict[k]
        print(model.load_state_dict(weights_dict, strict=False))
    # freeze training for L image should freeze some layers.########################
    if args.freeze_layers:
        for name, para in model.named_parameters():
            # head, pre_logits
            if "head" not in name and "pre_logits" not in name:
                para.requires_grad_(False)
            else:
                print("training {}".format(name))

    pg = [p for p in model.parameters() if p.requires_grad]

    optimizer = optim.SGD(pg, lr=args.lr, momentum=0.9, weight_decay=5E-5)
    # optimizer = optim.Adam(pg, lr=args.lr, weight_decay=5E-5)

    lf = lambda x: ((1 + math.cos(x * math.pi / args.epochs)) / 2) * (1 - args.lrf) + args.lrf  # cosine
    scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lf)

    # Initialize lists to store loss and accuracy for each epoch
    train_losses = []
    train_accuracies = []
    val_losses = []
    val_accuracies = []
    # drop_prob = 0.2  # replace 0.2 with your desired drop_prob
    # for module in model.modules():
    #     if isinstance(module, DropPath):
    #         module.drop_prob = drop_prob
    for epoch in range(1,args.epochs+1):

        # train
        train_loss, train_acc = train_one_epoch(model=model,
                                                optimizer=optimizer,
                                                data_loader=train_loader,
                                                device=device,
                                                epoch=epoch,
                                                class_number=args.num_classes)

        scheduler.step()

        # validate
        val_loss, val_acc = evaluate(model=model,
                                     data_loader=val_loader,
                                     device=device,
                                     epoch=epoch,
                                     class_number=args.num_classes)

        # Store metrics for this epoch
        train_losses.append(train_loss)
        train_accuracies.append(train_acc)
        val_losses.append(val_loss)
        val_accuracies.append(val_acc)

        tags = ["train_loss", "train_acc", "val_loss", "val_acc", "learning_rate"]
        tb_writer.add_scalar(tags[0], train_loss, epoch)
        tb_writer.add_scalar(tags[1], train_acc, epoch)
        tb_writer.add_scalar(tags[2], val_loss, epoch)
        tb_writer.add_scalar(tags[3], val_acc, epoch)
        tb_writer.add_scalar(tags[4], optimizer.param_groups[0]["lr"], epoch)

        # choose how many epoch to save one weight file.
        if epoch % args.save_interval == 0:
            torch.save(model.state_dict(), "./weight_result/model-{}.pth".format(epoch))

    # After training, plot and save metrics
        plot_and_save_metrics(train_losses, train_accuracies, val_losses, val_accuracies, "weight_result/metrics_ViT.png")


class Options:
    def __init__(self):
        #---rembemer to change the class number
        self.num_classes = 7
        self.epochs = 200
        self.batch_size = 8
        self.lr = 0.001
        self.lrf = 0.001
        self.data_path = "image_training"
        self.model_name = ''
        self.weights = 'weights/vit_base_patch16_224_in21k.pth'#Load the pre-trained model
        self.freeze_layers = False
        self.device = 'cuda:0'
        self.save_interval = 10 #specify the number of epochs to save the weight

if __name__ == '__main__':
    opt = Options()

    main(opt)
