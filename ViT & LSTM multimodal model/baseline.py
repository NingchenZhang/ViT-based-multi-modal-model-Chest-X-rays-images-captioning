import torch
import pandas as pd
import os
import math
import matplotlib.pyplot as plt
import torch.optim.lr_scheduler as lr_scheduler
from torch import nn
from decoder import  Decoder
from baseline_encoder import Image_encoder
import numpy as np
from utils import creat_tokenier,splite_data,Tokenizer,plot_and_save_metrics
from my_dataset import MyDataset
from torch.utils.data import DataLoader
from lossfunction import custom_loss
# from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from tensorboardX import SummaryWriter
from nltk.translate.bleu_score import sentence_bleu
import warnings
from datetime import datetime
warnings.filterwarnings('ignore')
warnings.simplefilter(action='ignore', category=FutureWarning)
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
rd = 98 #6688 #98 #666 #400
torch.manual_seed(rd)
torch.cuda.manual_seed(rd)
torch.backends.cudnn.deterministic = True


#Using dropout layer
class MyModel(nn.Module):
    def __init__(self, image_size, max_pad, embedding_dim, dense_dim, tokenizer, device, vit_weight=False, dropout_rate=0.1):
        super(MyModel, self).__init__()
        self.image_encoder = Image_encoder(dense_dim, dropout_rate, vit_weight).to(device)
        self.dropout_rate = dropout_rate
        vocab_size = tokenizer.get_vocab_size()
        # self.adaptive_avg_pool = nn.AdaptiveAvgPool1d(output_size=18)
        self.adaptive_avg_pool = nn.AdaptiveAvgPool1d(output_size=256)

        self.decoder = Decoder(max_pad, embedding_dim, dense_dim, vocab_size)
        self.sg = nn.Sigmoid()

    def forward(self, image1, image2, caption):
        encoder_output = self.image_encoder.encoder(image1, image2, self.dropout_rate)
        encoder_output = self.adaptive_avg_pool(encoder_output.transpose(1, 2)).transpose(1, 2)
        # encoder_output = self.sg(encoder_output)

        output = self.decoder(encoder_output, caption)
        return output

def train_process(model, device, train_loader, optimizer, epoch, text_processor):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    pbar = tqdm(train_loader, desc=f"Training epoch {epoch}")

    for i, (image1, image2, caption, target_caption) in enumerate(pbar):
        optimizer.zero_grad()
        image1 = image1.float().to(device)
        image2 = image2.float().to(device)
        # torchvision.utils.save_image(image1,"1.png", nrow=8, normalize=True, padding=2)
        # torchvision.utils.save_image(image2,"2.png", nrow=8, normalize=True, padding=2)
        # exit()
        caption = caption.to(device)
        target_caption = target_caption.to(device)
        # assert caption[:, 1:] == target_caption[:, :-2], "caption incorrect"

        # forward propagation
        output = model(image1, image2, caption)

        # get the loss
        loss = custom_loss(target_caption, output, text_processor.pad_index)  # using custom_loss function
        log_loss = loss
        # Backpropagation and optimization
        
        log_loss.backward()

        # for name, param in model.named_parameters():
        #     if param.grad is not None:
        #         grad_norm = torch.norm(param.grad)
        #         # print(f"{name}'s gradient norm: {grad_norm}")
        #         if grad_norm < 1e-5:
        #             print(f"{name}'s gradient is close to zero")
        #         # else:
        #         #     print(f"{name}'s gradient is not close to zero")

        optimizer.step()

        #Update runtime loss
        running_loss += log_loss.item()

        # calculate the accuracy
        _, predicted = torch.max(output.data, -1)
        predicted = predicted.squeeze(-1)  # now predicted has shape [4, 32]
        # mask = (target_caption != 0)  # assuming 0 is the index for padding word
        mask = (target_caption != text_processor.pad_index)
        correct += ((predicted == target_caption) & mask).sum().item()
        total += mask.sum().item()

        # get the process message
        pbar.set_postfix({"Loss": running_loss / (i + 1), "Accuracy": correct / total})

    train_loss = running_loss / len(train_loader)
    train_accuracy = correct / total

    return train_loss, train_accuracy


def get_bleu(references, predictions, mask_idx, text_processor):
    predictions = torch.argmax(predictions, dim=-1)
    batch_size = predictions.size()[0]
    bleu1 = []
    bleu2 = []
    bleu3 = []
    bleu4 = []
    for i in range(batch_size):
        reference = list(references[i, ...])
        reference = [[text_processor.itos[int(item)] for item in reference if item != mask_idx]]
        prediction = list(predictions[i, ...])
        prediction = [text_processor.itos[int(item)] for item in prediction if item != mask_idx]

        bleu1.append(sentence_bleu(reference, prediction, weights=(1, 0, 0, 0)))
        bleu2.append(sentence_bleu(reference, prediction, weights=(0.5, 0.5, 0, 0)))
        bleu3.append(sentence_bleu(reference, prediction, weights=(0.33, 0.33, 0.33, 0)))
        bleu4.append(sentence_bleu(reference, prediction, weights=(0.25, 0.25, 0.25, 0.25)))

    return np.array(bleu1).mean(), np.array(bleu2).mean(), np.array(bleu3).mean(), np.array(bleu4).mean()

def validate(model, device, val_loader, epoch, text_processor):
    model.eval()
    bleu_total = 0.0
    running_loss = 0.0
    correct = 0
    total = 0
    bleu_scores = [0, 0, 0, 0]
    pbar = tqdm(val_loader, desc=f"Validating epoch {epoch}")  # 添加进度条

    with torch.no_grad():
        for i, (image1, image2, caption, target_caption) in enumerate(val_loader):
            image1 = image1.to(device)
            image2 = image2.to(device)
            caption = caption.to(device)
            target_caption = target_caption.to(device)

            # Forward pass
            outputs = model(image1, image2, caption)

            # Remove the third dimension of 'outputs'
            outputs = outputs.squeeze(2)

            bleu1, bleu2, bleu3, bleu4 = get_bleu(target_caption, outputs, text_processor.pad_index, text_processor)
            bleu_scores[0] += bleu1
            bleu_scores[1] += bleu2
            bleu_scores[2] += bleu3
            bleu_scores[3] += bleu4

            bleu = (bleu1 + bleu2 + bleu3 + bleu4)/4
            bleu_total += bleu
            # Compute loss
            loss = custom_loss(target_caption, outputs, text_processor.pad_index)
            log_loss = loss

            predicted = outputs.view(-1, outputs.size(2))

            target_caption = target_caption.view(-1)

            # Compute accuracy
            predicted_indices = torch.argmax(predicted, dim=-1)

            # mask = (target_caption != 0)
            mask = (target_caption != text_processor.pad_index)
            correct += ((predicted_indices == target_caption) & mask).sum().item()
            total += mask.sum().item()
            # Update running loss
            running_loss += log_loss.item()
            pbar.set_postfix({"Loss": running_loss / (i + 1), "Accuracy": correct / total, "Blue": float(bleu_total)/(i+1)})
            pbar.update()

    val_loss = running_loss / len(val_loader)
    val_accuracy = correct / total
    bleu_total = bleu_total / len(val_loader)
    avg_bleu_scores = [score / len(val_loader) for score in bleu_scores]  # 计算每个BLEU值的平均分
    print(f"BLEU scores: {avg_bleu_scores}")
    pbar.close()
    return val_loss, val_accuracy, bleu_total, avg_bleu_scores


if __name__ == "__main__":
    epoch = 8000
    batch_size = 128
    save_number = 100
    lr = 0.001
    lrf = 0.001
    input_size = (224, 224)
    # Define the optimizer and loss function
    image_folder = 'textmodel/images_training'
    label_folder = 'processed_data.csv'
    df = pd.read_csv(label_folder)
    ViT_weights = 'weight/vit_base_patch16_224_in21k.pth'
    dataframe = creat_tokenier(df)
    dataframe.to_csv('check.csv', index=False)
    train, test = splite_data(dataframe)

    text_processor = Tokenizer(train, test)
    train_dataset = MyDataset(train, input_size=(224, 224), text_processor=text_processor)
    # indices = list(range(len(train_dataset)))
    # selected_indices = indices[:len(indices) // 10]
    # train_dataset = torch.utils.data.Subset(train_dataset, selected_indices)
    #
    # for item in train_dataset:
    #     print(len(item))

    test_dataset = MyDataset(test, input_size=(224, 224), text_processor=text_processor)
    #if you train on the Servers can make num_workers = 24 if on the windows num_workers = 0
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True, num_workers=24)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, drop_last=False, num_workers=24)#True)
    if not os.path.exists('weight'):
        os.makedirs('weight')


    # model = MyModel(image_size=224, max_pad=32, embedding_dim=256, dense_dim=512, tokenizer=text_processor, device=device, vit_weight=ViT_weights)
    model = MyModel(image_size=224, max_pad=text_processor.max_pad, embedding_dim=256, dense_dim=512, tokenizer=text_processor, device=device, vit_weight=ViT_weights)

    model = model.to(device)
    

    #optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.8, weight_decay=5E-5)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=5E-5)


    # Initialize a TensorBoard writer
    writer = SummaryWriter()
    train_losses = []
    train_accuracies = []
    val_losses = []
    val_accuracies = []
    start_epoch = 1
    lf = lambda x: ((1 + math.cos(x * math.pi / epoch)) / 2) * (1 - lrf) + lrf  # cosine annealing
    scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lf)
    checkpoint_path = ''
    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        #print('start_epoch :',start_epoch)
    for epoch in range(start_epoch , epoch+1):
        train_loss, train_accuracy = train_process(model, device, train_loader, optimizer, epoch, text_processor)
        scheduler.step()  # Update the learning rate at the end of each training epoch
        val_loss, val_accuracy, bleu,avg_bleu_scores = validate(model, device, test_loader, epoch, text_processor)
        if epoch % save_number ==0:
            torch.save({'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict()
            }, f'./runs/mutimodel_{epoch}.pth')
        train_losses.append(train_loss)
        train_accuracies.append(train_accuracy)

        val_losses.append(val_loss)
        val_accuracies.append(val_accuracy)

        # plot_and_save_metrics(train_losses, train_accuracies, val_losses, val_accuracies, f"metrics_1{datetime.now()}.png")
        plot_and_save_metrics(train_losses, train_accuracies, val_losses, val_accuracies, f"metrics_baseline.png")

        # Print results for this epoch
        writer.add_scalar('Loss/Train', train_loss, epoch)
        writer.add_scalar('Accuracy/Train', train_accuracy, epoch)
        writer.add_scalar('Loss/Validation', val_loss, epoch)
        writer.add_scalar('Accuracy/Validation', val_accuracy, epoch)
        writer.add_scalar('Bleu/Validation', bleu, epoch)

        writer.flush()
        print(f'Epoch: {epoch}, Train Loss: {train_loss}, Train Accuracy: {train_accuracy}, Val Loss: {val_loss}, Val Accuracy: {val_accuracy}')
    writer.close()
