import torch
import cv2
import numpy as np
from nltk.translate.bleu_score import sentence_bleu

from new_model import MyModel,get_bleu
from utils import creat_tokenier,splite_data,Tokenizer,plot_and_save_metrics
import pandas as pd

def greedy_search_predict(image1, image2, model, tokenizer, device, input_size=(224,224)):
    image1 = cv2.imread(image1, cv2.IMREAD_UNCHANGED) / 255.0
    image2 = cv2.imread(image2, cv2.IMREAD_UNCHANGED) / 255.0
    image1 = cv2.resize(image1, input_size, interpolation=cv2.INTER_NEAREST)
    image2 = cv2.resize(image2, input_size, interpolation=cv2.INTER_NEAREST)
    image1 = torch.from_numpy(image1).permute(2, 0, 1).float().unsqueeze(0).to(device)
    image2 = torch.from_numpy(image2).permute(2, 0, 1).float().unsqueeze(0).to(device)

    encoder_output = model.image_encoder.encoder(image1, image2, 0.0)
    encoder_output = model.adaptive_avg_pool(encoder_output.transpose(1, 2)).transpose(1, 2)
    # encoder_output = model.sg(encoder_output)

    decoder_h, decoder_c = torch.zeros(encoder_output.size(0), encoder_output.size(2)).to(encoder_output.device), \
                           torch.zeros(encoder_output.size(0), encoder_output.size(2)).to(encoder_output.device)

    # decoder_h, decoder_c = torch.zeros_like(enc_op[:, 0]), torch.zeros_like(enc_op[:, 0])
    a = []
    for i in range(max_pad):
        if i == 0:
            # normal
            caption = torch.tensor(tokenizer.vocab['begin']).unsqueeze(0).to(device)
        else:
            caption = caption.to(device).squeeze(0)
        output, decoder_h, attention_weights = model.decoder.one_step_decoder(caption, encoder_output, decoder_h)

        # prediction
        max_prob = torch.argmax(output, dim=-1)  # shape: (1, 1)
        caption = max_prob #tokenizer.glove.get_vecs_by_tokens(tokenizer.itos[int(max_prob)], True)   # will be sent to onstepdecoder for the next iteration
        if max_prob.item() == tokenizer.vocab['end']:
            break
        else:
            a.append(max_prob.item())
    print(a)
    # Convert indices back to text
    decoded_text = [tokenizer.itos[item] for item in a]
    return decoded_text
def extract_text_from_csv(image1_path, image2_path, csv_path):

    df = pd.read_csv(csv_path)
    image1_name = image1_path
    image2_name = image2_path
    row = df[(df['image_1'] == image1_name) | (df['image_2'] == image2_name)]
    return row['impression_x'].values[0]
def split_text_into_lines(text, max_length):
    words = text.split()
    lines = []
    current_line = []
    current_length = 0

    for word in words:
        if current_length + len(word) > max_length:
            lines.append(' '.join(current_line))
            current_line = []
            current_length = 0
        current_line.append(word)
        current_length += len(word) + 1

    lines.append(' '.join(current_line))
    return lines
def wrap_text(text, width):
    """Wrap text to fit within the specified width."""
    import textwrap
    return textwrap.wrap(text, width)

def add_centered_text_to_image(img, text, start_y_position, font=cv2.FONT_HERSHEY_SIMPLEX, font_scale=0.7,
                               font_thickness=2, color=(0, 0, 0), line_spacing=20):

    # Ensure text is a string
    if not isinstance(text, str):
        text = str(text)

    # Wrap text
    lines = wrap_text(text, 80)  # Here, 40 is the approximate number of characters per line. You can adjust as needed.

    y = start_y_position
    for line in lines:
        text_size = cv2.getTextSize(line, font, font_scale, font_thickness)[0]
        text_x = (img.shape[1] - text_size[0]) // 2
        cv2.putText(img, line, (text_x, y), font, font_scale, color, font_thickness, lineType=cv2.LINE_AA)
        y += line_spacing

    return img
def get_bleu(reference,prediction):
  """
  Given a reference and prediction string, outputs the 1-gram,2-gram,3-gram and 4-gram bleu scores
  """
  reference = [reference.split()] #should be in an array (cos of multiple references can be there here only 1)
  prediction = prediction.split()
  bleu1 = round(sentence_bleu(reference, prediction, weights=(1, 0, 0, 0)), 8)
  bleu2 = round(sentence_bleu(reference, prediction, weights=(0.5, 0.5, 0, 0)), 8)
  bleu3 = round(sentence_bleu(reference, prediction, weights=(0.33, 0.33, 0.33, 0)), 8)
  bleu4 = round(sentence_bleu(reference, prediction, weights=(0.25, 0.25, 0.25, 0.25)))

  return bleu1,bleu2,bleu3,bleu4
if __name__ == "__main__":
    input_size = (224, 224)
    label_folder = 'processed_data.csv'
    df = pd.read_csv(label_folder)
    dataframe = creat_tokenier(df)
    train, test = splite_data(dataframe)

    tokenizer = Tokenizer(train, test)
    max_pad = 300
    #-----there is no evidence of acute cardiopulmonary disease -------
    #image1 = r'textmodel/images_training\CXR3517_IM-1716-2001.png'
    #image2 = r'textmodel/images_training\CXR3517_IM-1716-2001.png'
    #-------------------------------------------------------------------
    #------negative for acute abnormality .-----------------------------
    image1 = r'textmodel/images_training\CXR1153_IM-0104-1001.png'
    image2 = r'textmodel/images_training\CXR1153_IM-0104-2001.png'
    # -------------------------------------------------------------------
    #-------no acute pulmonary disease--------------
    #image1 = r'textmodel/images_training\CXR962_IM-2453-1002001.png'
    #image2 = r'textmodel/images_training\CXR962_IM-2453-1003002.png'

    #mild cardiomegaly .  no overt edema .  lateral image is degraded by
    # motion but there suggestion of minimal bibasilar airspace disease atelectasis .  no appreciable pleural effusion or pneumothorax .
    #Low lung volume with mild cardiac enlargement, no acute cardiopulmonary abnormalities
    image1 = r'textmodel/images_training\CXR2557_IM-1061-4004.png'
    image2 = r'textmodel/images_training\CXR2557_IM-1061-4004.png'
    #------Best
    #image1 = r'textmodel/images_training\CXR2925_IM-1327-1001.png'
    #image2 = r'textmodel/images_training\CXR2925_IM-1327-2001.png'

    #image1 = r'textmodel/images_training\CXR702_IM-2267-1001.png'
    #image2 = r'textmodel/images_training\CXR702_IM-2267-1001.png'
    img1 = cv2.imread(image1)
    img2 = cv2.imread(image2)
    pretrained_weight = r'weight/mutimodel_final.pth'
    label_folder = 'check.csv'
    csv_text = extract_text_from_csv(image1, image2, label_folder)
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    ViT_weights = r'weight/model-vitfinal.pth'

    model = MyModel(image_size=224, max_pad=tokenizer.max_pad, embedding_dim=256, dense_dim=512, tokenizer=tokenizer, device=device, vit_weight=ViT_weights)
    model.load_state_dict(torch.load(pretrained_weight, map_location='cpu')['model_state_dict'])
    model.to(device)
    model.eval()

    out = greedy_search_predict(image1, image2, model, tokenizer, device)
    # Convert the list of words to a single string
    formatted_output = ' '.join(out)
    print(formatted_output)

    if img1.shape[0] != img2.shape[0]:
        img2 = cv2.resize(img2, (img2.shape[1], img1.shape[0]))

    # Horizontally stack two images
    stacked_images = cv2.hconcat([img1, img2])

    # Compute the width for padding on each side
    target_width = int(stacked_images.shape[1] * 3)
    padding_width = (target_width - stacked_images.shape[1]) // 2

    # Create padding
    height = stacked_images.shape[0]
    padding = 255 * np.ones(shape=[height, padding_width, 3], dtype=np.uint8)

    # Horizontally stack padding, images, and padding again
    stacked_images = cv2.hconcat([padding, stacked_images, padding])

    # Create a blank area for text
    height, width, _ = stacked_images.shape
    blank_area = 255 * np.ones(shape=[int(height * 1.5), width, 3], dtype=np.uint8)

    y_position1 = int(blank_area.shape[0] * 0.4)
    text_img = add_centered_text_to_image(blank_area, 'Predict: '+formatted_output, y_position1)

    lines = split_text_into_lines(csv_text, max_length=5)  # max_length\

    y_position2 = int(blank_area.shape[0] * 0.8)
    text_img = add_centered_text_to_image(text_img, 'Orginal: '+str(csv_text), y_position2)

    # Vertically stack images with the text area
    final_image = cv2.vconcat([stacked_images, text_img])
    bleu1=[]
    bleu2=[]
    bleu3=[]
    bleu4=[]
    # Save the final image
    cv2.imwrite('final_image_with_text.png', final_image)
    bleu1,bleu2,bleu3,bleu4=get_bleu(csv_text,formatted_output)
    print( bleu1,bleu2,bleu3,bleu4)
    # Display the image
    cv2.imshow("Result", final_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()