#In this EDA process should using some kears tokenier follow the initial author.
import pandas as pd
import numpy as np
import os
import cv2
import time
#from google.colab.patches import cv2_imshow
import re
import matplotlib.pyplot as plt
from bs4 import BeautifulSoup  # for reading xml file
import matplotlib.pyplot as plt
from wordcloud import WordCloud, STOPWORDS  # for plotting wordcloud
from tqdm import tqdm

image_folder = 'textmodel/images'  # path to folder containing images
total_images = len(os.listdir(image_folder))
print('The number of images in data are: %i' % (total_images))
np.random.seed(420)
# for i in range(3):  # print 5 sample images
#     k = np.random.choice(range(total_images))
#     image_file = os.listdir(image_folder)[k]
#     image = cv2.imread(os.path.join(image_folder, image_file))  # getting an image file
#     print("%i)\n" % (i + 1))
#     cv2_imshow(image)
#     print("\t\t", image_file)  # the image file name


# for i in range(3):  # print 3 sample images
#     k = np.random.choice(range(total_images))
#     image_file = os.listdir(image_folder)[k]
#     image = cv2.imread(os.path.join(image_folder, image_file))  # getting an image file
#     image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # convert color from BGR to RGB
#     print("%i)\n" % (i + 1))
#     plt.imshow(image)
#     plt.show()
#     print("\t\t", image_file)  # the image file name

reports_folder = "textmodel/reports"#report folder
total_reports = len(os.listdir(reports_folder))
print('The number of reports in the data are: %i' % (total_reports))

no_images = []  # stores the no. of images
for file in os.listdir(reports_folder):
    report_file = os.path.join(reports_folder, file)
    with open(report_file, 'r') as f:  # reading the xml data
        data = f.read()
    regex = r"parentImage id.*"  # getting all the image names
    k = re.findall(regex, data)
    temp = len(k)
    no_images.append(temp)


no_images = np.array(no_images)
print("The max no. of images found associated with a report: %i" % (no_images.max()))
print("The min no. of images found associated with a report: %i" % (no_images.min()))

# In[54]:


plt.figure(figsize=(6, 4))
ax = pd.Series(no_images).plot(kind='hist')
ax.set_xlabel('No. of images associated with report')
ax.set_title("Frequency vs No. of images associated with report")
plt.show()
print("Image Value_counts\n")
print(pd.Series(no_images).value_counts())


# We can see that the maximum number of images associated with a report can be 5 while the minimum is 0. We will extract all the information part ie comparison, indication, findings and impression part of the report and the corresponding 2 images (we will take 2 images as input as 2 images are highest frequecy of being associated with a report) of the concerned report to a dataframe with xml report file name. For images more than 2, we will creat new datapoint with new image and same info.

# In[70]:


# function for obtaining the different information part of the xml report file and preprocessing them and also adding the concernced image and report information to the dataframe
#Replacement of text abbreviations
def decontracted(phrase):  # https://stackoverflow.com/a/47091490
    """
    performs text decontraction of words like won't to will not
    """
    # specific
    phrase = re.sub(r"won\'t", "will not", phrase)
    phrase = re.sub(r"can\'t", "can not", phrase)

    # general
    phrase = re.sub(r"n\'t", " not", phrase)
    phrase = re.sub(r"\'re", " are", phrase)
    phrase = re.sub(r"\'s", " is", phrase)
    phrase = re.sub(r"\'d", " would", phrase)
    phrase = re.sub(r"\'ll", " will", phrase)
    phrase = re.sub(r"\'t", " not", phrase)
    phrase = re.sub(r"\'ve", " have", phrase)
    phrase = re.sub(r"\'m", " am", phrase)
    return phrase

#get the information and message from XML file from the specify part.
def get_info(xml_data, info):  # https://regex101.com/
    """
    extracts the information data from the xml file and does text preprocessing on them
    here info can be 1 value in this list ["COMPARISON","INDICATION","FINDINGS","IMPRESSION"]
    """
    regex = r"\"" + info + r"\".*"
    k = re.findall(regex, xml_data)[0]  # finding info part of the report

    regex = r"\>.*\<"
    k = re.findall(regex, k)[0]  # removing info string and /AbstractText>'

    regex = r"\d."
    k = re.sub(regex, "", k)  # removing all values like "1." and "2." etc

    regex = r"X+"
    k = re.sub(regex, "", k)  # removing words like XXXX

    regex = r" \."
    k = re.sub(regex, "", k)  # removing singular fullstop ie " ."

    regex = r"[^.a-zA-Z]"
    k = re.sub(regex, " ", k)  # removing all special characters except for full stop

    regex = r"\."
    k = re.sub(regex, " .", k)  # adding space before fullstop
    k = decontracted(k)  # perform decontraction
    k = k.strip().lower()  # strips the begining and end of the string of spaces and converts all into lowercase
    k = " ".join(k.split())  # removes unwanted spaces
    if k == "":  # if the resulting sentence is an empty string return null value
        k = np.nan
    return k

#get all message from specific part.
def get_final(data):
    """
    given an xml data returns "COMPARISON","INDICATION","FINDINGS","IMPRESSION" part of the data
    """
    try:  # assigning null values to the ones that don't have the concerned info
        comparison = get_info(data, "COMPARISON")
    except:
        comparison = np.nan;

    try:  # assigning null values to the ones that don't have the concerned info
        indication = get_info(data, "INDICATION")
    except:
        indication = np.nan;

    try:  # assigning null values to the ones that don't have the concerned info
        finding = get_info(data, "FINDINGS")
    except:
        finding = np.nan;

    try:  # assigning null values to the ones that don't have the concerned info
        impression = get_info(data, "IMPRESSION")
    except:
        impression = np.nan;

    return comparison, indication, finding, impression


def get_df():
    """
    Given an xml data, it will extract the two image names and corresponding info text and returns a dataframe
    """
    im1 = []  # there are 2 images associated with a report
    im2 = []
    # stores info
    comparisons = []
    indications = []
    findings = []
    impressions = []
    report = []  # stores xml file name
    for file in tqdm(os.listdir(reports_folder)):
        report_file = os.path.join(reports_folder, file)
        with open(report_file, 'r') as f:  # reading the xml data
            data = f.read()

        regex = r"parentImage id.*"  # getting all the image names
        k = re.findall(regex, data)

        if len(k) == 2:
            regex = r"\".*\""  # getting the name
            image1 = re.findall(regex, k[0])[0]
            image2 = re.findall(regex, k[1])[0]

            image1 = re.sub(r"\"", "", image1)
            image2 = re.sub(r"\"", "", image2)

            image1 = image1.strip() + ".png"
            image2 = image2.strip() + ".png"
            im1.append(image1)
            im2.append(image2)

            comparison, indication, finding, impression = get_final(data)
            comparisons.append(comparison)
            indications.append(indication)
            findings.append(finding)
            impressions.append(impression)
            report.append(file)  # xml file name


        elif len(k) < 2:
            regex = r"\".*\""  # getting the name
            try:  # if the exception is raised means no image file name was found
                image1 = re.findall(regex, k[0])[0]
                image1 = re.sub(r"\"", "", image1)  # removing "
                image2 = np.nan

                image1 = image1.strip() + ".png"
            except:
                image1 = np.nan
                image2 = np.nan

            im1.append(image1)
            im2.append(image2)
            comparison, indication, finding, impression = get_final(data)
            comparisons.append(comparison)
            indications.append(indication)
            findings.append(finding)
            impressions.append(impression)
            report.append(file)  # xml file name

        else:  # if there are more than 2 images concerned with report
            comparison, indication, finding, impression = get_final(data)

            for i in range(len(k) - 1):
                regex = r"\".*\""  # getting the name
                image1 = re.findall(regex, k[i])[0]  # re.findall returns a list
                image2 = re.findall(regex, k[i + 1])[0]

                image1 = re.sub(r"\"", "", image1)  # removing "
                image2 = re.sub(r"\"", "", image2)  # removing "

                image1 = image1.strip() + ".png"
                image2 = image2.strip() + ".png"

                im1.append(image1)
                im2.append(image2)
                comparisons.append(comparison)
                indications.append(indication)
                findings.append(finding)
                impressions.append(impression)
                report.append(file)  # xml file name

    df = pd.DataFrame(
        {"image_1": im1, "image_2": im2, "comparison": comparisons, "indication": indications, "findings": findings,
         "impression": impressions, "xml file name": report})
    return df

##get the time
start = time.time()
df = get_df()
end = time.time()
print(f"Execution time: {end - start} seconds")


# df.to_pickle("/content/drive/My Drive/Medical image Reporting/df.pkl")
#df = pd.read_pickle("tokenizer.pkl")

print("columns\t\t%missing values")
print('-' * 30)
print(df.isnull().sum() * 100 / df.shape[0])  # percentage missing values

# We can see that there are many missing values in the data. We could remove all rows where image_1 and impression values are null since they represent below 5% of the total datapoints.


df.drop(df[(df['impression'].isnull()) | (df['image_1'].isnull())].index, inplace=True)
df = df.reset_index(drop=True).copy()
print("%i datapoints were removed.\nFinal no. of datapoints: %i" % (4169 - df.shape[0], df.shape[0]))

# We removed 3.2% datapoints in total.

# There are missing values in image_2. For that we can use the same image file in image_1. We will also check for image size for both of them.

#------------modify here--------------
df.loc[df.image_2.isnull(),'image_2'] = df[df.image_2.isnull()]['image_1'].values
im1_size = []
im2_size = []
for index,row in df.iterrows():
  im1_size.append( cv2.imread(os.path.join(image_folder,row.get('image_1'))).shape[:2])
  im2_size.append(cv2.imread(os.path.join(image_folder,row.get('image_2'))).shape[:2])

df['im1_height'] = [i[0] for i in im1_size]
df['im1_width'] = [i[1] for i in im1_size]
df['im2_height'] = [i[0] for i in im2_size]
df['im2_width'] = [i[1] for i in im2_size]

end = time.time()

print(f"Execution time: {end - start} seconds")
#------------modify here--------------


print(df.head(2))

# df.to_pickle("/content/drive/My Drive/Medical image Reporting/df_final.pkl")
#df = pd.read_pickle("/content/drive/My Drive/Medical image Reporting/df_final.pkl")
#df.shape

# In[84]:


ax = df[['im1_height', 'im2_height']].plot(kind='kde')
ax.set_title("Distribution of image heights")
ax.set_xlabel("Image height size")
plt.show()

print("\n\nValue Counts of image_1 heights:\n")
print(df.im1_height.value_counts()[:5])
print("\n", "*" * 50, "\n")
print("Value Counts of image_2 heights:\n")
print(df.im2_height.value_counts()[:5])

# We can observe that 420 is the most common height for image_1 while for image_2 is 624. The next common for both of the images is 512.


print("Value Counts of image_1 widths:\n")
print(df.im1_width.value_counts()[:5])
print("\n", "*" * 50, "\n")
print("Value Counts of image_2 widths:\n")
print(df.im2_width.value_counts()[:5])


# Width for both of the images have only 1 unique value for all datapoints and that is 512. Since pretrained models are modelled for square-sized images we can choose 512\*512 as the specified size of the image. Hence we can resize all images into 512\*512 shape.

# Now we will print some sample datapoints with corresponding images and captions associated with that datapoint.

# ## **Sample Images + Caption**

# In[86]:


def show_image_captions(df=df, image_folder=image_folder, sample=3):
    """
    given the df, samples datapoints and prints the images and caption
    df: dataframe
    image_folder: folder which contains images
    """
    k = df.sample(sample)
    i = 1
    for index, row in k.iterrows():
        image_1 = cv2.imread(os.path.join(image_folder, row.get('image_1')))
        image_2 = cv2.imread(os.path.join(image_folder, row.get('image_2')))

        plt.figure(figsize=(12, 8))  # setting the figure size
        plt.subplot(121)  # first x-ray
        plt.imshow(image_1, aspect='auto')

        plt.subplot(122)  # 2nd x-ray
        plt.imshow(image_2, aspect='auto')
        print("%i)\n" % (i))
        i += 1
        plt.show()  # printing the image
        print("\n", "Comparison: ", row.get('comparison'))
        print("\n", "Indication: ", row.get('indication'))
        print("\n", "Findings: ", row.get('findings'))
        print("\n", "Impression: ", row.get('impression'), "\n\n", "*" * 150, "\n\n")


# showing sample 3 datapoints
show_image_captions()

# We can see that indications are patients medical complaints while findings are the obervations from X-ray while impression is inference obtained. In this case study, I will try to predict the impression part of the medical report given the two images.

# ### **Wordcloud of Impressions**

# getting wordclouds
# https://www.geeksforgeeks.org/generating-word-cloud-python/
temp = df.loc[:, 'impression'].str.replace(".",
                                           "").copy()  # removing all fullstops and storing the result in a temp variable
words = ""
for i in temp.values:
    k = i.split()
    words += " ".join(k) + " "
word = words.strip()
wc = WordCloud(width=1024, height=720,
               background_color='white',
               stopwords=STOPWORDS,
               min_font_size=15, ).generate(words)

del k, words, temp



plt.figure(figsize=(16, 16))
plt.imshow(wc)
plt.axis("off")
plt.show()

