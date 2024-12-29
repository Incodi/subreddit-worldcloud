import praw
import spacy
import numpy as np
from PIL import Image
from wordcloud import WordCloud
import matplotlib.pyplot as plt


reddit = praw.Reddit(
    client_id="",
    client_secret="",
    user_agent="",
)

nlp = spacy.load("en_core_web_sm")

def clean_text(input):
    doc = nlp(input)
    
    output = []
        
    for token in doc:
        if not token.is_stop and token.is_alpha: # Ignore stopwords and non-alphabetic characters
            output.append(token.lemma_.lower())
    
    return ' '.join(output)

data = []

for submission in reddit.subreddit("subreddit name").top(limit=None): # 1000 posts
    # print(submission.title)
    data.append(submission.title)
    #submission.comments.replace_more(limit=0)
    #for top_level_comment in submission.comments:
        #data.append(top_level_comment.body)

text = str(data)
text_cleaned = clean_text(text)

f = open("text.txt", "w")
f.write(text_cleaned)
f.close()

# print(text_cleaned)

my_mask = np.array(Image.open('./mask.png')) # Insert file path to image for the mask

wc = WordCloud(background_color="white", mask = my_mask, colormap = "plasma", font_path = "Insert file path to font file" , max_words=99999, max_font_size=400, random_state=42)
wc.generate(text_cleaned)

plt.imshow(wc.recolor(color_func=None),vmin=1000, interpolation="bilinear")
plt.figure(figsize=(600,400))
wc.to_file('wordcloud.png')

