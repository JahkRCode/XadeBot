'''
## Add file to git
git add FILE_TO_ADD
## Commit change with descriptive message
git commit -m "DESCRIPTION FOR COMMIT HERE!"
## Push commit to master branch
git push origin master
'''
import re
import tensorflow as tf
import numpy as np
import time

questions = open('questions.txt', encoding = 'utf-8', errors = 'ignore').read().split('\n')
answers = open('answers.txt', encoding = 'utf-8', errors = 'ignore').read().split('\n')

## Clean text of unwanted character combinations
def clean_text(text):
    text = text.lower()
    text = re.sub(r"i'm", "i am", text)
    text = re.sub(r"he's", "he is", text)
    text = re.sub(r"at's", "at is", text)
    text = re.sub(r"re's", "re is", text)
    text = re.sub(r"won't", "will not", text)
    text = re.sub(r"\'ll", " will", text)
    text = re.sub(r"\'ve", " have", text)
    text = re.sub(r"\'re", " are", text)
    text = re.sub(r"n't", " not", text)
    text = re.sub(r"can't", "cannot", text)
    text = re.sub(r"\'", " would", text)
    text = re.sub(r"newlinechar", "", text)
    text = re.sub(r"[-()\"#/@;:<>{}+=~|.?,]", "", text)
    return text

## Clean up the questions file and store results into list
clean_questions = []
for question in questions:
    clean_questions.append(clean_text(question))
    
## Clean up the answers file and store results into list
clean_answers = []
for answer in answers:
    clean_answers.append(clean_text(answer))