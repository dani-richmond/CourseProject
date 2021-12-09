# -*- coding: utf-8 -*-
import nltk
import io

# downaload and import the necessary components of NLTK library
nltk.download('punkt')

# open extracted textbook text file and remove the line feed
with io.open('Content_Text Data Management and Analysis.txt', "r", encoding="utf-8") as my_file:
    text = my_file.read().replace('\n', ' ')
my_file.close()

# process the text with sent_tokenize() function in nltk library to generate a list of sentence
a_list = nltk.tokenize.sent_tokenize(text)

# write the processed data to Textbook Content.txt file with as line separated sentence
f = io.open("Textbook Content.txt", "w",encoding="utf-8")
for x in a_list:
  f.write(x + "\n")
f.close()