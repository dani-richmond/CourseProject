# -*- coding: utf-8 -*-
import nltk
import io

with io.open('Input.txt', "r", encoding="utf-8") as my_file:
    text = my_file.read().replace('\n', ' ')

a_list = nltk.tokenize.sent_tokenize(text)

f = io.open("Output.txt", "w",encoding="utf-8")
for x in a_list:
  f.write(x + "\n")
f.close()