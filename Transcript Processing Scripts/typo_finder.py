import re

#Helper function to load ngram models from text file to dictionary
def load_prob_dict(file):
    dictionary = {}
    with open(file) as f:
        for line in f:
            (key, val) = line.strip().split(':')
            dictionary[key] = float(val)
    return dictionary

#Function to load full text into program and format (lower case + sentence breaks)
def load_corpus(file):
    text_file = open(file, "r")
    text_data = text_file.read()
    text_data = text_data.lower()
    text_data = re.sub(r"'s\b","", text_data)
    text_data = re.sub("[^a-zA-Z \n]", "", text_data)
    # print(text_data)
    # text_list = re.findall(r"[\w']+|[.,!?;]", text_data)
    text_file.close()
    return text_data

#Class to find typos in a corpus of text
class TypoFinder:
    def __init__(self, unigram_file, bigram_file, threshold, uni_weight, bi_weight ):
        self.unigram_LM = load_prob_dict(unigram_file)
        self.bigram_LM = load_prob_dict(bigram_file)
        self.threshold = threshold
        self.uni_weight = uni_weight
        self.bi_weight = bi_weight

    #Function to score words to deteremine if they could potentially be a typo
    def typo_flag(self, word, beg_combo, end_combo, threshold, uni_weight, bi_weight):
        before_bi_score = 0
        after_bi_score = 0
        bigrams_found = 0
        uni_score = 0
        bi_score = 0

        #Look in dictionary only if 'before' bigram word combo exists
        if beg_combo != '' and beg_combo in self.bigram_LM:
            before_bi_score = self.bigram_LM[beg_combo]
            bigrams_found += 1
        #Look in dictionary only if 'after' bigram word combo exists
        if end_combo != '' and end_combo in self.bigram_LM:
            after_bi_score = self.bigram_LM[end_combo]
            bigrams_found += 1
        #Look in dictionary only if word exists in unigram LM
        if word != '' and word in self.unigram_LM:
            uni_score = self.unigram_LM[word] * uni_weight
        #Only calculate bi_score if at least one bigram found (div by  0)
        if bigrams_found != 0:
            bi_score = ((before_bi_score + after_bi_score) /  bigrams_found) * bi_weight
        #Add both probabilities together to get uni/bi mixed score
        mix_score = uni_score + bi_score
        #Return the word if under the desired threshold.  Otherwise, return null
        if mix_score <= self.threshold:
            return word, mix_score
        else:
            return None, None

    #Function to loop through text and return a list of all words that are potential typos ('corpus' needs to be a list)
    def typo_finder(self, corpus):
        uni_word = ''
        before_combo = ''
        after_combo = ''
        typo_list = []
        #Loop through corpus list and pull out each word and word context (left/right of word) and run through 'typo_flag'
        for sentence in corpus.split('\n'):
            for index, word in enumerate(re.findall(r"[\w']+|[.,!?;]", sentence)):
                #Find appropriate unigram words (not punctuation)
                if word in ".,!?;":
                    continue
                else:
                    uni_word = word
                #Get bigram combo for word and word before it
                if index != 0 and corpus[index-1] not in ".,!?;":
                    before_combo = corpus[index-1] + " " + word
                else:
                    before_combo = ''
                #Get bigram combo for word and word after it
                if index != len(corpus)-1 and corpus[index+1] not in ".,!?;":
                    after_combo = word + " "  + corpus[index+1]
                else:
                    after_combo = ''
                #Run the word and bigram word combos through the 'typo_flag' to get mixed probability score and threshold comparison
                typo, probability = self.typo_flag(uni_word, before_combo, after_combo, self.threshold, self.uni_weight, self.bi_weight)
                #If 'typo_flag' returned None, not a typo.  Otherwise, add to typo_list
                if typo != None:
                    typo_list.append([index, word, probability, sentence])
        return typo_list
    
'''
#Testing functionality of TypoFinder

#File names + directory for text (corpus, unigram LM, bigram LM)
transcript_file = 'C:\\Users\\scott\\Documents\\School\\CS410\\CourseProject\\Transcript Processing Scripts\\transcript_master_file.txt'
unigram_LM_file = 'C:\\Users\\scott\\Documents\\School\\CS410\\CourseProject\\Transcript Processing Scripts\\mixt_frequencies.txt'
bigram_LM_file = 'C:\\Users\\scott\\Documents\\School\\CS410\\CourseProject\\Transcript Processing Scripts\\fake_bigram.txt'
#Load string text (transcript) into list
transcripts =  load_corpus(transcript_file)
#Instantiate TypoFinder
finder = TypoFinder(unigram_LM_file, bigram_LM_file, .00005, .05, .05)
#List of potential typos
typo_list = finder.typo_finder(transcripts)
#Print 'typos' found
for typo in typo_list:
    print(typo)
'''
