# import packages
import re
import os
import math

#initialize variables
timestamp_dict = {}
transcripts = []
wiki = []

# helper function for writing python lists to file
# use cases - to write all transcripts to one file and in the end to write all typos and their timestamps to files
def write_list_to_file(results, file_name):
    with open(file_name, 'w') as f:
        for item in results:
            f.write(item)
            f.write('\n')

# helper function for writing list of tuples to file
# use case - to write bigram keys to file
def write_list_of_tuples_to_file(results, file_name):
    with open(file_name, 'w') as f:
        for item in results:
            line = ' '.join(str(x) for x in item) # stores tuple items as separated by space
            f.write(line)
            f.write('\n')

# helper function for writing pythond dictionaries to file
def write_dict_to_file(results, file_name):
    with open(file_name, 'w') as f:
        for key, value in results.items():
            f.write('%s:%s\n' % (key, value))

# helper function for writing python dictionaries containing tuples as key to file
# use cases - to write bigram frequency or probability, for bigrams the key will be a tuple
def write_tuple_dict_to_file(results, file_name):
    with open(file_name, 'w') as f:
        for key, value in results.items():
            line = ' '.join(str(x) for x in key) # stores tuple items as separated by space
            f.write('%s:%s\n' % (line, value))


# helper founction for making the transcript text all lower case and without punctuation, also removes extra spaces
def unigram_text_formatter(text):
    new_line = text.lower()
    new_line = re.sub(r"'s\b","",new_line)
    new_line = re.sub("[^a-zA-Z]", " ", new_line)
    new_line = re.sub("\s+", "", new_line) # remove extra spaces
    return new_line

# opens each transcript document and converts it to a list then appends the entire transcript as one item to the transcripts list
# also adds all of the words from each transcript to a list for consumption by the unigram model
def read_transcript(files, dirname):
    curr_file = files
    unigram_text_data = []
    documents_path = dirname + '/' + files
    with open (documents_path, 'r') as doc:
        doc_list = []
        lines = doc.readlines()
        for index, line in enumerate(lines):
            # look to see if the line is only a number in which case we want to skip it entirely
            match = re.search(r'^\d+\s',line)
            if  not match:
                # if it's a timestamp line, add the timestamp as the dictionary key and add the next 2 lines of text as the dictionary value
                # try/except is to handle when it's the last line of the transcript
                if line.startswith('00:'):
                    try:
                        timestamp_dict[curr_file + ' : ' + line.strip()] = lines[index+1].strip() + ' ' + lines[index+2].strip()
                    except IndexError:
                        timestamp_dict[curr_file + ' : ' + line.strip()] = lines[index+1].strip()
                # if it's not a blank line, add the line to the transcript's list
                # removing the meaningless words that are often at beginning/end of transcripts [SOUND] etc.
                elif not line.startswith('\n'):
                    doc_list.append(line.strip().replace('[SOUND]','').replace('[MUSIC]','').replace('[NOISE]',''))
                    clean_text = unigram_text_formatter(line.strip().replace('[SOUND]','').replace('[MUSIC]','').replace('[NOISE]',''))
                    unigram_text_data.append(clean_text.split(' '))

        # after iterating through all of the lines in a transcript, append it to the main transcripts list     
        string = ' '
        string_list = string.join(doc_list)
        transcripts.append(string_list.strip())

        return unigram_text_data

# open wiki pages and covert to a list that can be used for background unigram model
def read_wiki(files, dirname):
    curr_file = files
    unigram_text_data = []
    documents_path = dirname + '/' + files
    with open (documents_path, 'r', encoding="utf8") as doc:
        doc_list = []
        for line in doc.readlines():
            clean_text = unigram_text_formatter(line.strip())
            unigram_text_data.append(clean_text.split(' '))
            
    return unigram_text_data

# iterates through all of the transcript files at the specified directory
def read_files(dirname, ftype, ext):
    for files in os.listdir(dirname):
        if files.endswith(ext) and ftype == 'transcript':
            transcript_text_data = read_transcript(files, dirname)
        elif files.endswith(ext) and ftype == 'wiki':
            wiki_text_data = read_wiki(files, dirname)

# test finding timestamp for typo
def get_timestamp(val):
    dict_entry = {key: value for key, value in timestamp_dict.items() if val in value}
    return dict_entry

UNK = None

# create class with functions that build unigram language model
class UnigramLanguageModel:
    def __init__(self, text_data, smoothing=False):
        self.unigram_frequencies = {}
        self.corpus_length = 0
        # iterate through the words in all transcripts and update each word's count 
        for line in text_data:
            for word in line:
                self.unigram_frequencies[word] = self.unigram_frequencies.get(word, 0) + 1
                self.corpus_length += 1
        self.unique_words = len(self.unigram_frequencies)
        #exact use of smoothing TBD
        self.smoothing = smoothing

    # find the probability of each word as the count of the word / total words in the corpus
    def calculate_unigram_probability(self, word):
        word_prob_num = self.unigram_frequencies.get(word,0)
        word_prob_den = self.corpus_length
        if self.smoothing:
                word_prob_num += 1
                # add one more to total number of seen unique words for UNK - unseen events
                word_prob_den += self.unique_words + 1
        return float(word_prob_num) / float(word_prob_den)

    # returns an alphabetically sorted list of all unique words in the vocabulary
    def sorted_vocabulary(self):
        full_vocab = list(self.unigram_frequencies.keys())
        full_vocab.sort()
        return full_vocab

# function to store the probabilities for each word in a dictionary
def store_unigram_probs(sorted_vocab_keys, model):
    unigram_prob_dict = {}
    for vocab_key in sorted_vocab_keys:
        unigram_prob_dict[vocab_key] = model.calculate_unigram_probability(vocab_key)
    return unigram_prob_dict

# function to create a mixture of 2 unigram models
# using basic linear interpolation between the probabilities for each model
def unigram_mixture_probs(transcript_prob_dict, wiki_prob_dict, lam = 0):
    mixture_prob_dict = {}
    for vocab_key, value in transcript_prob_dict.items():
        mixture_prob_dict[vocab_key] = ((1 - lam) * value) + (lam * wiki_prob_dict.get(vocab_key, 0))
        
    return mixture_prob_dict

# class for building bigram language model
class BigramLanguageModel(UnigramLanguageModel):
    def __init__(self, text_data, smoothing=False):
        UnigramLanguageModel.__init__(self, text_data, smoothing)
        self.bigram_frequencies = {}
        self.unique_bigrams = set()
        for sentence in text_data:
            previous_word = None
            for word in sentence:
                if previous_word != None:
                    self.bigram_frequencies[(previous_word, word)] = self.bigram_frequencies.get((previous_word, word), 0) + 1
                    self.unique_bigrams.add((previous_word, word))                                                     
                previous_word = word
        self.unique__bigram_words = len(self.unigram_frequencies)
        
    def calculate_bigram_probability(self, previous_word, word):
        bigram_word_prob_num = self.bigram_frequencies.get((previous_word, word), 0)
        bigram_word_prob_den = self.unigram_frequencies.get(previous_word, 0)
        if self.smoothing:
            bigram_word_prob_num += 1
            bigram_word_prob_den += self.unique__bigram_words
        return 0.0 if bigram_word_prob_num == 0 or bigram_word_prob_den == 0 else float(
            bigram_word_prob_num) / float(bigram_word_prob_den)

    # returns an alphabetically sorted list of all unique tuples in the vocabulary
    def sorted_vocabulary(self):
        full_vocab = list(self.bigram_frequencies.keys())
        full_vocab.sort()
        return full_vocab

# function to store the probabilities for each bigram in a dictionary
def store_bigram_probs(sorted_vocab_keys, model):
    bigram_prob_dict = {}
    for (previous_word, word) in sorted_vocab_keys:
        bigram_prob_dict[(previous_word, word)] = model.calculate_bigram_probability(previous_word, word)
    return bigram_prob_dict

# function to create a mixture of 2 bigram models
# using basic linear interpolation between the probabilities for each model
def bigram_mixture_probs(transcript_prob_dict, wiki_prob_dict, lam = 0):
    bigram_mixture_prob_dict = {}
    for vocab_key, value in transcript_prob_dict.items():
        bigram_mixture_prob_dict[vocab_key] = ((1 - lam) * value) + (lam * wiki_prob_dict.get(vocab_key, 0))
        
    return bigram_mixture_prob_dict

# read the transcript files and wiki files
read_files(dirname='transcripts_sample', ftype='transcript', ext ='.srt') #update to your own file path
read_files(dirname='wiki_sample', ftype='wiki', ext='.txt') #update to your own file path

# build the unigram models for transcripts and wiki
transcript_model = UnigramLanguageModel(transcripts)
wiki_model = UnigramLanguageModel(wiki)
transcript_sorted_vocab_keys = transcript_model.sorted_vocabulary()
wiki_sorted_vocab_keys = wiki_model.sorted_vocabulary()

# build the bigram models for transcripts and wiki
transcript_bigram_model = BigramLanguageModel(transcript_text_data)
wiki_bigram_model = BigramLanguageModel(wiki_text_data)
transcript_bigram_sorted_vocab_keys = transcript_bigram_model.sorted_vocabulary()
wiki_bigram_sorted_vocab_keys = wiki_bigram_model.sorted_vocabulary()

# calculate the probabilities
transcript_prob_dict = store_unigram_probs(transcript_sorted_vocab_keys, transcript_model)
wiki_prob_dict = store_unigram_probs(wiki_sorted_vocab_keys, wiki_model)
mix_prob_dict = unigram_mixture_probs(transcript_prob_dict, wiki_prob_dict, lam=.1)

# calculate the bigram probabilities
transcript_bigram_prob_dict = store_bigram_probs(transcript_bigram_sorted_vocab_keys, transcript_bigram_model)
wiki_bigram_prob_dict = store_bigram_probs(wiki_bigram_sorted_vocab_keys, wiki_bigram_model)
mix_bigram_prob_dict = bigram_mixture_probs(transcript_bigram_prob_dict, wiki_bigram_prob_dict, lam=.1)

# write things to file, mainly for inspection
write_list_to_file(transcript_sorted_vocab_keys, 'transcript_vocab_keys.txt')
write_list_to_file(wiki_sorted_vocab_keys, 'wiki_vocab_keys.txt')
write_dict_to_file(transcript_prob_dict, 'transcript_frequencies.txt')
write_dict_to_file(wiki_prob_dict, 'wiki_frequencies.txt')
write_dict_to_file(mix_prob_dict, 'mixt_frequencies.txt')
write_list_to_file(transcripts, 'transcript_master_file.txt')

# write bigram data to file
write_list_of_tuples_to_file(transcript_bigram_sorted_vocab_keys, 'transcript_bigram_vocab_keys.txt')
write_list_of_tuples_to_file(wiki_bigram_sorted_vocab_keys, 'wiki_bigram_vocab_keys.txt')
write_tuple_dict_to_file(transcript_bigram_prob_dict, 'transcript_bigram_frequencies.txt')
write_tuple_dict_to_file(wiki_bigram_prob_dict, 'wiki_bigram_frequencies.txt')
write_tuple_dict_to_file(mix_bigram_prob_dict, 'mixt_bigram_frequencies.txt')

#print(get_timestamp('classroom'))
