# import packages
import re
import os
import math

#initialize variables
dirname ='C:\\...\\transcripts_srt\\textanalytics_srt' #update to file directory where transcripts are on your local
ext = '.srt'
timestamp_dict = {}
transcripts = []
unigram_text_data = []
unigram_prob_dict = {}

# helper function for writing python lists to file
# use cases - to write all transcripts to one file and in the end to write all typos and their timestamps to files
def write_list_to_file(results, file_name):
    with open(file_name, 'w') as f:
        for item in results:
            f.write(item)
            f.write('\n')

# helper function for writing pythond dictionaries to file
def write_dict_to_file(results, file_name):
    with open(file_name, 'w') as f:
        for key, value in results.items():
            f.write('%s:%s\n' % (key, value))
            #f.write('\n')

# helper founction for making the transcript text all lower case and without punctuation
def unigram_text_formatter(text):
    new_line = text.lower()
    new_line = re.sub(r"'s\b","",new_line)
    new_line = re.sub("[^a-zA-Z]", " ", new_line)
    return new_line

# opens each transcript document and converts it to a list then appends the entire transcript as one item to the transcripts list
# also adds all of the words from each transcript to a list for consumption by the unigram model
def read_transcript(files):
    curr_file = files
    documents_path = dirname + '\\' + files
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


# iterates through all of the transcript files at the specified directory
def read_files():
    for files in os.listdir(dirname):
        if files.endswith(ext):
            read_transcript(files)

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
    for vocab_key in sorted_vocab_keys:
        unigram_prob_dict[vocab_key] = model.calculate_unigram_probability(vocab_key)

# read the transcript files and built the unigram model
read_files()
transcript_model = UnigramLanguageModel(unigram_text_data)
sorted_vocab_keys = transcript_model.sorted_vocabulary()
write_list_to_file(sorted_vocab_keys, 'vocab_keys.txt')
store_unigram_probs(sorted_vocab_keys, transcript_model)
write_dict_to_file(unigram_prob_dict, 'frequencies.txt')
write_list_to_file(transcripts, 'transcript_master_file.txt')


#print(get_timestamp('classroom'))
