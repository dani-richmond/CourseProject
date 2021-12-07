import final_unigram_bigram_script as ngmod
import typo_finder as tf
from os.path import exists

#Main function to set up unigram and bigram models, load transcripts and find typos
def main():

    #Input user directory and file name values, as well as typo finding weights, threshold, etc.
    transcript_dir = '../transcripts_srt'
    textbook_dir = '../Textbook'
    unigram_model = 'mixt_frequencies.txt'
    bigram_model = 'mixt_bigram_frequencies.txt'
    transcript_file = 'transcript_master_file.txt'
    transcript_dict = 'transcript_dict.txt'

    #Typo finding parameters - unigram_weight + bigram_weight should equal 1
    unigram_weight = 0.3
    bigram_weight = 0.7
    prob_threshold = 0.0000025 #it might be worth upping this to 0.000003.  Calls 100 more words, but better recall ('aa')


    #Check if ngram language models and consolidated transcripts already exist in directory.  If not, create them.
    # if(exists(unigram_model) and exists(bigram_model) and exists(transcript_file)):
    #     print("Using already existing ngram models")
    # else:
    #Read text files and load to data structures
    ngmod.read_files(dirname=transcript_dir, ftype='transcript', ext ='.srt')
    ngmod.read_files(dirname=textbook_dir, ftype='textbook', ext='.txt')

    #Build unigram models for transcripts and wiki
    transcript_model = ngmod.UnigramLanguageModel(ngmod.transcripts)
    textbook_model = ngmod.UnigramLanguageModel(ngmod.textbook)
    transcript_sorted_vocab_keys = transcript_model.sorted_vocabulary()
    textbook_sorted_vocab_keys = textbook_model.sorted_vocabulary()

    #Build the bigram models for transcripts and wiki
    transcript_bigram_model = ngmod.BigramLanguageModel(ngmod.transcripts)
    textbook_bigram_model = ngmod.BigramLanguageModel(ngmod.textbook)
    transcript_bigram_sorted_vocab_keys = transcript_bigram_model.sorted_vocabulary()
    textbook_bigram_sorted_vocab_keys = textbook_bigram_model.sorted_vocabulary()

    #Calculate Unigram probabilities
    transcript_prob_dict = ngmod.store_unigram_probs(transcript_sorted_vocab_keys, transcript_model)
    textbook_prob_dict = ngmod.store_unigram_probs(textbook_sorted_vocab_keys, textbook_model)
    mix_prob_dict = ngmod.unigram_mixture_probs(transcript_prob_dict, textbook_prob_dict, lam=.3)

    #Calculate the bigram probabilities
    transcript_bigram_prob_dict = ngmod.store_bigram_probs(transcript_bigram_sorted_vocab_keys, transcript_bigram_model)
    textbook_bigram_prob_dict = ngmod.store_bigram_probs(textbook_bigram_sorted_vocab_keys, textbook_bigram_model)
    mix_bigram_prob_dict = ngmod.bigram_mixture_probs(transcript_bigram_prob_dict, textbook_bigram_prob_dict, lam=1)

    #Write unigram model to file
    ngmod.write_dict_to_file(mix_prob_dict, unigram_model)

    #Write bigram model to file
    ngmod.write_tuple_dict_to_file(mix_bigram_prob_dict, bigram_model)

    #Write transcripts data to one file
    ngmod.write_list_to_file(ngmod.transcripts, transcript_file)

    #Load transcripts to data structure
    transcripts =  tf.load_corpus(transcript_file)
    #Instantiate TypoFinder, with user defined weights and threshold
    finder = tf.TypoFinder(unigram_model, bigram_model, prob_threshold, unigram_weight, bigram_weight)
    #List of potential typos
    typo_list = finder.typo_finder(transcripts)

    #Print 'typos' found
    #for typo in typo_list:
        #print("\n", typo)
        #dict_entry = ngmod.get_timestamp(typo[1])
        #print("\n", dict_entry, "\n")
        #print("-----------------------------------------\n")

    with open('results.txt', 'w') as f:
        for typo in typo_list:
            for element in typo:
                if type(element) != str:
                    f.write(str(element) + ' ')
                else:
                    f.write(element + ' ')
            f.write('\n')
            dict_entry = ngmod.get_timestamp(typo[1])
            f.write(str(dict_entry))
            f.write('\n')
            f.write('--------------------------------------------------')
            f.write('\n')
    f.close()

    # typo_df.to_pickle("results_df/typo_df_{}_{}_{}.pkl".format(unigram_weight, bigram_weight, prob_threshold))


#Run program
if __name__ == "__main__":
    main()
