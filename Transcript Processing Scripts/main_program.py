import unigram_bigram_v2 as ngmod
import typo_finder as tf
from os.path import exists


#Main function to set up unigram and bigram models, load transcripts and find typos
def main():

    transcript_dir = 'C:\\Users\\scott\\Documents\\School\\CS410\\CourseProject\\transcripts_srt\\transcripts_srt\\textanalytics_srt'
    textbook_dir = 'C:\\Users\\scott\\Documents\\School\\CS410\\CourseProject\\Textbook'

    unigram_model = 'mixt_frequencies.txt'
    bigram_model = 'mixt_bigram_frequencies.txt'

    transcript_file = 'C:\\Users\\scott\\Documents\\School\\CS410\\CourseProject\\Transcript Processing Scripts\\transcript_master_file.txt'

    #Check if ngram language models already exists in directory.  If not, create them.
    if(exists(unigram_model) and exists(bigram_model)):
        print("Using already existing ngram models")
    else:
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
        mix_prob_dict = ngmod.unigram_mixture_probs(transcript_prob_dict, textbook_prob_dict, lam=.1)

        #Calculate the bigram probabilities
        transcript_bigram_prob_dict = ngmod.store_bigram_probs(transcript_bigram_sorted_vocab_keys, transcript_bigram_model)
        textbook_bigram_prob_dict = ngmod.store_bigram_probs(textbook_bigram_sorted_vocab_keys, textbook_bigram_model)
        mix_bigram_prob_dict = ngmod.bigram_mixture_probs(transcript_bigram_prob_dict, textbook_bigram_prob_dict, lam=.1)

        #Write unigram model to file
        ngmod.write_dict_to_file(mix_prob_dict, unigram_model)

        #Write bigram model to file
        ngmod.write_tuple_dict_to_file(mix_bigram_prob_dict, bigram_model)
      

    #Load transcripts to data structure
    transcripts =  tf.load_corpus(transcript_file)
    #Instantiate TypoFinder
    finder = tf.TypoFinder(unigram_model, bigram_model, .00005, .05, .05)
    #List of potential typos
    typo_list = finder.typo_finder(transcripts)
    #Print 'typos' found
    for typo in typo_list:
        print(typo)


#Run program
if __name__ == "__main__":
    main()
