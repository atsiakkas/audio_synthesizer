from pathlib import Path
import simpleaudio
import argparse
import nltk
from nltk.corpus import cmudict
#nltk.download('cmudict')
from nltk.tokenize import word_tokenize, sent_tokenize
#nltk.download('punkt')
import re
import string
import wave

import numpy as np
import os
import sys


# Some default values for the audio data
RATE = 16000
ms400 = int(RATE*0.4)
ms200 = int(RATE*0.2)
ms10 = int(RATE*0.01)
tapering = np.array(range(ms10))/ms10              


class Synth:
    '''
    A class used to represent an utterance.
    
    Attributes
    ----------
    phrase: str
        the phrase used to instantiate the class provided by the user.
    spell: bool
        boolean value provided by the user indicating whether the phrase will be spelled.       
    reverse: string or None
        one of [None, 'signal', 'words', 'phones'] provided by the user indicating if and how the phrase will be reversed.         
    play: bool
        boolean value provided by the user indicating whether the synthesized audio will be played out.
    outfile: str or None
        if provided by the user, the synthesized audio will be saved in the outfile file.
    volume: int or None
        if provided by the user, an integer value between 0-100 corresponding the the rescaling factor amplied to the synthesized sequence's amplitudes.
    wav_folder: str
        the location of the folder provided by the user which contains the .wav files for all diphones. 
    crossfade: bool
        boolean value provided by the user indicating whether the diphone concatenation will be smoothed out.
    commas: list
        a list of indices corresponding to the words in the phrase which are followed by ','.
    stops: list
        a list of indices corresponding to the words in the phrase which are followed by '.', '!', '?', or ';'.
    marks: list
        a list of indices corresponding to the words in the phrase which are inside {}.    
    words: list
        the normalised word tokens from pre-processing the phrase.
    phones: list
        a list of lists of phones for each word in the phrase.
    diphones: list
        a list of lists of diphones for each word in the phrase.
    diphones_dict: dict
        a dictionary of diphones with their corresponding .wav file.        
    out: simpleaudio
        an instance of the simpleaudio class.
    
    Methods
    -------
    phrase_to_words()
        Returns normalised words for the given phrase.
    words_to_letters()
        Returns normalised letters for the given phrase.
    words_to_phones():
        Returns a list of lists of the phones for each of the words in the given phrase.
    preprocess_phones(pre_word):
        Takes as input a list of phones and returns a list of those phones in the correct format.
    phones_to_diphones():
        Returns a list of lists of diphones for each word in the given phrase.
    load_diphone_data(wav_folder):
        Returns a dictionary of diphone names and their corresponding .wav file filepaths.
    diphones_to_wav():
        Returns the synthesized audio as a simpleaudio.Audio object.
    '''

    def __init__(self, args, phrase, outfile=None):
        
        # parameters provided by the user
        self.phrase = phrase
        self.spell = args.spell
        self.reverse = args.reverse
        self.play = args.play
        self.outfile = outfile
        self.volume = args.volume
        self.wav_folder = args.diphones
        self.crossfade = args.crossfade
        
        # attributes used for the punctuation and emphasis markups.
        self.commas = []
        self.stops = []
        self.marks = []        
        print(f'Making utterance to synthesise phrase: {self.phrase}') 
        
        # splitting phrase into words
        self.words = self.phrase_to_words()
        self.word_lengths = [len(word) for word in self.words]
        
        # splitting further into letters if specified
        if self.spell:
            self.words = self.words_to_letters()
            # correcting punctuation indices
            self.commas = [sum(self.word_lengths[0:i+1])-1 for i in self.commas]
            self.stops = [sum(self.word_lengths[0:i+1])-1 for i in self.stops]
            self.marks = [sum(self.word_lengths[0:i+1])-1 for i in self.marks]
        
        #reversing if specified.
        if self.reverse == "words":
            self.words = self.words[::-1]
            # correcting punctuation indices
            self.commas = [len(self.words)-1-i for i in self.commas]
            self.stops = [len(self.words)-1-i for i in self.stops]
            self.marks = [len(self.words)-1-i for i in self.marks]
            
        # remove all pauses if phrase needs to be reversed and spelled
        if (self.reverse is not None and self.spell):
            self.commas = []
            self.stops = []
            self.marks = []            
        
        # retieving the phones for each of the phrase's words.
        self.phones = self.words_to_phones()
        
        # reversing if specified.
        if self.reverse == "phones":
            self.phones = [word[::-1] for word in self.phones[::-1]]
            # correcting punctuation indices
            self.commas = [len(self.words)-1-i for i in self.commas]
            self.stops = [len(self.words)-1-i for i in self.stops]
            self.marks = [len(self.words)-1-i for i in self.marks]         
        
        # converting phones into diphones.
        self.diphones = self.phones_to_diphones()
        
        print(f'Normalised phrase: {self.words}')
        print(f'Phone sequence of outputted phrase: {self.phones}')
        print(f'Diphone sequence of outputted phrase: {self.diphones}')
        
        # creating a dictionary of diphones and their corresponding .wav files.
        self.diphones_dict = self.load_diphone_data(self.wav_folder)
        
        # creating a simpleaudio instance using the given phrase.
        self.out = self.diphones_to_wav()
        
        print("Synthesised audio created succesfully!")

    def phrase_to_words(self):
        '''Returns normalised words for the given phrase.
        
        Normalisation involves the splitting of the phrase into individual words, the removal of punctuation 
        and their conversion to lower-case.
                   
        Returns
        -------
        words: list
            a list of normalised words.    
        '''
   
        # converts words into lower-case   
        p_lower = self.phrase.lower()
        # splits the phrase into word tokens using the word_tokenize method from the nltk module.
        tokens_with_punc = word_tokenize(p_lower)
        # if the phrase will be spelled, add "." after each word so that 400ms of silence will be added.
        if self.spell:
            # Code adapted from thread: https://stackoverflow.com/questions/5920643/add-an-item-between-each-item-already-in-the-list
            temp = ["."] * (len(tokens_with_punc)*2-1)
            temp[0::2] = tokens_with_punc
            tokens_with_punc = temp
        # saving indices of words which are followed by specific punctuation. If there is more than one punctuation between words, only the first is considered.
        counter = 0
        for i, word in enumerate(tokens_with_punc):
            try:
                if (word in ",") and (tokens_with_punc[i-1] not in "...,;?!") and i>0:
                    self.commas.append(i-counter-1)          
                elif (word in "...;?!") and (tokens_with_punc[i-1] not in "...,;?!") and i>0:
                    self.stops.append(i-counter-1)
                elif word in "{" and tokens_with_punc[i+2] in "}":
                    self.marks.append(i-counter)               
                if word in string.punctuation or word in "...":
                    counter += 1
            except IndexError:
                # in case punctuation is at the beginning of the phrase before any words, ignore and move on.
                pass
        # removes punctuation from sentence and puts remaining letters into a list 
        letters = [l for l in p_lower if l not in string.punctuation]
        # joins letters back together into a sentence
        p_norm = ''.join(letters)
        # splits the phrase into word tokens using the word_tokenize method from the nltk module
        words = word_tokenize(p_norm)                

        return words
        
    def words_to_letters(self):
        '''Returns normalised letters for the given phrase.
        
        Returns the list of letters for the given phrase based on the list of words.
                   
        Returns
        -------
        letters: list
            a list of normalised letters.    
        '''
        
        # Code adapted from thread: https://stackoverflow.com/questions/952914/how-to-make-a-flat-list-out-of-list-of-lists?rq=1
        letters = [letter for word in self.words for letter in word]
        
        return letters
      

    def words_to_phones(self):
        '''Returns a list of lists of the phones for each of the words in the given phrase.
        
        For each word in the given phrase, its comprised phones list is returned which is obtained from the phone 
        dictionary of the Carnegie Mellon University (cmudict corpus from nltk).
        
        Returns
        -------
        phones: list
            a list of lists of phones for each word in the phrase.    
        '''
        
        phones = []
        for i, word in enumerate(self.words):
            if word not in cmudict.dict():
                sys.exit(f"Synthesis failed and program exited. Reason: The word '{word}' is not in the cmu dictionary and cannot be synthesized. Please use a different word.")
            else:
                word_phones = self.preprocess_phones(cmudict.dict()[word][0])
                
                if len(self.words) == 1:
                    phones.append(['pau'] + word_phones + ['pau'])
                elif i==0 and not ((i in self.commas) or (i in self.stops)):
                    next_word_phones = self.preprocess_phones(cmudict.dict()[self.words[i+1]][0])
                    phones.append(['pau'] + word_phones + [next_word_phones[0]]) 
                elif i==0 and ((i in self.commas) or (i in self.stops)):
                    phones.append(['pau'] + word_phones + ['pau'])
                elif (i in self.commas) or (i in self.stops) or (i==len(self.words)-1):
                    phones.append(word_phones + ['pau'])
                else:
                    next_word_phones = self.preprocess_phones(cmudict.dict()[self.words[i+1]][0])
                    phones.append(word_phones + [next_word_phones[0]])
        
        return phones
    
    def preprocess_phones(self, pre_word):
        '''Takes as input a list of phones and returns a list of those phones in the correct format.
        
        Each phone in the input list is processed which involves removing the lexical stress markers and
        converting it to lower case.
        
        Parameters
        ----------
        pre_word: list
            a list of pre-processed diphones
        
        Returns
        -------
        pro_word: list
            a list of processed phones.    
        '''
         
        pro_word = []
        for phone in pre_word:
            # removing lexical stress markers of vowels. Code adapted from )thread: https://stackoverflow.com/questions/12851791/removing-numbers-from-string
            phone = ''.join(l for l in phone if not l.isdigit()) 
            # converting to lower case
            phone = phone.lower()
            pro_word.append(phone)
        
        return pro_word
    
    def phones_to_diphones(self):
        '''Returns a list of lists of diphones for each word in the given phrase.
        
        The list of phones of each word is converted into a list of diphones for the given phrase.
        
        Returns
        -------
        diphones: list
            a list of lists of diphones for each word in the given phrase.    
        '''
             
        # Create empty diphone list to store phrases diphones
        diphones = []
        for word in self.phones:
            # append list of diphones for the given list of phones
            diphones.append([word[i] + '-' + word[i+1] for i in range(len(word)-1)])
                         
        return diphones
  
    def load_diphone_data(self, wav_folder):
        '''Returns a dictionary of diphone names and their respective .wav file filepaths.
        
        Returns a dictionary of diphone names for the diphone files in the input folder filepath to their 
        corresponding filepaths in that folder.
        
        Parameters
        ----------
        wav_folder: str
            the string specifying the filepath to the diphones folder.
        
        Returns
        -------
        diphones_dict: dict
            a dictionary of diphone names and their corresponding .wav file filepath.    
        '''
        
        # create empty dictionary to store key-value pairs
        diphones_dict = {}
        # store all .wav files which are in the provided wav_folder
        all_diphone_wav_files = (str(item) for item in Path(wav_folder).glob('*.wav') if item.is_file())

        for wav_file in all_diphone_wav_files:
            # save the name of the diphone as the key and its filepath as its value.
            diphones_dict[wav_file[9:-4]] = wav_file

        return diphones_dict
    
    def diphones_to_wav(self):
        '''Returns the synthesized audio as a simpleaudio.Audio object.
        
        The list of diphones for the given phrase is processed, the diphone .wav files are retrieved and concatenated
        and saved in a simpleaudio object. The following are implemented if based on user's inputs:
            (i) Pauses are added where the user provided specific types of punctuation.
            (ii) Words marked by the user are emphasized by increasing their volume (rescaling their amplitude sequence).
            (iii) The audio file's volume is adjusted.
            (iv) The sythesized audio is played out.
            (v) The synthesized audio is saved in a .wav file with a name specified by the user.
            (vi) The synthesized audio's signal is reversed.
            (vii) Smooths out the diphone concatenation
        Finally, the function returns the processed synthesized audio as a simpleaudio object.
               
        Returns
        -------
        synth_audio: simpleaudio.Audio
            an instance of the simpleaudio.Audio class.    
        '''
        
        # initialise synth_audio as an empty simpleaudio.Audio object        
        synth_audio = simpleaudio.Audio(rate=RATE)
        
        # create simpleaudio.Audio instances to represent the required silence pauses.
        ms200_silence = simpleaudio.Audio(rate=RATE)
        ms200_silence.create_tone(0, ms200, 0)
        ms400_silence = simpleaudio.Audio(rate=RATE)
        ms400_silence.create_tone(0, ms400, 0)
        
        for i, word in enumerate(self.diphones):
            # create a temporary simpleaudio.Audio object to store the audio data for each word. 
            word_audio = simpleaudio.Audio(rate=RATE)
            
            for j, diphone in enumerate(word):
                # create a temporary simpleaudio.Audio object to store the audio data for each diphone.
                diphone_audio = simpleaudio.Audio(rate=RATE)
                # check if diphone is in the dictionary.
                if diphone not in self.diphones_dict.keys():
                    sys.exit(f"Synthesis failed and programme exited. Reason: The diphone '{diphone}' is not in the list of saved .wav diphone files.")
                # load the .wav file's data onto the simpleaudio.Audio instance.
                diphone_audio.load(self.diphones_dict[diphone])
                # if crossfade is true, smooth the diphone connections.
                if self.crossfade:
                    # check that the current diphone is not the first in the sequence. 
                    if not j==0:
                        # taper the first 10ms of the current diphone sequence with the last 10ms of the previous diphone sequence.
                        diphone_audio.data[0:ms10] = diphone_audio.data[0:ms10]*tapering + prev_diphone_last_10ms*tapering[::-1]
                    # check that the current diphone is not the last in the sequence.
                    if not (j==len(word)-1 and i==len(self.diphones)-1):
                        # save last 10ms of audio
                        prev_diphone_last_10ms = diphone_audio.data[-ms10:]
                        # remove last 10ms of audio
                        diphone_audio.data = diphone_audio.data[0:-ms10]
                # append the audio data of the current diphone to the audio data of the whole word.
                word_audio.data = np.append(word_audio.data, diphone_audio.data)
            
            # process audio data if necessary, based on the punctuation of the given phrase.
            if i in self.commas:
                # add 200ms of silence.
                word_audio.data = np.append(word_audio.data, ms200_silence.data)
            if i in self.stops:
                # add 400ms of silence.
                word_audio.data = np.append(word_audio.data, ms400_silence.data)
            if i in self.marks:
                # increase volume to max.
                word_audio.rescale(1)
            
            # append the audio data of the current word to the audio data of the whole phrase.
            synth_audio.data = np.append(synth_audio.data, word_audio.data)
        
        # if user provided a volume parameter, rescale accordingly.
        if self.volume is not None:
            synth_audio.rescale(self.volume*0.01)
        
        # if user provided a reverse parameter of "signal", reverse the audio signal.
        if self.reverse == "signal":
            synth_audio.data = synth_audio.data[::-1]
         
        # if user provided the play parameter, play the audio. 
        if self.play:
            synth_audio.play()
        
        # if user provided an output file, save the audio there.
        if self.outfile is not None:
            synth_audio.save(self.outfile)
            print(f"Synthesised audio file saved: {self.outfile}")
                    
        return synth_audio
                   
                
# NOTE: DO NOT CHANGE ANY OF THE ARGPARSE ARGUMENTS - CHANGE NOTHING IN THIS FUNCTION
def process_commandline():
    parser = argparse.ArgumentParser(
        description='A basic text-to-speech app that synthesises speech using diphone concatenation.')

    # basic synthesis arguments
    parser.add_argument('--diphones', default="./diphones",
                        help="Folder containing diphone wavs")
    parser.add_argument('--play', '-p', action="store_true", default=False,
                        help="Play the output audio")
    parser.add_argument('--outfile', '-o', action="store", dest="outfile",
                        help="Save the output audio to a file", default=None)
    parser.add_argument('phrase', nargs='?',
                        help="The phrase to be synthesised")

    # Arguments for extension tasks
    parser.add_argument('--volume', '-v', default=None, type=int,
                        help="An int between 0 and 100 representing the desired volume")
    parser.add_argument('--spell', '-s', action="store_true", default=False,
                        help="Spell the input text instead of pronouncing it normally")
    parser.add_argument('--reverse', '-r', action="store", default=None, choices=['words', 'phones', 'signal'],
                        help="Speak backwards in a mode specified by string argument: 'words', 'phones' or 'signal'")
    parser.add_argument('--fromfile', '-f', action="store", default=None,
                        help="Open file with given name and synthesise all text, which can be multiple sentences.")
    parser.add_argument('--crossfade', '-c', action="store_true", default=False,
                        help="Enable slightly smoother concatenation by cross-fading between diphone units")

    args = parser.parse_args()

    if (args.fromfile and args.phrase) or (not args.fromfile and not args.phrase):
        parser.error('Must supply either a phrase or "--fromfile" to synthesise (but not both)')

    return args


## Helper functions

def read_from_file(file):
    '''Returns the tokenized sentences of a text file.
        
    Returns the tokenized sentences of the provided file using the nltk.sent_tokenize function.
        
    Parameters
    ----------
    file: str
        the filepath of the text file.
        
    Returns
    -------
    sentences: list
        a list of sentences of the provided text file.    
    '''
    
    with open(file, 'r') as file:
        sentences = sent_tokenize(file.read())
    
    return sentences

def validate_volume(volume):
    '''Checks whether the provided volume is in the range [0,100].
        
    Checks whether the provided volume is in the range [0,100] and if not interrupts the program and prints an
    informative message for the user.
        
    Parameters
    ----------
    volume: int
        the volume provided by the user.
         
    '''
    
    if volume < 0 or volume > 100:
        sys.exit(f"Synthesis failed and programme exited. Reason: The volume provided is not between 0 and 100.")
    

if __name__ == "__main__":
    
    # retrieve user inputs
    args = process_commandline()
    
    # validate volume if given
    if args.volume is not None:
        validate_volume(args.volume)
    # if the user did not provide a fromfile argument, create an instance of the Synth class with the provided phrase.
    if args.fromfile is None:
        diphone_synth = Synth(args=args, phrase=args.phrase, outfile=args.outfile)
    # if the user provided a fromfile argument but not an outfile argument, create an instance of the Synth class for every sentence in the file.
    elif args.outfile is None:
        for sentence in read_from_file(args.fromfile):
            diphone_synth = Synth(args=args, phrase=sentence)
    # if the user provided a fromfile argument and an outfile argument, create an instance of the Synth class for every sentence in the file and save the audio for all sentences in the file.
    else:
        out = simpleaudio.Audio(rate=RATE)
        for sentence in read_from_file(args.fromfile):
            diphone_synth = Synth(args=args, phrase=sentence)
            out.data = np.append(out.data, diphone_synth.out.data)
        out.save(args.outfile)
        print(f"Synthesised audio file saved: {args.outfile}")
        


