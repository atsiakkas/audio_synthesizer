# Audio synthesizer

This project creates an audio synthesizer model able to receive a given phrase from the user and produce and play back to the user a synthesized audio pronunciation of that phrase. The implementated model splits the given phrase into diphones and pauses and uses a set of pre-recorded audio files of all english diphones to synthesize that phrase. 


## Project

https://github.com/atsiakkas/audio_synthesizer<br/>
<br/>

## Contents

**Diphones**: Contains audio files (.wav) of all english diphones

**simpleaudio**: Defines the Audio class which represents the synthesized audio output.

**audio_synthesizer**: Defines the Synth class which performs the text processing of the user's phrase and outputs the synthesized audio as a simpleaudio.Audio instance


## Getting started

The audio synthesizer function can be run through the command line:
```
python audio_synthesizer.py "[your phrase]"
```

The user can additionally specify the following arguments

    --diphones (default="./diphones") --> Folder containing diphone wavs
    
    --play or -p (default=False) --> Play the output audio
    
    --outfile or -o (default=None) --> Save the output audio to a file
    
    --volume or -v (default=None) --> An int between 0 and 100 representing the desired volume
    
    --spell or -s (default=False) --> Spell the input text instead of pronouncing it normally
    
    --reverse or -r (default=None, choices=['words', 'phones', 'signal']) --> Speak backwards in a mode specified by string argument: 'words', 'phones' or 'signal'
    
    --fromfile or -f (default=None) --> Open file with given name and synthesise all text, which can be multiple sentences
    
    --crossfade or -c (default=False) --> Enable slightly smoother concatenation by cross-fading between diphone units





