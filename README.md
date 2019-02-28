## **This package is named as “Apollo" in order to double sallute the great Apollo Program.**

*Many years ago the great British explorer George Mallory, who was to die on Mount Everest, was asked why did he want to climb it. He said, "Because it is there."
Well, space is there, and we're going to climb it, and the moon and the planets are there, and new hopes for knowledge and peace are there. And, therefore, as we set sail we ask God's blessing on the most hazardous and dangerous and greatest adventure on which man has ever embarked.*			------President John F. Kennedy

This package will be the foundation on which our platform would be based on, it will be very difficult, however, **DO NOT LET THE FAILURE DEFINE US, LET FAILURE TEACH US !**

## Find usage guide below


# Comment Principle
version_num_data_month, we are now at version 0.1, if want to update version, please talk with the group.
> example: *version_0.1_02_Nov	*

------------  
### 28_Feb_2019  
**Seperate the tutorial module from the Apollo**  
In order to make everything in order, creating a individual tutorial directory  
> For more info, see the README belonging to this directory directly  

------------
### 02_Feb_2019

**Happy New Year coming soon!**  
create NMT frameword by using seq2seq with attention under batch training method  
the attention mechanism is from  
> the Effective approaches to attention-based neural machine translation  
> see the seq2seq_with_attention.py in the '/Apollo/model' directory  

------------
### 28_Jan_2019

**GOOD NEWS !**  
finally, figure out how to train sequence model in batch mode  
this will be an extremely boost in the training speed  
> see the lstm_tagging_batch in the '/Apollo/model' directory  
_love you, my super duper beautiful girl_  

------------
### 22_Jan_2019

finish the lstm_crf tutorial  

------------
### 14_Jan_2019

the encoder_decoder_api could be used for training  
data_format :  
input tensor : a sequence  
output tensor : a sequence  
do not have sequence to tensor function, must be implemented manually  

------------
### 23_Dec_2018

add the crf_suite tutorial in the HMM_CRF directory

------------
### 17_Dec_2018

revise the viterbi algorithm, named 'Viterbi_2nd.py', which provides more clear comments

------------
### 16_Dec_2018

add the viterbi algorithm in HMM_CRF directory

------------

### 15_Nov_2018
**GOOD NEWS !**
add the CRF model to the Apollo directory
the test_folder is used for testing
add the encoder_decoder_attention to the Apollo directory

------------
### 09_Nov_2018
add the model which could be trained by CUDA parallel computing, however, we need to fix the problem as below:
> install CUDA 9.2 on Ubuntu 18.04.1 LTS, because Pytorch doesn't support CUDA 10.0 nowadays and CUDA 9.2 doesn't support Ubuntu 18.04.1 LTS
------------


### 07_Nov_2018
We have a strong weapon today --- our own CBOW model which could be used for training Word2Vec on the specific corpus. 
Notice: If you don't have a specific corpus, please use pre-trained model from google.
This frameword has limited function now, which could only trained the embedding matrix, it doesn't provide other useful interface, such as consine similarity, display the matrix, etc. These functions will be added soon.
Feel free to contact with me if you find any bug.
*Do not eat two ice cream one day, especially in the same time --- 07_Nov*
Jiaxin Zhang

------------
### 06_Nov_2018
1. add an open source framwork, recommended by my supervisor Dr. Andreas Vlachos
GitHub: https://github.com/jiesutd/NCRFpp
2. Add BERT from Google, GOOD news ! Google released its Chinese model.

------------



### 03_Nov_2018
add PyTorch source code tutorial into Apollo directory.
> ~~> one question in _score_sentence(), please fix it.~~
> fixed by Teacher Yan

------------

# Usage

------------

### 1. Word Embedding
> python CBOW_Start.py
use --help for information
