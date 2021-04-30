# ChineseSpeech2EnglishText
Used Wav2Vec2 model to achieve ASR and Machine Translation for Chinese Speech to English Text.

# Group work with:
Daniel Cheng (@EqdCheng)
Yuqing(Echo) Zhang (@piggyeq)
Yuqian Zhang (@badcode)
# Introduction
The nature of our task is to fine-tuning XSLR model on Chinese spoken data using data from Common Voice. We first took a sequence of spoken data from our Common Voice datasource, trained the model, checked the performance of the model using the Common Voice dataset, and compared it with the reference [performance](https://paperswithcode.com/sota/speech-recognition-on-common-voice-chinese-2). 
After the ASR model was completed, we created a Chinese to English machine translation model with openNMT, using a dataset from OPUS. Though the context and domain of the Common Voice dataset and the OPUS dataset are different, we evaluated the entire pipeline with a single BLEU score. This was done with the entire CoVoST dataset, to properly score our full model. We chose CoVoST because it contains spoken and written data for both languages, allowing us to easily evaluate whether our pipeline was performing correctly. 
# Motivation and contribution

We were interested in this project because we wanted to explore the complications of Chinese ASR. Currently, the off-the-shelf ASR from Google on our phones is great, but once it’s translated into English, it’s quite poor. We wanted to rebuild an ASR model and attach a Chinese to English machine translation model. This way, we would be able to understand where the translation errors arise, identifying if it’s an ASR problem, machine translation problem, or another unknown problem. 
In addition, we were interested in learning more about the Wav2Vec 2.0 from Facebook. We hoped to create a model that can quickly and accurately translate spoken Chinese into English text, while gaining experience with the newest models that relate to automatic speech recognition.
 
# Data

We used data from CommonVoice, which is an open-source Chinese Mandarin speech corpus that contains roughly 78 hours of spoken texts recorded by people from different regions in China. The additional evaluation part included data from CoVost with about 8 hours’ long of spoken texts. We obtained the CommonVoice data via HuggingFace load_dataset method.
Preprocessing the data was much more difficult than anticipated, since we had seen a tutorial and it looked relatively simple. However, since we only had 1 GPU on Google Colab, we ran into bottlenecks during our downsampling, model training, as well as model saving. 
 
# Engineering

We used Google Colab to run all our notebooks and models. We tried to do different steps in separate files to limit the memory and disk usage, but we still ran into ram, memory, and storage issues. For example, we tried loading and processing data in one notebook to store as pickle files and load pickle files later from another notebook. However, we ran into Drive storage quota problems. In addition, after we had done all necessary pre-processing steps and converted the needed features for each audio to a Dataset type of dictionary in a JSON file, we ran out of allocated RAM on Colab. This made it impossible to generate and re-use what we currently had for the training data. 
Ultimately, our solution was to split the dictionary into smaller datasets. The original train dictionary had around 18,000 entries, with about 26 GB in JSON format. We used basic looping to split the data into multiple different sub-train dictionary JSON files, which contained approximately 1,000 entries. 
For the first part of speech recognition, we referenced code from [HuggingFace](https://huggingface.co/ydshieh/wav2vec2-large-xlsr-53-chinese-zh-cn-gpt/blob/main/README.md). This is fine-tuned based on the CommonVoice dataset (Common Voice Corpus 6.1, Chinese) as well, which provides us an apple-to-apple comparison in terms of character error rate. Then we did further tuning on the CoVoST [train dataset](https://github.com/facebookresearch/covost).
 
# Evaluation
 
We first compared our model with the [hugging face model](https://huggingface.co/ydshieh/wav2vec2-large-xlsr-53-chinese-zh-cn-gpt/blob/main/README.md) to get the character error on CoVoST dev set. The reason we are using character error rate, is because character-level models benefit from having less grammatical errors ([Wang](https://www.aclweb.org/anthology/2020.aacl-main.20.pdf), 2020). In addition, character-level models can avoid errors caused by incorrect segmentation and out-of-vocabulary words ([Jia](https://journals.sagepub.com/doi/pdf/10.1177/0020294020952456), 2020). 

We chose a baseline of 30% CER, since the HuggingFace model achieved around 20%. After achieving this, we continued onto the machine translation, scoring a XXX% accuracy score. As a final evaluation, we used the CovoST dataset to test the entire end-to-end model, using character-level BLEU scores. In our research, a character-level BLEU test score of 5.8 on the CoVoST dataset was achieved with Zh-En transformer models, and we achieved XXX ([Wang](https://arxiv.org/pdf/2010.05171.pdf), 2020). 
