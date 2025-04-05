<<<<<<< HEAD
# Emotions-detections-using-machine-learning-and-deep-learning
=======
# Speech-Emotion-Recognition


**Data Description:**

Combined data from RAVDESS speech and song datasets and Toronto emotional speech set. Data sets available in Kaggle:

**TESS Data**: https://www.kaggle.com/ejlok1/toronto-emotional-speech-set-tess : There are a set of 200 target words were spoken in the carrier phrase "Say the word _' by two actresses (aged 26 and 64 years) and recordings were made of the set portraying each of seven emotions (anger, disgust, fear, happiness, pleasant surprise, sadness, and neutral). There are 2800 data points (audio files) in total.
The dataset is organised such that each of the two female actor and their emotions are contain within its own folder. And within that, all 200 target words audio file can be found. The format of the audio file is a WAV format.

**RAVDESS Data**: https://www.kaggle.com/uwrfkaggler/ravdess-emotional-speech-audio : RAVDESS data contains 1440 files: 60 trials per actor x 24 actors = 1440. The RAVDESS contains 24 professional actors (12 female, 12 male), vocalizing two lexically-matched statements in a neutral North American accent. Speech emotions includes calm, happy, sad, angry, fearful, surprise, and disgust expressions. Each expression is produced at two levels of emotional intensity (normal, strong), with an additional neutral expression. Each of the 1440 files has a unique filename. The filename consists of a 7-part numerical identifier (e.g., 03-01-06-01-02-01-12.wav). These identifiers define the stimulus characteristics:
File name Identifiers:
•	Modality (01 = full-AV, 02 = video-only, 03 = audio-only).
•	Vocal channel (01 = speech, 02 = song).
•	Emotion (01 = neutral, 02 = calm, 03 = happy, 04 = sad, 05 = angry, 06 = fearful, 07 = disgust, 08 = surprised).
•	Emotional intensity (01 = normal, 02 = strong). NOTE: There is no strong intensity for the 'neutral' emotion.
•	Statement (01 = "Kids are talking by the door", 02 = "Dogs are sitting by the door").
•	Repetition (01 = 1st repetition, 02 = 2nd repetition).
•	Actor (01 to 24. Odd numbered actors are male, even numbered actors are female).


**Extraction of features for audio files using Mel Frequency Cepstral Coefficients**

The RAVDESS and TESS datasets were used to create combined dataset. Then, the features were extracted for all the audio files using MFCC. Mel-frequency cepstral coefficients (MFCC) includes following steps:

	**A/D Conversion**: It converts the analog signal into discrete space.

	**Pre-emphasis**: It boosts the amount of energy in higher frequencies. Boosting the high-frequency energy makes information in higher formants more available to the acoustic model. Filter is used to boost the amount of energy in higher frequencies.  

	**Windowing**: It involves slicing of waveform for audio into sliding frames. For slicing, the amplitude needs to gradually drop off at the edge of the frame. Hamming or hanning window can be used for windowing. 

	**DFT (Discrete fourier Transform (DFT)**: Used to extract information in the frequency domain.
 
	**Mel Filter bank**: Mel scale is used to map the actual frequency to the frequency that the humans will perceive.

	**Applying log**: Log function is applied to Mel filter output to mimic the human hearing system. At a low value of input log function gradient will be higher and at high value of input log function, function gradient will be lower which kind of similar to how humans perceive sound.

	**IDFT**: Inverse transformation to the output in the previous step, is applied. The periods in the time domain and frequency domain are inverted after the transformations. So, the frequency domain’s fundamental frequency with the lowest frequency will have the highest frequency in the time domain. The inverse of the log of the magnitude of the signal is called a cepstrum.

	The MFCC model takes the first 12 coefficients of the signal after applying the IDFT operations. Energy of signal sample is also considered as feature, which helps in identifying the phones.  

Along with these 13 features, the MFCC technique will consider the first order derivative and second order derivatives of the features which constitute another 26 features. Overall, MFCC technique will generate 39 features from each audio signal sample which are used as input for the speech recognition model.



**Exploratory Data Analysis**

•	Plotting waveform for some random audio files
•	Visualizing MFCC transformation for some audio files
•	Bar graph to visualize the count of various emotions


**Models Implementation**


  

	**CNN – Convolutional Neural Network: Class of deep neural network:**

Convolutional neural networks are distinguished from other neural networks by their superior performance with image, speech, or audio signal inputs. They have three main types of layers: 
Convolutional layer, pooling layer and fully-connected layer.
The convolutional layer is the first layer of a convolutional network. While convolutional layers can be followed by additional convolutional layers or pooling layers, the fully-connected layer is the final layer. The convolutional layer is the core building block of a CNN, and it is where the majority of computation occurs.

	**MLP – Multilayer Perceptron: Class of feed forward artificial neural network**

MLP Classifier stands for Multi-layer Perceptron classifier. MLP is a feedforward artificial neural network model that maps input data sets to a set of appropriate outputs. An MLP consists of multiple layers and each layer is fully connected to the following one. The nodes of the layers are neurons with nonlinear activation functions, except for the nodes of the input layer. Between the input and the output layer there may be one or more nonlinear hidden layers.


•	Standardization of data to improve score in both models.

•	Evaluating score for training and testing data for both the models

•	Applying early stopping to balance the number of epochs and reduce overfitting in the models.

•	Accuracy plot and Loss plot for training and testing data for different number of epochs.

•	Saving models for further use

**Testing audio data**

•	Applied saved CNN model on randomly selected audio files to test the results.

5.	Making model accessible as a web application using streamlit

•	Via Streamlit, created web application to test speech emotion using CNN and MLP models both, providing end user with the interface to directly provide audio file and detect emotion.




>>>>>>> 4224e4a (Initial commit)
