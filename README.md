# UTIR


# sentimental_analysis.py
This file attemps to grab the word vectors from the reviews in the IMBD analysis set This file extracts the negative reviews from the IMBD set, tockenizes the negative reviews, and creates a word-integer mapping dictionary with the words in the negative reviews. Glove Model was then initialized for an embedded dimension of 32, hidden dimension of 32 and batch size of 16. Number of Epochs is 1, train length was split into a ration of 0.95 for training and 0.05 for validation. Had the classifier been successful, training and validation results would have an accuracy ranging from 88 - 98 % within 11 seconds [2]. Unresolved issues are "string" variable type input into torch.tensor() (line 21).

Code development was derived from:  
[1]https://github.com/pytorch/tutorials/blob/master/beginner_source/text_sentiment_ngrams_tutorial.py  
[2]https://pytorch.org/tutorials/beginner/text_sentiment_ngrams_tutorial.html


# main.py 
This file attempts to implement LSTM classifer .This file extracts the negative reviews from the IMBD analysis, tockenizes the negative reviews, and creates a word-integer mapping dictionary with the words in the negative reviews. Unresolved issue is passing class parameters into the classifer SentimentLSTM with torch.nn.Module (line 37).

Code development was derived from:    
[3]https://towardsdatascience.com/sentiment-analysis-using-lstm-step-by-step-50d074f09948?gi=3f7be9732bbe  
