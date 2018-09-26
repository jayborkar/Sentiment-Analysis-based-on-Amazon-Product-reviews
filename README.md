# Sentiment-Analysis-based-on-Amazon-Product-reviews

Created a sentiment classifier with LSTM network using tensorflow to determine whether the sentiment is positive or negative.
Used pretrained word vectors, created ids matrix for the whole training set of Amazon dataset having 400,000 reviews.
Built Deep Learning model – LSTM in TensorFlow and Python, Accuracy achieved is greater than 87%.

- batchSize = 24
- lstmUnits = 64
- numClasses = 2
- iterations = 100000

## Loading Data
- Created our word vectors using a pretrained model like Word2Vec model or GloVe model.
- Since the word vectors matrix using Word2Vec model is quite large (3.6 GB!), Used a much more manageable matrix that is trained using GloVe (Global Vectors for Word Representation), a similar word vector generation model. 
- The matrix contains 400,000 word vectors, each with a dimensionality of 50.
- Created the ids matrix for the whole training set
 
Training a word vector generation model (such as Word2Vec) or loading pretrained word vectors
- Created IDs matrix, used Word2Vec model and trained the model for the whole training Amazon dataset having more than 400,000 reviews.

## RNN Model (LSTM) 
Created our Tensorflow graph with hyperparameters :
- batchSize = 24
- lstmUnits = 64
- numClasses = 2
- iterations = 100000

- Specified two placeholders, one for the inputs into the network, and one for the labels.
- Called the tf.nn.lookup() function in order to get our word vectors, returns a 3-D Tensor of dimensionality batch size by max sequence length by word vector dimensions. 
- Called the tf.nn.rnn_cell.BasicLSTMCell function,takes in an integer for the number of LSTM units (64)
- fed both the LSTM cell and the 3-D tensor full of input data into a function called tf.nn.dynamic_rnn which unrolls the whole network and creates a pathway for the data to flow through the RNN graph

## Accuracy
- For the optimizer, used Adam and the default learning rate of .001.
- Used Tensorboard to visualize the loss and accuracy values.
- Accuracy achieved is greater than 87 %.

