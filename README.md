# word2graph2vec

Representation learning for words and Labelled Documents by jointly representing words, labels and documents as nodes of a graph and then learning low dimensional representation of it. 

The method is based on the paper by Tang J. et. al (http://research.microsoft.com/pubs/255567/fp292-Tang.pdf). 

# Current state of project

Current task is sentiment analysis on IMDB dataset with two labels (sentiments - positive, negative). It constructs word-word, word-doc and word-label graph from data and trains a skip-gram based model for learning low dimensional representation of nodes in graph.

Currently it trains model only on word-word graph. Training jointly on all three graphs is in development phase. We are running experiments.

# Results

Current implementaion has quite poor results. It gives ~55% accuracy on test.

# State - Development

The project is still in development state. We will change the state of the project to alpha when it gives decent performance of dataset

# Authors

For any queries related to project you can contact the authors:

Shashank Gupta - 27392shashankgupta@gmail.com

Karan Chandnani - karanchandnani21@gmail.com

Nishant Prateek - nishant.prateek@research.iiit.ac.in
