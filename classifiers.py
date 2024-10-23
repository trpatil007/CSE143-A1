import numpy as np

# You need to build your own model here instead of using existing Python
# packages such as sklearn!

## But you may want to try these for comparison, that's fine.
# from sklearn.naive_bayes import MultinomialNB
# from sklearn.linear_model import LogisticRegression

class BinaryClassifier(object):
    """Base class for classifiers.
    """
    def __init__(self):
        pass
    def fit(self, X, Y):
        """Train your model based on training set
        
        Arguments:
            X {array} -- array of features, such as an N*D shape array, where N
              is the number of sentences, D is the size of feature dimensions
            Y {type} -- array of actual labels, such as an N shape array, where
              N is the number of sentences
        """
        pass
    def predict(self, X):
        """Predict labels based on your trained model
        
        Arguments:
            X {array} -- array of features, such as an N*D shape array, where N
            is the number of sentences, D is the size of feature dimensions
        
        Returns:
            array -- predict labels, such as an N shape array, where N is the
            number of sentences
        """
        pass


class AlwaysPredictZero(BinaryClassifier):
    """Always predict the 0
    """
    def predict(self, X):
        return [0]*len(X)

# TODO: Implement this
class NaiveBayesClassifier(BinaryClassifier):
    """Naive Bayes Classifier
    """
    def __init__(self):
        # Add your code here!
        self.class_probs = np.array([]) # P(Class)
        self.wordFreqNegative = np.array([]) # Word Frequencies (negative)
        self.wordFreqPositive = np.array([]) # Word Frequencies (positive)
        self.negativeConditionals = np.array([]) # P(token(i) | Negative)
        self.positiveConditionals = np.array([]) # P(token(i) | Positive)

    def fit(self, X, Y):
        # First, convert our data to numpy arrays
        samples = np.array(X)
        labels = np.array(Y)
        self.wordFreqNegative = np.ones(samples.shape[1]) # now we know the shapes of our input data
        self.wordFreqPositive = np.ones(samples.shape[1])
        # determine the number of classes:
        seen = set()
        total = 0
        numLabels = len(labels)
        for i in range(numLabels):
            if labels[i] not in seen:   
                total += 1
                seen.add(labels[i])
        self.class_probs = np.zeros(total) # initialize our array of class probabilities
        for i in range(len(labels)):
            self.class_probs[labels[i]] += 1
        for i in range(total):
            self.class_probs[i] = np.float32((self.class_probs[i] / numLabels))
        num_samples = samples.shape[0]
        for i in range(num_samples):
            for j in range(samples.shape[1]):
                # add our word frequencies from our training data
                if labels[i] == 0:
                    self.wordFreqNegative[j] += samples[i][j] 
                else:
                    self.wordFreqPositive[j] += samples[i][j]
        totalNeg = np.sum(self.wordFreqNegative)
        totalPos = np.sum(self.wordFreqPositive)
        self.negativeConditionals = np.zeros(samples.shape[1])
        self.positiveConditionals = np.zeros(samples.shape[1])
        # Initialize and calculate our conditional probabilities P(word_i | class)
        for i in range(samples.shape[1]):
            self.negativeConditionals[i] = self.wordFreqNegative[i] / totalNeg
            self.positiveConditionals[i] = self.wordFreqPositive[i] / totalPos
    def predict(self, X):
        # Convert our data to numpy arrays
        samples = np.array(X)
        predLabels = np.zeros(samples.shape[0]) 
        for i in range(samples.shape[0]):
            # Calculate the probability scores of each class
            positiveScore = np.log(self.class_probs[1])
            negativeScore = np.log(self.class_probs[0])
            for j in range(samples.shape[1]):
                positiveScore += samples[i][j] * (np.log(self.positiveConditionals[j]))
                negativeScore += samples[i][j] * (np.log(self.negativeConditionals[j]))
            predLabels[i] = 1 if positiveScore > negativeScore else 0 # determine the class with greater score
        return predLabels

# TODO: Implement this
class LogisticRegressionClassifier(BinaryClassifier):
    """Logistic Regression Classifier
    """
    def __init__(self, lr=0.01, num_steps=1000):
        self.lr = lr
        self.num_steps = num_steps
        self.weights = None
        self.lam = 0.01  # L2 penalty parameter

    def sigmoid(self, z): 
        z = np.clip(z, -88.72, 88.72) # prevent overflow issue by clipping value to range
        return 1 / (1 + np.exp(-z))

    def fit(self, X, Y):
        self.weights = np.zeros(X.shape[1]) # build our model params
        for _ in range(self.num_steps):
            scores = np.dot(X, self.weights)
            predictions = self.sigmoid(scores)
            error = (Y - predictions) # derivative of crossentropy loss = Y - Y_pred
            gradient = np.dot(X.T, error) + (self.lam * self.weights)
            self.weights += self.lr * gradient # update our weights based on our loss

    def predict(self, X):
        scores = np.dot(X, self.weights)
        probs = self.sigmoid(scores)
        labels = []
        for i in range(probs.shape[0]):
            if (probs[i] >= 0.5): 
                labels.append(1)
            else:
                labels.append(0)
        return labels # Return the class with a larger probability score


# you can change the following line to whichever classifier you want to use for
# the bonus.
# i.e to choose NaiveBayes classifier, you can write
# class BonusClassifier(NaiveBayesClassifier):
class BonusClassifier(NaiveBayesClassifier):
    def __init__(self):
        super().__init__()
