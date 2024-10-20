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
        self.class_probs = np.array([])
        self.wordFreqNegative = np.array([])
        self.wordFreqPositive = np.array([])
        self.negativeConditionals = np.array([]) # P(token(i) | Negative)
        self.positiveConditionals = np.array([]) # P(token(i) | Positive)

    def fit(self, X, Y):
        samples = np.array(X)
        labels = np.array(Y)
        self.wordFreqNegative = np.ones(samples.shape[1])
        self.wordFreqPositive = np.ones(samples.shape[1])
        seen = set()
        total = 0
        numLabels = len(labels)
        for i in range(numLabels):
            if labels[i] not in seen:   
                total += 1
                seen.add(labels[i])
        self.class_probs = np.zeros(total)

        for i in range(len(labels)):
            self.class_probs[labels[i]] += 1
        for i in range(total):
            self.class_probs[i] = np.float32((self.class_probs[i] / numLabels))
        num_samples = samples.shape[0]
        for i in range(num_samples):
            for j in range(samples.shape[1]):
                if labels[i] == 0:
                    self.wordFreqNegative[j] += samples[i][j]
                else:
                    self.wordFreqPositive[j] += samples[i][j]
        totalNeg = np.sum(self.wordFreqNegative)
        totalPos = np.sum(self.wordFreqPositive)
        self.negativeConditionals = np.zeros(samples.shape[1])
        self.positiveConditionals = np.zeros(samples.shape[1])
        for i in range(samples.shape[1]):
            self.negativeConditionals[i] = self.wordFreqNegative[i] / totalNeg
            self.positiveConditionals[i] = self.wordFreqPositive[i] / totalPos
    def predict(self, X):
        samples = np.array(X)
        predLabels = np.zeros(samples.shape[0])
        for i in range(samples.shape[0]):
            negProb = 0
            posProb = 0
            positiveScore = np.log(self.class_probs[1])
            negativeScore = np.log(self.class_probs[0])
            for j in range(samples.shape[1]):
                positiveScore += samples[i][j] * (np.log(self.positiveConditionals[j]))
                negativeScore += samples[i][j] * (np.log(self.negativeConditionals[j]))
            predLabels[i] = 1 if positiveScore > negativeScore else 0
        return predLabels

# TODO: Implement this
class LogisticRegressionClassifier(BinaryClassifier):
    """Logistic Regression Classifier
    """
    def __init__(self, lr=0.01, num_steps=10000):
        self.lr = lr
        self.num_steps = num_steps
        self.weights = None

    def sigmoid(self, z):
        return 1 / (1 + np.exp(-z))

    def fit(self, X, Y):
        num_samples, num_features = X.shape
        self.weights = np.zeros(num_features)

        for step in range(self.num_steps):
            scores = np.dot(X, self.weights)
            predictions = self.sigmoid(scores)
            error = Y - predictions
            gradient = np.dot(X.T, error)
            self.weights += self.lr * gradient

    def predict(self, X):
        scores = np.dot(X, self.weights)
        probabilities = self.sigmoid(scores)
        return (probabilities >= 0.5).astype(int)


# you can change the following line to whichever classifier you want to use for
# the bonus.
# i.e to choose NaiveBayes classifier, you can write
# class BonusClassifier(NaiveBayesClassifier):
class BonusClassifier(NaiveBayesClassifier):
    def __init__(self):
        super().__init__()
