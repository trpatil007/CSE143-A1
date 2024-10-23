# Movie Review Sentiment Classification- Tanishq Patil, Manav Patel, Saumit Vedula, Ashwin Marichetty
Given movie review dataset, predict sentiment as positive(1), negative(0). 

We implement 2 main models: Naive Bayes and Logistic Regression and track/observe the results. 

At the current moment, there is no implementation to save model files locally (to avoid retraining).
## Classifiers
Usage (training and testing):
```bash
python main.py --model "AlwaysPredictZero"
```

```bash
python main.py --model "NaiveBayes"
```

```bash
python main.py --model "LogisticRegression"
```


## Data

Data has been pre-split into training, dev, and test splits in `data/`, with a CSV file for each split.

The file: 'download_and_split_data.py' processes our data into python for us. 