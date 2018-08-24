from pyspark.sql import SQLContext
from pyspark import SparkContext
from pyspark.ml.feature import RegexTokenizer, StopWordsRemover, CountVectorizer, OneHotEncoder, StringIndexer, \
    VectorAssembler, HashingTF, IDF
from pyspark.ml.classification import LogisticRegression
from pyspark.ml import Pipeline
from pyspark.ml.evaluation import MulticlassClassificationEvaluator


def readData(file):
    sc = SparkContext()
    sqlContext = SQLContext(sc)
    data = sqlContext.read.format('com.databricks.spark.csv').options(header='true', inferSchema='True').load(file)
    return data


def word_Frequency(data):
    regexTokenizer = RegexTokenizer(inputCol="Descript", outputCol="Words", pattern="\\W")
    stopWords = ["http", "https", "amp", "rt", "t", "c", "the"]
    stopWordsRemover = StopWordsRemover(inputCol="Words", outputCol="Filtered").setStopWords(stopWords)
    countVectorizer = CountVectorizer(inputCol="Filtered", outputCol="features", vocabSize=10000, minDF=5)
    lable_StringIdx = StringIndexer(inputCol="Category", outputCol="label")
    pipeline = Pipeline(stages=[regexTokenizer, stopWordsRemover, countVectorizer, lable_StringIdx])
    pipelineFit = pipeline.fit(data)
    return pipelineFit.transform(data)


def TF_IDF(data):
    regexTokenizer = RegexTokenizer(inputCol="Descript", outputCol="Words", pattern="\\W")
    stopWords = ["http", "https", "amp", "rt", "t", "c", "the"]
    stopWordsRemover = StopWordsRemover(inputCol="Words", outputCol="Filtered").setStopWords(stopWords)
    hashingTF = HashingTF(inputCol="Filtered", outputCol="rawFeatures", numFeatures=1000)
    idf = IDF(inputCol="rawFeatures", outputCol="features", minDocFreq=5)
    lable_StringIdx = StringIndexer(inputCol="Category", outputCol="label")
    pipeline = Pipeline(stages=[regexTokenizer, stopWordsRemover, hashingTF, idf, lable_StringIdx])
    pipelineFit = pipeline.fit(data)
    return pipelineFit.transform(data)


def LR_Model_with_WordFrequency(data):
    # split train and test data
    (trainningData, testData) = data.randomSplit([0.7, 0.3], seed=100)
    lr = LogisticRegression(maxIter=20, regParam=0.1, elasticNetParam=0)
    lrModel = lr.fit(trainningData)
    predictions = lrModel.transform(testData)
    evaluator = MulticlassClassificationEvaluator(predictionCol="prediction")
    print("LR model using CountVectorizer precision: ")
    print(evaluator.evaluate(predictions))
    print("\n")


def LR_Medel_with_TF_IDF(data):
    (trainingData, testData) = data.randomSplit([0.7, 0.3], seed=100)
    lr = LogisticRegression(maxIter=20, regParam=0.1, elasticNetParam=0)
    lrModel = lr.fit(trainingData)
    predictions = lrModel.transform(testData)
    evaluator = MulticlassClassificationEvaluator(predictionCol="prediction")
    print("LR model using TF_IDF precision: ")
    print(evaluator.evaluate(predictions))
    print("\n")


if __name__ == "__main__":
    file = "../data/multi_text_classification/train.csv"
    data = readData(file)
    # dataset=word_Frequency(data)
    # LR_Model_with_WordFrequency(dataset)
    dataset = TF_IDF(data)
    LR_Medel_with_TF_IDF(dataset)
