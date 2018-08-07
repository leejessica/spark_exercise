from pyspark import SparkConf, SparkContext

File='../data/temperature.txt'

def secondSort(file):

    # Create context and set environment variable
    conf=SparkConf().setAppName("Second Sort").setMaster("local")
    sc=SparkContext(conf=conf)
    sc.setLogLevel("WARN")


    text=sc.textFile(file)
    rdd=text.map(lambda line: line.split(','))