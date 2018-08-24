from pyspark import SparkContext, SparkConf
import logging
import os
import re
import shutil


def wordcount(sourceFile, destFile):
    conf = SparkConf().setAppName("Word Count").setMaster("local[*]")
    sc = SparkContext(conf=conf)
    file = sc.textFile(sourceFile)
    rdd = file.map(lambda line: preprocessing(line)).flatMap(lambda line: line.split(" ")).map(
        lambda word: str.strip(word)).filter(lambda word: len(word)>0).map(lambda word: (word, 1)).reduceByKey(lambda x, y: x+y)
    if os.path.isdir(destFile):
        shutil.rmtree(destFile)
    rdd.coalesce(1).saveAsTextFile(destFile)


def preprocessing(text):
    regEx = "[`~!@#$%^&*()+=|{}':;',\\[\\].<>/?~！@#￥%……&*（）——+|{}【】‘；：”“’。，、？]"
    text = re.sub(regEx, "", text)
    text = re.sub(r"\d", "", text)
    text = re.sub(r"\t", "", text)
    return text


if __name__ == "__main__":
    sourceFile = "../data/wordcount.txt"
    destFile = "../data/WORDCOUNT1"
    wordcount(sourceFile, destFile)
    logging.info("Finish word count!!!!")
