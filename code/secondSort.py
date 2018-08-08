from pyspark import SparkConf, SparkContext

File='../data/temperature.txt'


def f(x):
    buf=""
    for i in x[1]:
        buf=buf+i
        buf=buf+" ,"
    buf=buf[0:len(buf)-1]
    print(x[0]+" "+buf)

def secondSort(file):

    # Create context and set environment variable
    conf=SparkConf().setAppName("Second Sort").setMaster("local")
    sc=SparkContext(conf=conf)
    sc.setLogLevel("WARN")
    text=sc.textFile(file)
    rdd=text.map(lambda line: line.split(",")).map(lambda x: (x[0]+"-"+x[1], [x[3]])).sortByKey(False).reduceByKey(lambda x, y: sorted(x+y, reverse=True))
    rdd.foreach(lambda x: f(x))
    sc.stop()
