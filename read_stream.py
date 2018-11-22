import sys
assert sys.version_info >= (3, 5) # make sure we have Python 3.5+

from pyspark.sql import SparkSession, functions, types
spark = SparkSession.builder.appName('read stream').getOrCreate()
assert spark.version >= '2.3' # make sure we have Spark 2.3+
spark.sparkContext.setLogLevel('WARN')

def main(input_topic):
    
    messages = spark.readStream.format('kafka') \
    .option('kafka.bootstrap.servers', '199.60.17.210:9092,199.60.17.193:9092') \
    .option('subscribe', input_topic).load()
    values = messages.select(messages['value'].cast('string'))
    
    split_col = functions.split(values['value'], ' ')
    values = values.withColumn('x', split_col.getItem(0).cast('float')) \
    .withColumn('y', split_col.getItem(1).cast('float')).drop('value')
    
    df = values.select('x','y', (values.x*values.y).alias('xy'), (values.x**2).alias('x2'))
    df.createOrReplaceTempView('df')
	
    sums = spark.sql('SELECT SUM(x) AS sum_x, SUM(y) AS sum_y, SUM(xy) AS sum_xy, \
    SUM(x2) AS sum_x2, COUNT(x) AS count FROM df')
    sums.createOrReplaceTempView('sums')
	
    slope = spark.sql('SELECT (sum_xy-sum_x*sum_y/count)/(sum_x2-sum_x*sum_x/count) AS slope, \
    sum_x, sum_y, count FROM sums')
    slope.createOrReplaceTempView('slope')
	
    df_final = spark.sql('SELECT slope, (sum_y-sum_x*slope)/count AS intercept FROM slope')
    stream = df_final.writeStream.format('console').option('truncate', False) \
    .outputMode('complete').start()
    stream.awaitTermination(600)
	
if __name__ == '__main__':
    input_topic = sys.argv[1]
    main(input_topic)