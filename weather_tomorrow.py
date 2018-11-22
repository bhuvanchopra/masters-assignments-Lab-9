import sys
assert sys.version_info >= (3, 5) # make sure we have Python 3.5+

import datetime
from pyspark.sql import SparkSession, functions, types
spark = SparkSession.builder.appName('weather tom').getOrCreate()
assert spark.version >= '2.3' # make sure we have Spark 2.3+
spark.sparkContext.setLogLevel('WARN')

from pyspark.ml import PipelineModel, linalg

weather_predict = types.StructType([
    types.StructField('station', types.StringType()),
    types.StructField('date', types.DateType()),
    types.StructField('latitude', types.FloatType()),
    types.StructField('longitude', types.FloatType()),
    types.StructField('elevation', types.FloatType()),
    types.StructField('tmax', types.FloatType())])

def weather_tom(model_file):
    
    data = [('bhuvanSFU', datetime.date(2018,11,13), 49.2771, -122.9146, 330.0, 12.0), \
    ('bhuvanSFU', datetime.date(2018,11,12), 49.2771, -122.9146, 330.0, 12.0)]
    weather_pred = spark.createDataFrame(data, schema=weather_predict)
	   
    model = PipelineModel.load(model_file)
    
    prediction = model.transform(weather_pred)
    
    print('Predicted tmax tomorrow:', round(prediction.select('prediction').collect()[0][0], 2))

if __name__ == '__main__':
    model_file = sys.argv[1]
    weather_tom(model_file)
