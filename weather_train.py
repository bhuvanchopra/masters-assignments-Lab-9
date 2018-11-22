import sys
assert sys.version_info >= (3, 5) # make sure we have Python 3.5+

from pyspark.sql import SparkSession, functions, types
spark = SparkSession.builder.appName('weather prediction').getOrCreate()
spark.sparkContext.setLogLevel('WARN')
assert spark.version >= '2.3' # make sure we have Spark 2.3+

from pyspark.ml import Pipeline
from pyspark.ml.feature import StringIndexer, VectorAssembler, SQLTransformer
from pyspark.ml.regression import DecisionTreeRegressor, GBTRegressor
from pyspark.ml.evaluation import RegressionEvaluator

tmax_schema = types.StructType([
    types.StructField('station', types.StringType()),
    types.StructField('date', types.DateType()),
    types.StructField('latitude', types.FloatType()),
    types.StructField('longitude', types.FloatType()),
    types.StructField('elevation', types.FloatType()),
    types.StructField('tmax', types.FloatType()),
])

def main(input, model_file):
    
    data = spark.read.csv(input, schema=tmax_schema)
    train, validation = data.randomSplit([0.75, 0.25])
    train = train.cache()
    validation = validation.cache()
    
    sqlTrans = SQLTransformer(statement='SELECT today.*, dayofyear(today.date) AS \
    day_of_year, yesterday.tmax AS yesterday_tmax FROM __THIS__ as today INNER JOIN __THIS__ as \
    yesterday ON date_sub(today.date, 1) = yesterday.date AND today.station = yesterday.station')
    features_assembler = VectorAssembler(inputCols=['latitude', 'longitude', 'elevation', \
    'day_of_year', 'yesterday_tmax'], outputCol='features')
	
    regressor = GBTRegressor(featuresCol='features', labelCol='tmax')
	#regressor = DecisionTreeRegressor(featuresCol='features', labelCol='tmax') 
    weather_pipeline = Pipeline(stages=[sqlTrans, features_assembler, regressor])
    
    weather_model = weather_pipeline.fit(train)
    training_predictions = weather_model.transform(train)
    validation_predictions = weather_model.transform(validation)
        
    evaluator = RegressionEvaluator(predictionCol='prediction', labelCol='tmax')
    training_score = evaluator.evaluate(training_predictions)
    validation_score = evaluator.evaluate(validation_predictions)
	
    print('RMSE for weather_model on training data: %g' % (training_score, ))
    print('RMSE for weather_model on validation data: %g' % (validation_score, ))
    #training_predictions.show(20, False)
    #validation_predictions.show(20, False)
    weather_model.write().overwrite().save(model_file)
        
if __name__ == '__main__':
    input = sys.argv[1]
    model_file = sys.argv[2]
    main(input, model_file)