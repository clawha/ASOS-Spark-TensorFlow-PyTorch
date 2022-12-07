from pyspark.sql import SparkSession
from pyspark.ml.feature import VectorAssembler
from pyspark.sql.functions import rand
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.ml.pipeline import Pipeline, PipelineModel
from sparktorch import PysparkPipelineWrapper
from sparktorch import SparkTorch, serialize_torch_obj
import torch
import torch.nn as nn


EPOCHS = 500
MASTER = 'local[2]'
LR = 0.001

if __name__ == '__main__':
    spark = SparkSession.builder \
        .appName("apachespark-pytorch") \
        .master(MASTER) \
        .getOrCreate()

    df = spark.read.option("inferSchema", "true").csv('mnist_train.csv')

    network = nn.Sequential(
        nn.Linear(784, 256),
        nn.ReLU(),
        nn.Linear(256, 256),
        nn.ReLU(),
        nn.Linear(256, 10)
    )
    
    vector_assembler = VectorAssembler(inputCols=df.columns[1:785], outputCol='features')
    
    torch_obj = serialize_torch_obj(
        model=network,
        criterion=nn.CrossEntropyLoss(),
        optimizer=torch.optim.Adam,
        lr=0.001
    )
    
    spark_model = SparkTorch(
        inputCol='features',
        labelCol='_c0',
        predictionCol='predictions',
        torchObj=torch_obj,
        iters=EPOCHS,
        verbose=1,
        earlyStopPatience=40,
        validationPct=0.2
    )

    p = Pipeline(stages=[vector_assembler, spark_model]).fit(df)

    predictions = p.transform(df).persist()
    evaluator = MulticlassClassificationEvaluator(labelCol="_c0", predictionCol="predictions", metricName="accuracy")
    accuracy = evaluator.evaluate(predictions)
    print("Accuracy = %g" % accuracy)

