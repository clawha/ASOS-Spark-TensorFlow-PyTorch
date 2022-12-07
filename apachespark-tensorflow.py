from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.utils import to_categorical
from elephas.spark_model import SparkModel
from elephas.utils.rdd_utils import to_simple_rdd
from pyspark import SparkContext, SparkConf


batch_size = 64
nb_classes = 10
epochs = 10

def sparkContext():
  conf = SparkConf().setAppName('SPARKTENSORFLOW').setMaster('local[*]')
  sc = SparkContext(conf=conf)
  return sc

def prepareData():
  (training_images, training_labels), (test_images, test_labels) = mnist.load_data()
  training_images = training_images.reshape(60000, 28*28)
  test_images = test_images.reshape(10000, 28*28)
  training_images = training_images.astype("float32")
  test_images = test_images.astype("float32")
  training_images = training_images / 255
  test_images = test_images / 255
  training_labels = to_categorical(training_labels, nb_classes)
  test_labels = to_categorical(test_labels, nb_classes)

  return training_images, training_labels, test_images, test_labels


def neuralNetwork(learning_rate):
  model = Sequential()
  model.add(Dense(32, input_dim=28*28))
  model.add(Activation('relu'))
  model.add(Dropout(0.1))
  model.add(Dense(64))
  model.add(Activation('relu'))
  model.add(Dropout(0.1))
  model.add(Dense(10))
  model.add(Activation('softmax'))
  optimizer = SGD(lr=learning_rate)
  model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

  return model

training_images, training_labels, test_images, test_labels = prepareData()

resilient_distributed_dataset = to_simple_rdd(sparkContext(), training_images, training_labels)

spark_model = SparkModel(neuralNetwork(0.1), frequency='epoch', mode='asynchronous')
spark_model.fit(resilient_distributed_dataset, epochs=epochs, batch_size=batch_size, verbose=0, validation_split=0.1)
score = spark_model.evaluate(test_images, test_labels, verbose=2)

print('Test accuracy:', score[1])
print('Loss:', score[0])