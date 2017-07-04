import os

from pyspark import SparkContext, SparkConf
from pyspark.sql import SQLContext
from elasticsearch import Elasticsearch


class Study(object):
    def __init__(self, id, bucket_url="data/"):
        print "Study: " + id
        self.id = id

        self.bucket_url = bucket_url
        print "Bucket URL: " + bucket_url
        self.bucket_dataset_url = bucket_url + 'datasets/'
        print "Bucket Dataset URL: " + self.bucket_dataset_url
        self.bucket_study_url = bucket_url + "studies/" + self.id + '/'
        print "Bucket Study URL: " + self.bucket_study_url

        # Spark
        spark_app = "SocioKnowledge"
        if os.environ.get('SPARK_APP') is not None:
            spark_app = os.environ.get('SPARK_APP')

        spark_master = "local[*]"
        if os.environ.get('SPARK_MASTER') is not None:
            spark_master = os.environ.get('SPARK_MASTER')

        spark_conf = SparkConf()
        spark_conf.setAppName(spark_app)
        spark_conf.setMaster(spark_master)
        self.sc = SparkContext(conf=spark_conf)

        # Elasticsearch
        self.es_host = "54.71.72.134"
        if os.environ.get('ES_HOST') is not None:
            self.es_host = os.environ.get('ES_HOST')

        self.es_port = "9200"
        if os.environ.get('ES_PORT') is not None:
            self.es_port = os.environ.get('ES_PORT')

        self.es = Elasticsearch([{'host': self.es_host, 'port': self.es_port}])

        # Mongo
        self.mongoURL = "mongodb://127.0.0.1/socioknowledge"
        if os.environ.get('MONGO_URL') is not None:
            self.mongo_url = os.environ.get('MONGO_URL')

        # SQL Context
        self.sqlc = SQLContext(self.sc)