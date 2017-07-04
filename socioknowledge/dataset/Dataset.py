import json
import numpy as np

from pyspark.sql.functions import col, udf, lit, array, struct, create_map, split, explode
from pyspark.sql.types import ArrayType, StructType, StructField, DoubleType, IntegerType, LongType, StringType, \
    DateType, DataType, BooleanType
from pyspark.ml.feature import HashingTF, IDF, Tokenizer, StopWordsRemover, PCA
from pyspark.ml.linalg import Vectors

from unidecode import unidecode
from nltk.stem.snowball import SnowballStemmer
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory

from sklearn.metrics import jaccard_similarity_score



class Dataset(object):
    # schemas
    schema = StructType([
        # stream
        StructField("id", StringType()),
        StructField("label", IntegerType()),
        StructField("text", StringType()),
        StructField("tokens", ArrayType(StringType())),

    ])

    def __init__(self, study, dataset=None):
        self.study = study

        if dataset is None:
            self.df = self.study.sqlc.createDataFrame(study.sc.emptyRDD(), self.schema)
        else:
            self.df = dataset

        self.name = "dataset"
        self.index_name = "/".join([self.study.id + "-datasets", "dataset"])

    def concat(self, dataset):
        self.df = self.df.unionAll(dataset.df)
        return self

    def classify_label(self, labels=[]):
        labels_const = array([array([lit(m) for m in l]) for l in labels])
        dataset_df = self.df

        def classify(label, labels):
            label_idx = len(labels) - 1
            for idx, val in enumerate(labels):
                if label in val:
                    label_idx = idx

            return label_idx
                
            # if label in labels:
            #     return 1
            # else:
            #     return 0

        classify_udf = udf(lambda c, labels: classify(c, labels), IntegerType())
        dataset_df = dataset_df.withColumn("label", classify_udf(dataset_df['label'], labels_const))
        return dataset_df

    def extract_tf_idf(self, num_features=2^20):
        dataset_df = self.df

        hashing_tf = HashingTF(inputCol="tokens", outputCol="tfidf_raw_features", numFeatures=num_features)
        # hashing_tf = HashingTF(inputCol="tokens", outputCol="raw_features")
        featurized_data = hashing_tf.transform(dataset_df)

        idf = IDF(inputCol="tfidf_raw_features", outputCol="tfidf_features")
        idf_model = idf.fit(featurized_data)
        result = idf_model.transform(featurized_data)

        # remove raw features
        result = result.drop('tfidf_raw_features')

        self.df = result
        return self

    def extract_pca(self, k=3):
        dataset_df = self.df

        pca = PCA(k=k, inputCol="features", outputCol="pca_features")
        pca_model = pca.fit(dataset_df)

        result = pca_model.transform(dataset_df)
        self.df = result
        return self

    def split_training_test(self, training_ratio=0.7, test_set=0.3):
        dataset_df = self.df
        training_data, test_data = dataset_df.randomSplit([training_ratio, test_set])
        self.training = training_data
        self.test = test_data
        return self

    def export_csv(self, file_name=None, repartition=True):
        if file_name is None:
            file_name = self.study.bucket_study_url + self.name + ".csv"
            print "No file name specified to export dataset. Using default file: " + file_name

        df = self.df
        join_array_udf = udf(lambda x: "|".join(x), StringType())
        df = df.withColumn('tokens', join_array_udf(col("tokens")))

        df = self.df.select('id','label','tokens')
        
        if repartition:
            df = df.repartition(1)

        df.write.format('com.databricks.spark.csv').mode('overwrite').option("header", "true").save(file_name)

    def export_es(self):
        index_name = self.index_name

        es_conf = {
            "es.nodes": "http://" + self.study.es_host,
            "es.port": self.study.es_port,
            "es.resource": index_name,
            "es.input.json": "true",
            "es.write.ignore_exception": "true",
            "es.read.ignore_exception": "true",
            "es.nodes.client.only": "false",
            "es.nodes.discovery": "false",
            "es.index.auto.create": "true",
            "es.nodes.wan.only": "true"
        }

        # delete existing index
        self.study.es.indices.delete(index=index_name.split("/")[0], ignore=[400, 404])

        # create new index using given settings and mappings
        # with open('elasticsearch/dictionary-index-settings.json') as data_file:
        #     index_settings = json.load(data_file)

        # self.study.es.indices.create(index=index_name.split("/")[0], body=index_settings)

        # index data


        def convert_sparse_vector(row):
            row = row.asDict()
            row['features'] = row['features'].toArray().tolist()
            return row

        self.df.rdd.map(lambda row: (None, json.dumps(convert_sparse_vector(row)))).saveAsNewAPIHadoopFile(
            path='-',
            outputFormatClass="org.elasticsearch.hadoop.mr.EsOutputFormat",
            keyClass="org.apache.hadoop.io.NullWritable",
            valueClass="org.elasticsearch.hadoop.mr.LinkedMapWritable",
            conf=es_conf)


