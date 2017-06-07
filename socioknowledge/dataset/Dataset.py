import json

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
        StructField("tokens", ArrayType(StringType()))
    ])

    def __init__(self, study, dataset=None):
        self.study = study

        if dataset is None:
            self.dataset = self.study.sqlc.createDataFrame(study.sc.emptyRDD(), self.schema)
        else:
            self.dataset = dataset

        self.dataset_name = "dataset"
        self.dataset_index_name = "/".join([self.study.id + "-datasets", "dataset"])

    def count(self):
        return self.dataset.count()

    def head(self, num=1):
        return self.dataset.head(num)

    def show(self):
        return self.dataset.show()

    def concat(self, dataset):
        self.dataset = self.dataset.unionAll(dataset.dataset)
        return self

    def transform_binomial(self, labels=[]):
        labels_const = array([array([lit(m) for m in l]) for l in labels])
        dataset_df = self.dataset

        print labels_const

        def binomial(label, labels):
            if label in labels[0]:
                return 0
            elif label in labels[1]:
                return 1

        binomial_udf = udf(lambda c, labels: binomial(c, labels), IntegerType())
        dataset_df = dataset_df.withColumn("label", binomial_udf(dataset_df['class'], labels_const))
        return dataset_df

    def extract_tf_idf(self, num_features=20):
        dataset_df = self.dataset

        hashing_tf = HashingTF(inputCol="tokens", outputCol="raw_features", numFeatures=num_features)
        # hashing_tf = HashingTF(inputCol="tokens", outputCol="raw_features")
        featurized_data = hashing_tf.transform(dataset_df)

        idf = IDF(inputCol="raw_features", outputCol="features")
        idf_model = idf.fit(featurized_data)
        result = idf_model.transform(featurized_data)

        # remove raw features
        result = result.drop('raw_features')

        self.dataset = result
        return self

    def extract_pca(self, k=3):
        dataset_df = self.dataset

        pca = PCA(k=k, inputCol="features", outputCol="pca_features")
        pca_model = pca.fit(dataset_df)

        result = pca_model.transform(dataset_df)
        self.dataset = result
        return self

    def split_training_test(self, training_ratio=0.7, test_set=0.3):
        dataset_df = self.dataset
        training_data, test_data = dataset_df.randomSplit([training_ratio, test_set])
        self.training = training_data
        self.test = test_data
        return self


