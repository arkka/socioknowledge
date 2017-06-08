import json
import logging
from pyspark import StorageLevel
from pyspark.sql.functions import col, udf, lit, array, struct, create_map
from pyspark.sql.types import ArrayType, StructType, StructField, DoubleType, IntegerType, LongType, StringType, DateType, DataType
from pyspark.ml.feature import HashingTF, IDF, Tokenizer, StopWordsRemover

from unidecode import unidecode
from nltk.stem.snowball import SnowballStemmer
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory

from sklearn.metrics import jaccard_similarity_score

class Dictionary(object):

    # schemas
    schema = StructType([
        StructField("class", StringType()),
        StructField("term", StringType()),
        StructField("term_sense", StringType()),
        StructField("language", StringType()),
        StructField("source", StringType()),
        StructField("term_tokenized", StringType()),
        StructField("term_tokenized_length", StringType()),
        StructField("term_tokens", ArrayType(StringType())),
        StructField("term_tokens_num", IntegerType())
    ])

    compiled_schema = StructType([
        StructField("class", StringType()),
        StructField("terms", ArrayType(StringType()))
    ])

    def __init__(self, study):
        self.study = study
        self.dictionary = None
        self.dictionary_name = "dictionary"
        self.dictionary_index_name = "/".join([self.study.id + "-dictionaries", "dictionary"])
        self.raw_dictionary_index_name = "/".join([self.study.id + "-raw-dictionaries", "dictionary"])

        self.dictionary = self.study.sqlc.createDataFrame(study.sc.emptyRDD(), self.schema)

    def load(self, file_name=None):
        self.load_csv(file_name)
        return self

    def load_csv(self, file_name):
        if file_name is None:
            file_name = self.dictionary_name + "-seeds.csv"
            print "No file name specified to load dictionary. Using default file: " + self.study.bucket_study_url + file_name

        self.dictionary = self.study.sqlc.read.csv(
            self.study.bucket_study_url + file_name, header=True, mode="PERMISSIVE", schema=self.schema
        ).persist(StorageLevel.MEMORY_AND_DISK)
        return self


    def load_stopwords(self):
        schema = StructType([StructField("term", StringType())])
        en = self.study.sqlc.read.csv(self.study.bucket_dataset_url + "stopwords/stopwords-en.csv", header=False, schema=schema)
        id = self.study.sqlc.read.csv(self.study.bucket_dataset_url + "stopwords/stopwords-id.csv", header=False, schema=schema)
        nl = self.study.sqlc.read.csv(self.study.bucket_dataset_url + "stopwords/stopwords-nl.csv", header=False, schema=schema)
        stopwords = en.unionAll(id).unionAll(nl)
        self.stopwords = stopwords.rdd.map(lambda row: row.asDict()['term']).cache()
        return self

    def get(self):
        return self.dictionary

    def collect(self):
        return self.dictionary.collect()

    def count(self):
        return self.dictionary.count()

    def head(self, num=1):
        return self.dictionary.head(num)

    def show(self):
        return self.dictionary.show()

    def concat(self, dictionary):
        self.dictionary = self.dictionary.unionAll(dictionary.dictionary)
        return self

    def expand(self, dictionary_expansion):
        dictionary_expansion.expand()
        self.schema = dictionary_expansion.schema
        self.dictionary = dictionary_expansion.dictionary
        return self

    def tokenize(self, input_col='term'):
        df = self.dictionary

        lower_udf = udf(lambda term: unidecode(term.lower()), StringType())

        if input_col is not 'term_tokenized':
            df = df.withColumn("term_tokenized", lower_udf(df[input_col]))

        tokenizer = Tokenizer(inputCol="term_tokenized", outputCol="term_tokens")
        df = df.drop('term_tokens', 'term_tokens_num')
        df = tokenizer.transform(df)

        self.dictionary = df
        self.extract_tokens_stats()
        return self

    def filter_stopwords(self, stopwords=None):
        df = self.dictionary

        # load default stop words
        if stopwords is None:
            self.load_stopwords()
            stopwords = self.stopwords.collect()

        join_words = udf(lambda words: " ".join(words), StringType())
        remover = StopWordsRemover(inputCol="term_tokens", outputCol="term_tokens_filtered", stopWords=stopwords)

        df = df.drop('term_tokens_filtered')
        df = remover.transform(df)
        df = df.withColumn("term_tokens", df.term_tokens_filtered)
        df = df.withColumn("term_tokenized", join_words(df.term_tokens_filtered))
        df = df.drop('term_tokens_filtered')

        self.dictionary = df

        self.extract_tokens_stats()
        return self

    def filter_stemming(self, valid_language=['en', 'nl', 'id', 'ms']):
        df = self.dictionary

        # filter stemming
        def stem(language, words):
            if language in ['en', 'nl', 'id', 'ms']:
                stemmer = {
                    'en': SnowballStemmer("english"),
                    'nl': SnowballStemmer("dutch"),
                    'id': StemmerFactory().create_stemmer(),
                    'ms': StemmerFactory().create_stemmer(),
                }

                try:
                    stemmed_words = [stemmer[language].stem(word.encode("utf-8")) for word in words]
                    return stemmed_words
                except:
                    pass
            return words

        stem_words_udf = udf(lambda language, words: stem(language, words), ArrayType(StringType()))
        df = df.withColumn("term_tokens", stem_words_udf(df.language, df.term_tokens))

        # compute tokens
        join_words_udf = udf(lambda words: " ".join(words), StringType())

        df = df.withColumn("term_tokenized", join_words_udf(df.term_tokens))

        self.dictionary = df

        self.extract_tokens_stats()
        return self

    def extract_tokens_stats(self):
        df = self.dictionary

        count_udf = udf(lambda words: len(words), IntegerType())
        df = df.withColumn("term_tokens_num", count_udf(col("term_tokens")))
        df = df.withColumn("term_tokenized_length", count_udf(col("term_tokenized")))

        self.dictionary = df
        return self

    def compile(self, max_tokens_num=2):
        rdd = self.dictionary.rdd \
            .map(lambda row: row.asDict()) \
            .filter(lambda row: row['term_tokenized_length'] > 0 and row['term_tokens_num'] <= max_tokens_num) \
            .map(lambda row: (row['class'], [row['term_tokenized']])) \
            .reduceByKey(lambda x, y: x + y) \
            .map(lambda (x, y): {'class': x, 'terms': list(set(y))})
            
        self.compiled = self.study.sqlc.createDataFrame(rdd, self.compiled_schema).persist(StorageLevel.MEMORY_AND_DISK)
        return self

    def export_csv(self, file_name=None, repartition=True):
        if file_name is None:
            file_name = self.study.bucket_study_url + self.dictionary_name + ".csv"
            print "No file name specified to export dictionary. Using default file: " + file_name

        df = self.dictionary
        join_array_udf = udf(lambda x: "|".join(x), StringType())
        df = df.withColumn('term_tokens', join_array_udf(col("term_tokens")))

        if repartition:
            df = df.repartition(1)

        df.write.format('com.databricks.spark.csv').mode('overwrite').option("header", "true").save(file_name)

    def export_es(self):
        index_name = self.dictionary_index_name

        es_conf = {
            "es.nodes": "http://" + self.study.esHost,
            "es.port": "9200",
            "es.resource": index_name,
            "es.input.json": "true",
            "es.mapping.id": "class",
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
        with open('elasticsearch/dictionary-index-settings.json') as data_file:
            index_settings = json.load(data_file)

        self.study.es.indices.create(index=index_name.split("/")[0], body=index_settings)

        # index data
        self.compiled.rdd.map(lambda row: (row.asDict()['class'], json.dumps(row.asDict()))).saveAsNewAPIHadoopFile(
            path='-',
            outputFormatClass="org.elasticsearch.hadoop.mr.EsOutputFormat",
            keyClass="org.apache.hadoop.io.NullWritable",
            valueClass="org.elasticsearch.hadoop.mr.LinkedMapWritable",
            conf=es_conf)