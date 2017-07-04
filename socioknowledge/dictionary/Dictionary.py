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
        StructField("term_raw", StringType()),
        StructField("term_length", IntegerType()),
        StructField("term_tokens", ArrayType(StringType())),
        StructField("term_tokens_num", IntegerType())
    ])

    compiled_schema = StructType([
        StructField("class", StringType()),
        StructField("terms", ArrayType(StringType()))
    ])

    def __init__(self, study):
        self.study = study
        self.df = None
        self.name = "dictionary"
        self.index_name = "/".join([self.study.id + "-dictionaries", "dictionary"])

        self.df = self.study.sqlc.createDataFrame(study.sc.emptyRDD(), self.schema)

    def import_csv(self, file_name=None):
        if file_name is None:
            file_name = self.name + "-seeds.csv"
            print "No file name specified to load dictionary. Using default file: " + self.study.bucket_study_url + file_name

        df = self.study.sqlc.read.csv(
            self.study.bucket_study_url + file_name, header=True, mode="PERMISSIVE", schema=self.schema
        )
        df = df.withColumn('term_raw', df['term'])
        self.df = df.cache()
        return self

    def load_stopwords(self,languages=['en','id','nl']):
        schema = StructType([StructField("term", StringType())])
        self.stopwords = self.study.sqlc.createDataFrame(self.study.sc.emptyRDD(), schema)

        for language in languages:
            self.stopwords = self.stopwords.unionAll(self.study.sqlc.read.csv(self.study.bucket_dataset_url + "stopwords/stopwords-" + language + ".csv", header=False, schema=schema))

        self.stopwords = self.stopwords.rdd.map(lambda row: row.asDict()['term']).cache()
        return self

    def concat(self, dictionary):
        self.df = self.df.unionAll(dictionary.df)
        return self

    def expand(self, dictionary_expansion):
        dictionary_expansion.expand()
        self.schema = dictionary_expansion.schema
        self.df = dictionary_expansion.df.persist(StorageLevel.MEMORY_AND_DISK)
        return self

    def tokenize(self, input_col='term_raw'):
        df = self.df

        if input_col is not 'term':
            lower_udf = udf(lambda term: unidecode(term.lower()) if term is not None else "", StringType())
            df = df.withColumn("term", lower_udf(df[input_col]))

        tokenizer = Tokenizer(inputCol="term", outputCol="term_tokens")
        df = df.drop('term_tokens', 'term_tokens_num')
        df = tokenizer.transform(df)

        self.df = df
        self.extract_tokens_stats()
        return self

    def filter_stopwords(self, stopwords=None):
        df = self.df

        # load default stop words
        if stopwords is None:
            self.load_stopwords()
            stopwords = self.stopwords.collect()

        join_words = udf(lambda words: " ".join(words), StringType())
        remover = StopWordsRemover(inputCol="term_tokens", outputCol="term_tokens_filtered", stopWords=stopwords)

        df = df.drop('term_tokens_filtered')
        df = remover.transform(df)
        df = df.withColumn("term_tokens", df.term_tokens_filtered)
        df = df.withColumn("term", join_words(df.term_tokens_filtered))
        df = df.drop('term_tokens_filtered')

        self.df = df

        self.extract_tokens_stats()
        return self

    def filter_stemming(self, valid_languages=['en', 'nl', 'id', 'ms']):
        df = self.df

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

        df = df.withColumn("term", join_words_udf(df.term_tokens))

        self.df = df

        self.extract_tokens_stats()
        return self

    def filter_term(self, min_term_length=0,max_tokens_num=2, valid_languages=['en', 'nl', 'id', 'ms']):
        # lang_const = array([lit(language) for language in valid_languages])
        df = self.df
        df = df.where((col("term_length") > lit(min_term_length)) & (col("term_tokens_num") <= lit(max_tokens_num)))
        df = df.where(col("language").isin(valid_languages))

        self.df = df
        return self

    def extract_tokens_stats(self):
        df = self.df

        count_udf = udf(lambda words: len(words), IntegerType())
        df = df.withColumn("term_tokens_num", count_udf(col("term_tokens")))
        df = df.withColumn("term_length", count_udf(col("term")))

        self.df = df
        return self

    def compile(self):
        rdd = self.df.rdd \
            .map(lambda row: row.asDict()) \
            .map(lambda row: (row['class'], [row['term']])) \
            .reduceByKey(lambda x, y: x + y) \
            .map(lambda (x, y): {'class': x, 'terms': list(set(y))})
        
        self.df = self.df.persist(StorageLevel.MEMORY_AND_DISK)    
        self.compiled = self.study.sqlc.createDataFrame(rdd, self.compiled_schema)
        return self

    def export_csv(self, file_name=None, repartition=True):
        if file_name is None:
            file_name = self.study.bucket_study_url + self.name + ".csv"
            print "No file name specified to export dictionary. Using default file: " + file_name

        df = self.df.select('class','term','term_sense','language','source','term_raw','term_length','term_expanded_conceptnet_num','term_expanded_conceptnet_relationship','term_expanded_conceptnet_concept','term_expanded_conceptnet_from','term_expanded_conceptnet_hop','term_expanded_conceptnet_weight','term_tokens_num')

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
        self.df.rdd.map(lambda row: (None, json.dumps(row.asDict()))).saveAsNewAPIHadoopFile(
            path='-',
            outputFormatClass="org.elasticsearch.hadoop.mr.EsOutputFormat",
            keyClass="org.apache.hadoop.io.NullWritable",
            valueClass="org.elasticsearch.hadoop.mr.LinkedMapWritable",
            conf=es_conf)