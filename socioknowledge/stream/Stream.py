import json
import re

from pyspark import StorageLevel
from pyspark.sql.functions import col, udf, lit, array, struct, create_map, split, explode, size, length
from pyspark.sql.types import ArrayType, StructType, StructField, DoubleType, IntegerType, LongType, StringType, \
    DateType, DataType, BooleanType
from pyspark.ml.feature import HashingTF, IDF, Tokenizer, StopWordsRemover, PCA
from pyspark.ml.linalg import Vectors

from unidecode import unidecode
from nltk.stem.snowball import SnowballStemmer
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory

from sklearn.metrics import jaccard_similarity_score

from ..dataset import Dataset

class Stream(object):
    # schemas
    schema = StructType([
        # stream
        StructField("id", StringType()),
        StructField("provider", StringType()),
        StructField("timestamp", LongType()),
        StructField("text", StringType()),
        StructField("language", StringType()),
        StructField("link", StringType()),

        # actor
        StructField("actor_id", StringType()),
        StructField("actor_name", StringType()),
        StructField("actor_link", StringType()),

        # coordinate
        StructField("coordinate_latitude", DoubleType()),
        StructField("coordinate_longitude", DoubleType()),

        # secondary coordinate
        StructField("place_id", StringType()),
        StructField("place_link", StringType()),
        StructField("place_name", StringType()),
        StructField("place_type", StringType()),
        StructField("place_country", StringType()),
        StructField("place_latitude", DoubleType()),
        StructField("place_longitude", DoubleType()),

        StructField("text_raw", StringType()),
        
        # external legacy
        StructField("data", StringType()),

        # computed field
        StructField("text_tokens", ArrayType(StringType())),
        StructField("text_tokens_num", IntegerType()),
        StructField("text_length", IntegerType()),

        StructField("dictionary_classes", ArrayType(StringType())),

    ])

    def __init__(self, study, name="stream"):
        self.study = study
        self.df = None
        self.name = name
        self.index_name = "/".join([self.study.id + "-streams", "stream"])

        self.df = self.study.sqlc.createDataFrame(study.sc.emptyRDD(), self.schema)


    def concat(self, stream):
        self.df = self.df.unionAll(stream.df)
        return self

    def cache(self):
        self.df = self.df.cache()
        return self

    def import_csv(self, file_name=None):
        if file_name is None:
            file_name = self.name + ".csv"
            print "No file name specified to load stream. Using default file: " + self.study.bucket_study_url + file_name

        df = self.study.sqlc.read.csv(
            self.study.bucket_study_url + file_name, header=True, mode="PERMISSIVE", schema=self.schema
        )

        # tokens_string_udf = udf(lambda tokens: tokens.split('|') if tokens is not None else [], ArrayType(StringType()))
        # df = df.withColumn('text_tokens', tokens_string_udf(df.text_tokens))
        # df = df.withColumn('text_raw', df['text'])
        self.df = df.cache()

        return self

    def load_stopwords(self,languages=['en','id','nl']):
        schema = StructType([StructField("term", StringType())])


        self.stopwords = self.study.sqlc.createDataFrame(self.study.sc.emptyRDD(), schema)

        for language in languages:
            self.stopwords = self.stopwords.unionAll(self.study.sqlc.read.csv(self.study.bucket_dataset_url + "stopwords/stopwords-" + language + ".csv", header=False, schema=schema))

        self.stopwords = self.stopwords.rdd.map(lambda row: row.asDict()['term']).cache()
        return self

    def tokenize(self, input_col='text_raw'):
        df = self.df

        if input_col is not 'text':
            # lower
            lower_udf = udf(lambda text: unidecode(text.lower()) if text is not None else "", StringType())
            df = df.withColumn("text", lower_udf(df[input_col]))

        tokenizer = Tokenizer(inputCol="text", outputCol="text_tokens")
        df = df.drop('text_tokens', 'text_tokens_num')
        df = tokenizer.transform(df)

        self.df = df
        self.extract_tokens_stats()
        return self

    def filter_standard(self, min_text_length=10, min_token_length=3):
        def filter_standard(tokens):
            filtered_tokens = []
            for token in tokens:
                # remove sign char on token
                stripped_token = re.sub('[^\w\s]', '', token)

                # remove token with user mention and with length < n and link
                if len(stripped_token) >= min_token_length and not token.startswith("@") and not token.startswith("http"):
                    filtered_tokens.append(stripped_token)

            return filtered_tokens

        df = self.df

        # filter min text
        df = df.filter(length(col("text")) > min_text_length)


        # filter valid token
        filter_standard_udf = udf(lambda tokens: filter_standard(tokens), ArrayType(StringType()))
        df = df.withColumn("text_tokens", filter_standard_udf(df.text_tokens))

        self.df = df
        self.extract_tokens_stats()
        return self

    def filter_stopwords(self, stopwords=None, languages=['en','id','nl']):
        df = self.df

        # load default stop words
        if stopwords is None:
            self.load_stopwords(languages=['en','id','nl'])
            stopwords = self.stopwords.collect()

        join_words = udf(lambda words: " ".join(words), StringType())
        remover = StopWordsRemover(inputCol="text_tokens", outputCol="text_tokens_filtered", stopWords=stopwords)

        df = df.drop('text_tokens_filtered')
        df = remover.transform(df)
        df = df.withColumn("text_tokens", df.text_tokens_filtered)
        df = df.withColumn("text", join_words(df.text_tokens_filtered))
        df = df.drop('text_tokens_filtered')

        self.df = df

        self.extract_tokens_stats()
        return self

    def filter_shingle(self, max_n=2):
        df = self.df

        # generate shingle
        def generate_shingles(tokens, max_n):
            merged_shingles = tokens
            for n in range(2, max_n + 1):
                shingles = []

                for i in range(0, len(tokens) - n + 1):
                    shingle = " ".join(tokens[i:i + n])
                    shingles.append(shingle)

                merged_shingles = merged_shingles + shingles

            return merged_shingles

        generate_shingles_udf = udf(lambda tokens, max_n: generate_shingles(tokens, max_n), ArrayType(StringType()))
        df = df.withColumn("text_tokens", generate_shingles_udf(df.text_tokens, lit(max_n)))

        self.df = df

        self.extract_tokens_stats()
        return self

    def filter_stemming(self, languages=['en', 'nl', 'id', 'ms']):
        df = self.df

        # filter stemming
        def stem(language, languages, words):
            if language in languages:
                stemmer = {
                    'en': SnowballStemmer("english"),
                    'nl': SnowballStemmer("dutch"),
                    'id': StemmerFactory().create_stemmer(),
                    'ms': StemmerFactory().create_stemmer(),
                }

                try:
                    stemmed_words = [stemmer[language].stem(word) for word in words]
                    return stemmed_words
                except:
                    pass
            return words

        stem_words_udf = udf(lambda language, languages, words: stem(language, languages, words), ArrayType(StringType()))
        df = df.withColumn("text_tokens", stem_words_udf(df.language, array([lit(l) for l in languages]), df.text_tokens))

        # compute tokens
        join_words_udf = udf(lambda words: " ".join(words), StringType())

        df = df.withColumn("text", join_words_udf(df.text_tokens))

        self.df = df

        self.extract_tokens_stats()
        return self

    def extract_tokens_stats(self):
        df = self.df

        count_udf = udf(lambda words: len(words), IntegerType())
        df = df.withColumn("text_tokens_num", count_udf(col("text_tokens")))
        df = df.withColumn("text_length", count_udf(col("text")))

        self.df = df
        return self

    def export_hdfs(self, file_name='streams.csv'):
        df = self.df

        df.write \
            .format("com.databricks.spark.csv") \
            .option("header", "true") \
            .save("hdfs://" + file_name)
        return self

    def export_csv(self, file_name='streams.csv'):
        df = self.df

        df.repartition(1)\
            .write \
            .format("com.databricks.spark.csv") \
            .option("header", "true") \
            .save("file:///" + self.study.bucket_study_url + file_name) \

        return self

    def match_dictionary(self, dictionary, partial_match_min_prefix=4, partial_match_min_similarity=0.75):
        stream_rdd = self.df.rdd.map(lambda row: row.asDict()).repartition(4).cache()
        dictionary_rdd = self.study.sc.parallelize([{ k: v for d in dictionary.compiled.rdd.map(lambda row: {row['class']:row['terms']}).collect() for k, v in d.items() }]).cache()

        stream_dictionary_rdd = stream_rdd.cartesian(dictionary_rdd)

        def match(stream, dictionary):
            stream_matches = []
            stream_exact_matches = []
            stream_partial_matches = []
            
            tokens = stream['text_tokens']
 
            for key, terms in dictionary.iteritems():
                # exact match
                exact_matches = list(set(tokens) & set(terms))
                # stream['dictionary_' + key + '_exact_matches'] = exact_matches
                # stream['dictionary_' + key + '_exact_matches_num'] = len(exact_matches)
                # stream_exact_matches = stream_exact_matches + exact_matches
                
                # partial match
                partial_matches = []
                for token in tokens:
                    # minimun token length
                    if len(token) > partial_match_min_prefix:
                        for term in terms:
                            term_chars = list(term)
                            term_chars_num = len(term_chars)
                            token_chars = list(token)
                            token_chars_num = len(token_chars)

                            if (term_chars_num > token_chars_num):
                                [token_chars.append("") for i in range(0, term_chars_num - token_chars_num)]
                            else:
                                [term_chars.append("") for i in range(0, token_chars_num - term_chars_num)]

                            similarity = float(jaccard_similarity_score(term_chars, token_chars))
                            if similarity >= partial_match_min_similarity:
                                partial_matches.append(term)    
                                
                # stream['dictionary_' + key + '_partial_matches'] = partial_matches
                # stream['dictionary_' + key + '_partial_matches_num'] = len(partial_matches)
                # stream_partial_matches = stream_partial_matches + partial_matches
                
                # label class        
                if len(exact_matches) > 0 or len(partial_matches) > 0:
                    stream_matches.append(key)

            if len(stream_matches) == 0:
                stream_matches.append('none')

            stream['dictionary_classes'] = stream_matches

            return stream
        stream_dictionary_rdd = stream_dictionary_rdd.map(lambda (stream, dictionary): match(stream, dictionary))
        stream_df = self.study.sqlc.createDataFrame(stream_dictionary_rdd, self.schema)
        self.df = stream_df.persist(StorageLevel.MEMORY_AND_DISK)
        return self

    def export_dataset(self):
        stream_df = self.df
        # explode
        dataset_df = stream_df.select(col("id"), explode(col("dictionary_classes")).alias("label"), col("text"), col("text_tokens").alias("tokens")).dropDuplicates()

        dataset = Dataset(self.study, dataset_df)

        return dataset

    def export_csv(self, file_name=None, repartition=True):
        if file_name is None:
            file_name = self.name + ".csv"
            print "No file name specified to export stream. Using default file: " + self.study.bucket_study_url + file_name

        df = self.df #.select('id','provider','timestamp','text','language','link','actor_id','actor_name','actor_link','coordinate_latitude','coordinate_longitude','place_id','place_link','place_name','place_type','place_country','place_latitude','place_longitude')
        
        tokens_string_udf = udf(lambda tokens: "|".join(tokens), StringType())

        df = df.withColumn('text_tokens', tokens_string_udf(df.text_tokens))
        if repartition:
            df = df.repartition(1)

        df.write.format('com.databricks.spark.csv').mode('overwrite').option("header", "true").save(self.study.bucket_study_url + file_name)

    def export_es(self):
        index_name = self.index_name

        es_conf = {
            "es.nodes": "http://" + self.study.es_host,
            "es.port": self.study.es_port,
            "es.resource": index_name,
            "es.input.json": "true",
            "es.mapping.id": "id",
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
        self.df.rdd.map(lambda row: (row.asDict()['id'], json.dumps(row.asDict()))).saveAsNewAPIHadoopFile(
            path='-',
            outputFormatClass="org.elasticsearch.hadoop.mr.EsOutputFormat",
            keyClass="org.apache.hadoop.io.NullWritable",
            valueClass="org.elasticsearch.hadoop.mr.LinkedMapWritable",
            conf=es_conf)
