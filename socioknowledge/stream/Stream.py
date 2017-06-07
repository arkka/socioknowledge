import json
import re

from pyspark.sql.functions import col, udf, lit, array, struct, create_map, split, explode
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

        # external legacy
        StructField("data", StringType())
    ])

    def __init__(self, study):
        self.study = study
        self.stream = None
        self.stream_name = "stream"
        self.stream_index_name = "/".join([self.study.id + "-streams", "stream"])

        self.stream = self.study.sqlc.createDataFrame(study.sc.emptyRDD(), self.schema)

    def count(self):
        return self.stream.count()

    def head(self, num=1):
        return self.stream.head(num)

    def show(self):
        return self.stream.show()

    def concat(self, stream):
        self.stream = self.stream.unionAll(stream.stream)
        return self

    def load_stopwords(self):
        schema = StructType([StructField("term", StringType())])
        en = self.study.sqlc.read.csv(self.study.bucket_dataset_url + "stopwords/stopwords-en.csv", header=False,
                                      schema=schema)
        id = self.study.sqlc.read.csv(self.study.bucket_dataset_url + "stopwords/stopwords-id.csv", header=False,
                                      schema=schema)
        nl = self.study.sqlc.read.csv(self.study.bucket_dataset_url + "stopwords/stopwords-nl.csv", header=False,
                                      schema=schema)
        stopwords = en.unionAll(id).unionAll(nl)
        self.stopwords = stopwords.rdd.map(lambda row: row.asDict()['term']).collect()
        return self

    def tokenize(self, input_col='text'):
        df = self.stream

        # lower
        lower_udf = udf(lambda term: unidecode(term.lower()), StringType())
        if input_col is not 'text_tokenized':
            df = df.withColumn("text_tokenized", lower_udf(df[input_col]))

        tokenizer = Tokenizer(inputCol="text_tokenized", outputCol="text_tokens")
        df = df.drop('text_tokens', 'text_tokens_num')
        df = tokenizer.transform(df)

        self.stream = df
        self.extract_tokens_stats()
        return self

    def filter_standard(self, min_token_length=3):
        def filter_standard(tokens):
            filtered_tokens = []
            for token in tokens:
                # remove sign char on token
                stripped_token = re.sub('[^\w\s]', '', token)

                # remove token with user mention and with length < n and link
                if len(stripped_token) >= min_token_length and not token.startswith("@") and not token.startswith("http"):
                    filtered_tokens.append(stripped_token)

            return filtered_tokens

        df = self.stream

        filter_standard_udf = udf(lambda tokens: filter_standard(tokens), ArrayType(StringType()))
        df = df.withColumn("text_tokens", filter_standard_udf(df.text_tokens))

        self.stream = df
        self.extract_tokens_stats()
        return self

    def filter_stopwords(self, stopwords=None):
        df = self.stream

        # load default stop words
        if stopwords is None:
            self.load_stopwords()
            stopwords = self.stopwords

        join_words = udf(lambda words: " ".join(words), StringType())
        remover = StopWordsRemover(inputCol="text_tokens", outputCol="text_tokens_filtered", stopWords=stopwords)

        df = df.drop('text_tokens_filtered')
        df = remover.transform(df)
        df = df.withColumn("text_tokens", df.text_tokens_filtered)
        df = df.withColumn("text_tokenized", join_words(df.text_tokens_filtered))
        df = df.drop('text_tokens_filtered')

        self.stream = df

        self.extract_tokens_stats()
        return self

    def filter_shingle(self, max_n=2):
        df = self.stream

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

        self.stream = df

        self.extract_tokens_stats()
        return self

    def filter_stemming(self, valid_language=['en', 'nl', 'id', 'ms']):
        df = self.stream

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
                    stemmed_words = [stemmer[language].stem(word) for word in words]
                    return stemmed_words
                except:
                    pass
            return words

        stem_words_udf = udf(lambda language, words: stem(language, words), ArrayType(StringType()))
        df = df.withColumn("text_tokens", stem_words_udf(df.language, df.text_tokens))

        # compute tokens
        join_words_udf = udf(lambda words: " ".join(words), StringType())

        df = df.withColumn("text_tokenized", join_words_udf(df.text_tokens))

        self.stream = df

        self.extract_tokens_stats()
        return self

    def extract_tokens_stats(self):
        df = self.stream

        count_udf = udf(lambda words: len(words), IntegerType())
        df = df.withColumn("text_tokens_num", count_udf(col("text_tokens")))
        df = df.withColumn("text_tokenized_length", count_udf(col("text_tokenized")))

        self.stream = df
        return self

    def export_hdfs(self, file_name='streams.csv'):
        df = self.stream

        df.write \
            .format("com.databricks.spark.csv") \
            .option("header", "true") \
            .save("hdfs://" + file_name)
        return self

    def export_csv(self, file_name='streams.csv'):
        df = self.stream

        df.repartition(1)\
            .write \
            .format("com.databricks.spark.csv") \
            .option("header", "true") \
            .save("file:///" + self.study.bucket_study_url + file_name) \

        return self

    def export_es(self, indexName=None):
        if indexName is None:
            indexName = self.streamIndexName

        es_conf = {
            "es.nodes": "http://" + self.study.esHost,
            "es.port": "9200",
            "es.resource": indexName,
            "es.query": '{  "query": { "match_all": {} } }',
            "es.write.ignore_exception": "true",
            "es.read.ignore_exception": "true",
            "es.nodes.client.only": "false",
            "es.nodes.discovery": "false",
            "es.index.auto.create": "true",
            "es.nodes.wan.only": "true"
        }

        # delete index
        #         self.study.es.indices.delete(index=self.streamIndexName.split("/")[0], ignore=[400, 404])

        # create new index using given settings and mappings
        with open('elasticsearch/stream-index-settings.json') as data_file:
            indexSettings = json.load(data_file)

        self.study.es.indices.create(index=self.streamIndexName.split("/")[0], body=indexSettings)

        # index data
        self.stream.rdd.map(lambda row: (None, row.asDict(recursive=True))).saveAsNewAPIHadoopFile(
            path='-',
            outputFormatClass="org.elasticsearch.hadoop.mr.EsOutputFormat",
            keyClass="org.apache.hadoop.io.NullWritable",
            valueClass="org.elasticsearch.hadoop.mr.LinkedMapWritable",
            conf=es_conf)

    def match_dictionary(self, dictionary, partial_match_min_prefix=4, partial_match_min_similarity=0.75):
        def exact_match(dictionary_terms, tokens):
            return list(set(dictionary_terms) & set(tokens))

        def partial_match(dictionary_terms, tokens, partial_match_min_prefix, partial_match_min_similarity):
            partial_matches = []

            for token in tokens:

                # minimun token length
                if len(token) > partial_match_min_prefix:
                    for term in dictionary_terms:

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
            return partial_matches

        def merge_match(exact, partial):
            matches = []
            matches = matches + exact
            matches = matches + partial
            return list(set(matches))

        def label_match(label, matches_num, c):
            if matches_num > 0:
                label.append(c)
            return list(set(label))

        def label_match(label, matches_num, c):
            if matches_num > 0:
                label.append(c)
            return list(set(label))

        def label_unmatch(matches):
            if len(matches) == 0:
                return ['none']
            return matches


        count_udf = udf(lambda tokens: len(tokens), IntegerType())
        exact_match_udf = udf(exact_match, ArrayType(StringType()))
        partial_match_udf = udf(partial_match, ArrayType(StringType()))
        merge_match_udf = udf(merge_match, ArrayType(StringType()))
        label_match_udf = udf(label_match, ArrayType(StringType()))
        label_unmatch_udf = udf(label_unmatch, ArrayType(StringType()))

        stream_df = self.stream
        dictionary_df = dictionary.compiled
        dictionary_data = dictionary_df.rdd.map(lambda row: row.asDict()).collect()
        for d in dictionary_data:
            terms = array([lit(term) for term in d['terms']])
            stream_df = stream_df.withColumn('dictionary_' + d['class'], terms)

            # exact match
            stream_df = stream_df.withColumn('dictionary_' + d['class'] + '_exact_matches',
                                             exact_match_udf(terms, stream_df['text_tokens']))
            stream_df = stream_df.withColumn('dictionary_' + d['class'] + '_exact_matches_num',
                                             count_udf(stream_df['dictionary_' + d['class'] + '_exact_matches']))

            # partial match
            stream_df = stream_df.withColumn('dictionary_' + d['class'] + '_partial_matches',
                                             partial_match_udf(terms, stream_df['text_tokens'],
                                                               lit(partial_match_min_prefix),
                                                               lit(partial_match_min_similarity)))
            stream_df = stream_df.withColumn('dictionary_' + d['class'] + '_partial_matches_num',
                                             count_udf(stream_df['dictionary_' + d['class'] + '_partial_matches']))

            # total match
            stream_df = stream_df.withColumn('dictionary_' + d['class'] + '_matches',
                                             merge_match_udf(stream_df['dictionary_' + d['class'] + '_exact_matches'], stream_df[
                                                 'dictionary_' + d['class'] + '_partial_matches']))
            stream_df = stream_df.withColumn('dictionary_' + d['class'] + '_matches_num',
                                             count_udf(stream_df['dictionary_' + d['class'] + '_matches']))

            # match classes
            if not 'dictionary_classes' in stream_df.columns:
                stream_df = stream_df.withColumn('dictionary_classes', array())
            stream_df = stream_df.withColumn('dictionary_classes', label_match_udf(stream_df['dictionary_classes'], stream_df['dictionary_' + d['class'] + '_matches_num'], lit(d['class'])))

        stream_df = stream_df.withColumn('dictionary_classes', label_unmatch_udf(stream_df['dictionary_classes']))

        self.stream = stream_df
        return self

    def export_dataset(self):
        stream_df = self.stream
        # explode
        dataset_df = stream_df.select(col("id"), explode(col("dictionary_classes")).alias("class"), col("text_tokens").alias("tokens"))

        dataset = Dataset(self.study, dataset_df)

        return dataset

