from DictionaryExpansion import DictionaryExpansion

import json
from pyspark import StorageLevel
from pyspark.sql.types import ArrayType, StructType, StructField, DoubleType, IntegerType, LongType, StringType, DateType, DataType


class ConceptNetDictionaryExpansion(DictionaryExpansion):
    # schemas
    schema = StructType([
        StructField("class", StringType()),
        StructField("term", StringType()),
        StructField("term_sense", StringType()),
        StructField("language", StringType()),
        StructField("source", StringType()),
        StructField("term_raw", StringType()),
        StructField("term_length", StringType()),
        StructField("term_tokens", ArrayType(StringType())),
        StructField("term_tokens_num", IntegerType()),
        StructField("term_expanded_conceptnet_num", IntegerType()),
        StructField("term_expanded_conceptnet_relationship", StringType()),
        StructField("term_expanded_conceptnet_concept", StringType()),
        StructField("term_expanded_conceptnet_from", StringType()),
        StructField("term_expanded_conceptnet_hop", IntegerType()),
        StructField("term_expanded_conceptnet_weight", DoubleType())
    ])

    concept_schema = StructType([
        StructField("uri", StringType()),
        StructField("relationship", StringType()),
        StructField("start", StringType()),
        StructField("end", StringType()),
        StructField("meta", StringType()),
        StructField("relationshipName", StringType()),
        StructField("startName", StringType()),
        StructField("startSense", StringType()),
        StructField("startLanguage", StringType()),
        StructField("endName", StringType()),
        StructField("endSense", StringType()),
        StructField("endLanguage", StringType()),
        StructField("weight", DoubleType()),
    ])

    relationship_schema = StructType([
        StructField("relationship", StringType())
    ])

    valid_languages = [
        'en',
        'nl',
        'id',
        'ms'
    ]

    valid_relationships = [
        '/r/IsA',
        '/r/PartOf',
        '/r/HasA',
        '/r/UsedFor',
        '/r/AtLocation',
        '/r/Causes',
        '/r/HasSubEvent',
        '/r/MotivatedByGoal',
        '/r/Desires',
        '/r/CreatedBy',
        '/r/Synonym',
        '/r/DerivedFrom',
        '/r/Entails',
        '/r/MannerOf',
        '/r/LocatedNear',
        '/r/EtymologicallyRelatedTo',
        '/r/dbpedia/genre',
        '/r/dbpedia/influencedBy',
        '/r/dbpedia/knownFor',
        '/r/dbpedia/occupation',
        '/r/dbpedia/language',
        '/r/dbpedia/field',
        '/r/dbpedia/product'
    ]

    def __init__(self, dictionary, conceptnet_file='conceptnet.csv', input_col="term", valid_min_weight=1.0, valid_languages=None, valid_relationships=None):
        super(ConceptNetDictionaryExpansion, self).__init__(dictionary, input_col)

        self.name = self.name + "-conceptnet"

        if valid_min_weight is not None:
            self.valid_min_weight = valid_min_weight
        if valid_languages is not None:
            self.valid_languages = valid_languages
        if valid_relationships is not None:
            self.valid_relationships = valid_relationships

        print "Valid minimum weight: " + str(self.valid_min_weight)
        print "Valid languages: " + str(self.valid_languages)
        print "Valid relationships: " + str(self.valid_relationships)

        # load concept with valid relationships and languages
        self.load_concepts(conceptnet_file)

    def load_concepts(self, file_name=None):
        if file_name is None:
            file_name = "conceptnet.csv"
            print "No file name specified to load ConceptNet. Using default file: " + self.study.bucket_study_url + file_name

        df = self.study.sqlc.read.format("com.databricks.spark.csv") \
            .options(header=False, delimiter='\t', mode="PERMISSIVE") \
            .load(self.study.bucket_study_url + file_name, schema=self.concept_schema)

        self.concepts = df

        # extract more features
        self.extract_expanded_stats()
        self.extract_concept()

        # filter concepts using valid relationships and languages
        self.filter_concepts()

        return self

    def filter_concepts(self):
        valid_languages = self.valid_languages
        valid_relationships = self.valid_relationships
        valid_min_weight = self.valid_min_weight

        rdd = self.concepts.rdd.filter(lambda concept: concept['startLanguage'] in valid_languages and concept['endLanguage'] in valid_languages and concept['relationship'] in valid_relationships and concept['weight'] >= valid_min_weight)
        df = self.study.sqlc.createDataFrame(rdd, self.concept_schema)
        self.concepts = df
        return self

    def sample_concepts(self, ratio=0.01):
        rdd = self.concepts.rdd.sample(False, ratio)
        df = self.study.sqlc.createDataFrame(rdd, self.concept_schema)
        self.concepts = df
        return self

    def extract_expanded_stats(self):
        def parse_dictionary(dictionary):
            row = dictionary.asDict();

            if row.get('term_expanded_conceptnet_num', None) is None:
                row['term_expanded_conceptnet_num'] = 0

            if row.get('term_expanded_conceptnet_hop', None) is None:
                row['term_expanded_conceptnet_hop'] = 0

            if row.get('term_expanded_conceptnet_weight', None) is None:
                row['term_expanded_conceptnet_weight'] = 10.0

            return row

        rdd = self.df.rdd.map(parse_dictionary)
        df = self.study.sqlc.createDataFrame(rdd, self.schema)
        self.df = df
        return self

    def extract_concept(self):
        def parse_relationship_name(rel):
            rel_name = ""
            rel_parts = rel.split('/')
            if len(rel_parts) == 4:
                rel_name = rel_parts[2] + '/' + rel_parts[3]
            else:
                rel_name = rel_parts[2]
            return rel_name.replace('_', ' ')

        def parse_concept_name(concept):
            return concept.split('/')[3].replace('_', ' ')

        def parse_concept_sense(concept):
            concept_parts = concept.split('/')
            if len(concept_parts) > 4:
                return concept.split('/')[4]
            else:
                return None

        def parse_language(concept):
            return concept.split('/')[2]

        def parse_weight(meta):
            weight = 1.0

            if isinstance(meta, basestring):
                try:
                    meta = json.loads(meta)
                    weight = float(meta['weight'])
                except ValueError, e:
                    pass

            return weight

        def parse_concept(concept):
            row = concept.asDict();
            row['relationshipName'] = parse_relationship_name(concept.relationship)
            row['startName'] = parse_concept_name(concept.start)
            row['startSense'] = parse_concept_name(concept.start)
            row['startLanguage'] = parse_language(concept.start)
            row['endName'] = parse_concept_name(concept.end)
            row['endSense'] = parse_concept_sense(concept.end)
            row['endLanguage'] = parse_language(concept.end)
            row['weight'] = parse_weight(concept.meta)
            return row

        rdd = self.concepts.rdd.map(parse_concept)
        df = self.study.sqlc.createDataFrame(rdd, self.concept_schema)
        self.concepts = df
        return self

    def expand(self):
        input_col = self.input_col
        seed = self.df
        concepts = self.concepts

        def generate_dictionary_senses(dictionary):
            senses = [dictionary]

            if dictionary['term_sense'] is None:
                #  noun
                noun_dictionary = dictionary.copy()
                noun_dictionary['term_sense'] = 'n'
                senses.append(noun_dictionary)

                # verb
                verb_dictionary = dictionary.copy()
                verb_dictionary['term_sense'] = 'v'
                senses.append(verb_dictionary)

                # adj
                adj_dictionary = dictionary.copy()
                adj_dictionary['term_sense'] = 'a'
                senses.append(adj_dictionary)
            return senses

        def preprocess_dictionary(dictionary, term_col):

            # conceptnet language fix for `id`, which registered in `ms`
            language = dictionary['language']
            if language == 'id':
                language = 'ms'

            concept_parts = ['', 'c', language, '_'.join(dictionary[term_col].split(' '))]

            if dictionary['term_sense']:
                concept_parts.append(dictionary['term_sense'])

            concept = "/".join(concept_parts)
            dictionary['term_expanded_conceptnet_concept'] = concept
            return concept, dictionary

        def preprocess_concept(concept):
            return concept['start'], concept

        def extract_dictionary(data):
            seed, concept = data

            if concept is not None:
                hop = 0
                if seed['term_expanded_conceptnet_hop'] is not None:
                    hop = seed['term_expanded_conceptnet_hop']

                hop = hop + 1

                dictionary = {
                    'term_expanded_conceptnet_concept': concept['end'],
                    'term_expanded_conceptnet_from': seed['term_expanded_conceptnet_concept'],
                    'term_expanded_conceptnet_hop': hop,
                    'term_expanded_conceptnet_num': 0,
                    'term_expanded_conceptnet_relationship': concept['relationship'],
                    'term_expanded_conceptnet_weight': concept['weight'],
                    'language': concept['endLanguage'],
                    'source': 'conceptnet',
                    'term': concept['endName'],
                    'term_sense': concept['endSense'],
                    'class': seed['class']
                }
            else:
                dictionary = None
            return dictionary

        def extract_expanded_stats(seed, value):
            if value is not None:
                seed['term_expanded_conceptnet_num'] = len(value)
            else:
                seed['term_expanded_conceptnet_num'] = 0
            return seed
        
        dictionary_rdd = seed.rdd.map(lambda row: row.asDict()).map(lambda dictionary: preprocess_dictionary(dictionary, input_col))
        seed_rdd = seed.rdd.map(lambda row: row.asDict()).flatMap(generate_dictionary_senses).map(lambda dictionary: preprocess_dictionary(dictionary, input_col))
        concept_rdd = concepts.rdd.map(lambda row: row.asDict()).map(preprocess_concept)

        seed_concept_rdd = seed_rdd.leftOuterJoin(concept_rdd)
        
        related_dictionary_rdd = seed_concept_rdd.map(lambda (k, v): extract_dictionary(v)).filter(lambda x: x is not None)
        related_dictionary_num_rdd = related_dictionary_rdd.groupBy(lambda x: x['term_expanded_conceptnet_from'])
        expanded_dictionary_rdd = dictionary_rdd.leftOuterJoin(related_dictionary_num_rdd).map(lambda (k, (v, z)): extract_expanded_stats(v, z))
        
        expanded_dictionary = self.study.sqlc.createDataFrame(expanded_dictionary_rdd, self.schema)
        related_dictionary = self.study.sqlc.createDataFrame(related_dictionary_rdd, self.schema)
        
        # temporary hacks
        # self.df = related_dictionary
        # self.tokenize()
        # related_dictionary = self.df
        
        # self.df = expanded_dictionary
        # self.tokenize(input_col="term")
        # expanded_dictionary = self.df
        
        df = expanded_dictionary.unionAll(related_dictionary).dropDuplicates(['class', 'term'])
        df = df.withColumn('term_raw', df['term'])
        
        self.df = df
        return self
