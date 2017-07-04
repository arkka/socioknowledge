import json

from Stream import Stream

class TwitterStream(Stream):
    def __init__(self, study, name="twitter-stream"):
        super(TwitterStream, self).__init__(study)
        self.study = study
        self.df = None
        self.name = name
        self.index_name = "/".join([self.study.id + "-streams", "stream"])

    def import_mongo(self, collection='tweets', query='{}', limit=0, skip=0):
        tweetStreamsRDD = self.study.sc.newAPIHadoopRDD(
            inputFormatClass='com.mongodb.hadoop.MongoInputFormat',
            keyClass='org.apache.hadoop.io.Text',
            valueClass='org.apache.hadoop.io.MapWritable',
            conf={
                'mongo.input.uri': self.study.mongo_url + '.' + collection,
                'mongo.input.query': query,
                'mongo.input.limit': str(limit),
                'mongo.input.skip': str(skip)
            }
        )

        def extractStream(tweet):
            key, val = tweet

            provider = "twitter"
            text = val['text'].replace('\r', '').replace('\n', '')
            stream = {
                'id': "-".join([provider, val['id_str']]),
                'provider': provider,
                'timestamp': int(val['timestamp_ms']),
                'text': text,
                'link': "/".join(['https://twitter.com', val['user']['id_str'], 'status', val['id_str']]),
                'actor_id': "-".join([provider, val['user']['id_str']]),
                'actor_name': val['user']['name'],
                'actor_link': "/".join(['https://twitter.com', val['user']['id_str']]),
                'text_raw': text
                # 'data': json.dumps(val)
            }

            if val.get('lang', None) is not None:
                stream['language'] = val['lang']

            if val.get('coordinates', None) is not None and val['coordinates'].get('type', None) is 'Point' and val[
                'coordinates'].get('coordinates', None) is not None:
                stream['coordinate_latitude'] = float(val['coordinates']['coordinates'][1])
                stream['coordinate_longitude'] = float(val['coordinates']['coordinates'][0])

            if val.get('place', None) is not None:
                stream['place_id'] = "-".join([provider, val['place'].get('id', None)])
                stream['place_name'] = val['place'].get('full_name', None)
                stream['place_link'] = val['place'].get('url', None)
                stream['place_type'] = val['place'].get('place_type', None)
                stream['place_country'] = val['place'].get('country_code', None)

                if val.get('place', {}).get('bounding_box', {}).get('coordinates', None) is not None:
                    stream['place_latitude'] = (float(val['place']['bounding_box']['coordinates'][0][0][1]) + float(
                        val['place']['bounding_box']['coordinates'][0][1][1])) / 2
                    stream['place_longitude'] = (float(val['place']['bounding_box']['coordinates'][0][0][0]) + float(
                        val['place']['bounding_box']['coordinates'][0][2][0])) / 2
            return stream

        tweetStreamsRDD = tweetStreamsRDD.map(extractStream)

        df = self.study.sqlc.createDataFrame(tweetStreamsRDD, self.schema)
        self.df = df.cache()
        return self