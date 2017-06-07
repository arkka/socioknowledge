from Dictionary import Dictionary


class DictionaryExpansion(Dictionary):
    def __init__(self, dictionary, input_col="term"):
        super(DictionaryExpansion, self).__init__(dictionary.study)
        self.dictionary = dictionary.dictionary
        self.dictionary_name = self.dictionary_name + "-expansion"

        self.input_col = input_col

    def expand(self):
        pass