from Dictionary import Dictionary


class DictionaryExpansion(Dictionary):
    def __init__(self, dictionary, input_col="term"):
        super(DictionaryExpansion, self).__init__(dictionary.study)
        self.df = dictionary.df
        self.name = self.name + "-expansion"

        self.input_col = input_col

    def expand(self):
        pass