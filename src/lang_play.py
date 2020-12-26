from gensim.models import KeyedVectors


class LangPlay:
    def __init__(self, w2v_fn):
        self.MAN = "man"
        self.WOMAN = "woman"
        # Loading word2vec is very slow :(
        self.wv_from_text = KeyedVectors.load_word2vec_format(w2v_fn)

    
    def male_to_female(self, male_word):
        female_word = self.wv_from_text.get_vector(male_word) + self.wv_from_text.get_vector(self.WOMAN) -\
            self.wv_from_text.get_vector(self.MAN) 
        return female_word


    def female_to_male(self, female_word):
        male_word = self.wv_from_text.get_vector(female_word) + self.wv_from_text.get_vector(self.MAN) -\
            self.wv_from_text.get_vector(self.WOMAN)
        return male_word


    def print_rank(self, word, similars):
        for index, similar in enumerate(similars):
            if similar[0] == word:
                print("rank for %s is: %s with score: %s" % (word, index, similar[1]))
                return
        print("word: %s not found in similars" % (word))


    def male_female_rank(self, male_word, female_word):
        female_similars = self.wv_from_text.most_similar_cosmul(positive=[self.WOMAN, male_word], negative=[self.MAN])
        male_similars = self.wv_from_text.most_similar_cosmul(positive=[self.MAN, female_word], negative=[self.WOMAN])
        print("male word:%s" % (male_word))
        print("female word:%s" % (female_word))
        print("female_similars:%s" % (female_similars))
        print("male_similars:%s" % (male_similars))
        self.print_rank(female_word, female_similars)
        self.print_rank(male_word, male_similars)


def main():
    w2v_fn = "/Volumes/PHD 3.0 SP/MBP Backup/2020-12-25/fasttext/cc.en.300.vec"
    play = LangPlay(w2v_fn)
    play.male_female_rank("actor", "actress")
