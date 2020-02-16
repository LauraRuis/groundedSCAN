import pronounceable
from typing import List, Set, Dict


class Vocabulary(object):
    """
    Method containing functionality for vocabulary. Allows for both random sampling of nonce-vocabulary by initializing
    through class method `sample` as well as setting user-defined words through default constructor.
    """
    INTRANSITIVE_VERBS = {"walk"}
    TRANSITIVE_VERBS = {"push", "pull"}
    ADVERBS = {"quickly", "slowly", "while zigzagging", "while spinning", "cautiously", "hesitantly"}
    NOUNS = {"circle", "square", "cylinder"}
    COLOR_ADJECTIVES = {"green", "red", "blue", "yellow"}
    SIZE_ADJECTIVES = {"small", "big"}

    def __init__(self, intransitive_verbs: Dict[str, str], transitive_verbs: Dict[str, str], adverbs: Dict[str, str],
                 nouns: Dict[str, str], color_adjectives: Dict[str, str], size_adjectives: Dict[str, str]):
        all_words = list(intransitive_verbs.keys()) + list(transitive_verbs.keys()) + list(adverbs.keys()) \
                    + list(nouns.keys()) + list(color_adjectives.keys()) + list(size_adjectives.keys())
        all_unique_words = set(all_words)
        self._intransitive_verbs = intransitive_verbs
        self._transitive_verbs = transitive_verbs
        self._adverbs = adverbs
        self._nouns = nouns
        self._color_adjectives = color_adjectives
        self._size_adjectives = size_adjectives
        assert len(all_words) == len(all_unique_words), "Overlapping vocabulary (the same string used twice)."
        if len(color_adjectives) > 0 and len(size_adjectives) > 0:
            self._adjectives = list(self._color_adjectives.values()) + list(self._size_adjectives.values())
        elif len(color_adjectives) > 0:
            self._adjectives = list(self._color_adjectives.values())
        else:
            self._adjectives = list(self._size_adjectives.values())
        self._translation_table = {'to': 'to', 'a': 'a', 'and': 'and'}
        self._translation_table.update(self._intransitive_verbs)
        self._translation_table.update(self._transitive_verbs)
        self._translation_table.update(self._nouns)
        self._translation_table.update(self._color_adjectives)
        self._translation_table.update(self._size_adjectives)
        self._translation_table.update(self._adverbs)
        self._translate_to = {semantic_word: word for word, semantic_word in self._translation_table.items()}

    def get_intransitive_verbs(self):
        return list(self._intransitive_verbs.keys()).copy()

    def get_transitive_verbs(self):
        return list(self._transitive_verbs.keys()).copy()

    def get_adverbs(self):
        return list(self._adverbs.keys()).copy()

    def get_nouns(self):
        return list(self._nouns.keys()).copy()

    def get_color_adjectives(self):
        return list(self._color_adjectives.keys()).copy()

    def get_size_adjectives(self):
        return list(self._size_adjectives.keys()).copy()

    def get_semantic_shapes(self):
        return list(self._nouns.values()).copy()

    def get_semantic_colors(self):
        return list(self._color_adjectives.values()).copy()

    def translate_word(self, word: str) -> str:
        if word in self._translation_table:
            return self._translation_table[word]
        else:
            return ""

    def translate_meaning(self, meaning: str) -> str:
        if meaning in self._translate_to:
            return self._translate_to[meaning]
        else:
            return ""

    @property
    def n_attributes(self):
        return len(self._nouns) * len(self._color_adjectives)

    @staticmethod
    def bind_words_to_meanings(available_words: List[str], available_semantic_meanings: Set[str]) -> Dict[str, str]:
        assert len(available_words) <= len(available_semantic_meanings), "Too many words specified for available"\
                                                                         "semantic meanings: {}".format(
            available_semantic_meanings)
        translation_table = {}
        for word in available_words:
            if word in available_semantic_meanings:
                translation_table[word] = word
                available_semantic_meanings.remove(word)
            else:
                translation_table[word] = available_semantic_meanings.pop()
        return translation_table

    @classmethod
    def initialize(cls, intransitive_verbs: List[str], transitive_verbs: List[str], adverbs: List[str],
                   nouns: List[str], color_adjectives: List[str], size_adjectives: List[str]):
        intransitive_verbs = cls.bind_words_to_meanings(intransitive_verbs, cls.INTRANSITIVE_VERBS.copy())
        transitive_verbs = cls.bind_words_to_meanings(transitive_verbs, cls.TRANSITIVE_VERBS.copy())
        nouns = cls.bind_words_to_meanings(nouns, cls.NOUNS.copy())
        color_adjectives = cls.bind_words_to_meanings(color_adjectives, cls.COLOR_ADJECTIVES.copy())
        size_adjectives = cls.bind_words_to_meanings(size_adjectives, cls.SIZE_ADJECTIVES.copy())
        adverbs = cls.bind_words_to_meanings(adverbs, cls.ADVERBS.copy())
        return cls(intransitive_verbs, transitive_verbs, adverbs, nouns, color_adjectives, size_adjectives)

    @classmethod
    def sample(cls, num_intransitive=1, num_transitive=1, num_adverbs=6, num_nouns=3,
               num_color_adjectives=3, num_size_adjectives=2):
        """
        Sample random nonce-words and initialize the vocabulary with these.
        """
        # Generate random nonce-words
        intransitive_verbs = [pronounceable.generate_word() for _ in range(num_intransitive)]
        transitive_verbs = [pronounceable.generate_word() for _ in range(num_transitive)]
        adverbs = [pronounceable.generate_word() for _ in range(num_adverbs)]
        nouns = [pronounceable.generate_word() for _ in range(num_nouns)]
        color_adjectives = [pronounceable.generate_word() for _ in range(num_color_adjectives)]
        size_adjectives = [pronounceable.generate_word() for _ in range(num_size_adjectives)]
        intransitive_verbs = cls.bind_words_to_meanings(intransitive_verbs, cls.INTRANSITIVE_VERBS.copy())
        transitive_verbs = cls.bind_words_to_meanings(transitive_verbs, cls.TRANSITIVE_VERBS.copy())
        nouns = cls.bind_words_to_meanings(nouns, cls.NOUNS.copy())
        color_adjectives = cls.bind_words_to_meanings(color_adjectives, cls.COLOR_ADJECTIVES.copy())
        size_adjectives = cls.bind_words_to_meanings(size_adjectives, cls.SIZE_ADJECTIVES.copy())
        adverbs = cls.bind_words_to_meanings(adverbs, cls.ADVERBS.copy())
        return cls(intransitive_verbs, transitive_verbs, adverbs, nouns, color_adjectives, size_adjectives)

    def to_representation(self):
        return {
            "intransitive_verbs": self._intransitive_verbs,
            "transitive_verbs": self._transitive_verbs,
            "nouns": self._nouns,
            "adverbs": self._adverbs,
            "color_adjectives": self._color_adjectives,
            "size_adjectives": self._size_adjectives
        }

    @classmethod
    def from_representation(cls, representation: Dict[str, Dict[str, str]]):
        return cls(representation["intransitive_verbs"], representation["transitive_verbs"],
                   representation["adverbs"], representation["nouns"], representation["color_adjectives"],
                   representation["size_adjectives"])
