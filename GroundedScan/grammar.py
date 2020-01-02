#!/usr/bin/env python3
# TODO: max recursion choice per rule
from GroundedScan.world import LogicalForm
from GroundedScan.world import Term
from GroundedScan.world import SemType
from GroundedScan.world import ENTITY
from GroundedScan.world import Variable
from GroundedScan.world import Weights
from GroundedScan.world import EVENT
from GroundedScan.world import COLOR
from GroundedScan.world import SIZE

from typing import List
from typing import ClassVar
from collections import namedtuple
import numpy as np
from itertools import product


Nonterminal = namedtuple("Nonterminal", "name")
Terminal = namedtuple("Terminal", "name")

ROOT = Nonterminal("ROOT")
VP = Nonterminal("VP")
VV_intransitive = Nonterminal("VV_intransitive")
VV_transitive = Nonterminal("VV_transitive")
RB = Nonterminal("RB")
DP = Nonterminal("DP")
NP = Nonterminal("NP")
NN = Nonterminal("NN")
JJ = Nonterminal("JJ")


# TODO cleaner
VAR_COUNTER = [0]
def free_var(sem_type):
    name = "x{}".format(VAR_COUNTER[0])
    VAR_COUNTER[0] += 1
    return Variable(name, sem_type)


class Rule(object):
    """
    Rule-class of form LHS -> RHS with method instantiate that defines its meaning.
    """
    def __init__(self, lhs: Nonterminal, rhs: List, max_recursion=2):
        self.lhs = lhs
        self.rhs = rhs
        self.sem_type = None
        self.max_recursion = max_recursion

    def instantiate(self, *args, **kwargs):
        raise NotImplementedError()


class LexicalRule(Rule):
    """
    Rule of form Non-Terminal -> Terminal.
    """
    def __init__(self, lhs: Nonterminal, word: str, specs: Weights, sem_type: SemType):
        super().__init__(lhs=lhs, rhs=[Terminal(word)], max_recursion=1)
        self.name = word
        self.sem_type = sem_type
        self.specs = specs

    def instantiate(self, meta=None, **kwargs) -> LogicalForm:
        # TODO a little fishy to have recursion meta here rather than in wrapper
        var = free_var(self.sem_type)
        return LogicalForm(
            variables=(var, ),
            terms=(Term(self.name, (var, ), specs=self.specs, meta=meta), )
        )

    def __repr__(self):
        lhs = self.lhs.name
        rhs = self.rhs[0].name
        return "{} -> {}".format(lhs, rhs)


class Root(Rule):
    def __init__(self):
        super().__init__(lhs=ROOT, rhs=[VP])

    def instantiate(self, child, **kwargs):
        return child

    def __repr__(self):
        return "ROOT -> VP"


class RootConj(Rule):
    def __init__(self, max_recursion=0):
        super().__init__(lhs=ROOT, rhs=[VP, Terminal("and"), ROOT], max_recursion=max_recursion)

    def instantiate(self, left_child, right_child, **kwargs):
        return LogicalForm(
            variables=left_child.variables + right_child.variables,
            terms=left_child.terms + right_child.terms + (Term("seq", (left_child.head, right_child.head)),)
        )

    def __repr__(self):
        return "ROOT -> VP 'and' ROOT"


class VpWrapper(Rule):
    def __init__(self, max_recursion=0):
        super().__init__(lhs=VP, rhs=[VP, RB], max_recursion=max_recursion)

    def instantiate(self, rb, vp, meta, **kwargs):
        bound = rb.bind(vp.head)
        assert bound.variables[0] == vp.head
        return LogicalForm(variables=vp.variables + bound.variables[1:], terms=vp.terms + bound.terms)

    def __repr__(self):
        return "VP -> VP RB"


class VpIntransitive(Rule):
    def __init__(self):
        super().__init__(lhs=VP, rhs=[VV_intransitive, Terminal("to"), DP])

    def instantiate(self, vv, dp, meta, **kwargs):
        role = Term("patient", (vv.head, dp.head))
        meta["arguments"].append(dp)
        return LogicalForm(variables=vv.variables + dp.variables, terms=vv.terms + dp.terms + (role,))

    def __repr__(self):
        return "VP -> VV_intrans 'to' DP"


class VpTransitive(Rule):
    def __init__(self):
        super().__init__(lhs=VP, rhs=[VV_transitive, DP])

    def instantiate(self, vv, dp, meta, **kwargs):
        role = Term("patient", (vv.head, dp.head))
        meta["arguments"].append(dp)
        return LogicalForm(variables=vv.variables + dp.variables, terms=vv.terms + dp.terms + (role,))

    def __repr__(self):
        return "VP -> VV_trans DP"


class Dp(Rule):
    def __init__(self):
        super().__init__(lhs=DP, rhs=[Terminal("a"), NP])

    def instantiate(self, np, **kwargs):
        return np

    def __repr__(self):
        return "DP -> 'a' NP"


class NpWrapper(Rule):
    def __init__(self, max_recursion=0):
        super().__init__(lhs=NP, rhs=[JJ, NP], max_recursion=max_recursion)

    def instantiate(self, jj, np, meta=None, **kwargs):
        bound = jj.bind(np.head)
        assert bound.variables[0] == np.head
        return LogicalForm(variables=np.variables + bound.variables[1:], terms=np.terms + bound.terms)

    def __repr__(self):
        return "NP -> JJ NP"


class Np(Rule):
    def __init__(self):
        super().__init__(lhs=NP, rhs=[NN])

    def instantiate(self, nn, **kwargs):
        return nn

    def __repr__(self):
        return "NP -> NN"


class Derivation(object):
    """
    Holds a constituency tree that makes up a sentence. Can be used to obtain the meaning of a sentence in terms
    of a Logical Form. The meaning of a derivation is made up of the meaning of its children.
    """

    def __init__(self, rule, children=None, meta={}):
        self.rule = rule
        self.lhs = rule.lhs
        self.children = children
        self.meta = meta

    @classmethod
    def from_rules(cls, rules: list, symbol=ROOT, lexicon=None):
        """Recursively form a derivation from a rule list that has been constructed in a depth-first manner,
        use the lexicon for the Lexical Rules at the leafs of the constituency tree."""
        # If the current symbol is a Terminal, close current branch and return.
        if isinstance(symbol, Terminal):
            return symbol
        if symbol not in lexicon.keys():
            next_rule = rules.pop()
        else:
            next_rule = lexicon[symbol].pop()

        return Derivation(
            next_rule,
            tuple(cls.from_rules(rules, symbol=next_symbol, lexicon=lexicon) for next_symbol in next_rule.rhs)
        )

    def to_rules(self, rules: list, lexicon: dict):
        for child in self.children:
            if isinstance(child, Derivation):
                child.to_rules(rules, lexicon)
            else:
                lexicon[child] = [child]
        if isinstance(self.rule, LexicalRule):
            if self.rule.lhs not in lexicon:
                lexicon[self.rule.lhs] = [self.rule]
            else:
                lexicon[self.rule.lhs] = [self.rule] + lexicon[self.rule.lhs]
        else:
            rules.append(self.rule)
        return

    def words(self) -> tuple:
        """Obtain all words of a derivation by combining the words of all the children."""
        out = []
        for child in self.children:
            if isinstance(child, Terminal):
                out.append(child.name)
            else:
                out += child.words()
        return tuple(out)

    # TODO canonical variable names, not memoization
    def meaning(self, arguments: list) -> LogicalForm:
        """Recursively define the meaning of the derivation by instantiating the meaning of each child."""
        self.meta["arguments"] = arguments
        if not hasattr(self, "_cached_logical_form"):
            child_meanings = [
                child.meaning(arguments)
                for child in self.children
                if isinstance(child, Derivation)
            ]
            meaning = self.rule.instantiate(*child_meanings, meta=self.meta)
            self._cached_logical_form = meaning
        return self._cached_logical_form

    @classmethod
    def from_str(cls, rules_str, lexicon_str, grammar):
        # TODO: method to instantiate derivation from str (see __repr__)
        rules_list = []
        for rule in rules_str.split(','):
            rules_list.append(grammar.rule_str_to_rules[rule])
        lexicon = {}
        lexicon_list = lexicon_str.split(',')
        for entry in lexicon_list:
            items = entry.split(':')
            symbol_type = items[0]
            for item in items[1:]:
                if symbol_type == 'T':
                    new_terminal = Terminal(item)
                    lexicon[new_terminal] = [new_terminal]
                else:
                    rule = grammar.rule_str_to_rules[item]
                    if rule.lhs not in lexicon:
                        lexicon[rule.lhs] = [rule]
                    else:
                        lexicon[rule.lhs].append(rule)
        return cls.from_rules(rules_list, lexicon=lexicon)

    def __repr__(self):
        rules = []
        lexicon = {}
        self.to_rules(rules, lexicon)
        rules_str = ','.join([str(rule) for rule in rules])
        lexicon_list = []
        for key, value in lexicon.items():
            if isinstance(key, Nonterminal):
                symbol_str = "NT"
                for rhs_symbol in value:
                    symbol_str += ":{}".format(rhs_symbol)
                lexicon_list.append(symbol_str)
            else:
                lexicon_list.append("T:{}".format(value[0].name))
        lexicon_str = ','.join(lexicon_list)
        return rules_str + ';' + lexicon_str


class Template(object):
    """
    A template is a constituency-tree without lexical rules. From a template together with a lexicon, multiple
    constituency trees can be formed.
    """

    def __init__(self):
        self._left_values = []
        self._right_values = []
        self._leftmost_nonterminal = None
        self.rules = []

    def add_value(self, value, expandable):
        if expandable and not self._leftmost_nonterminal:
            self._leftmost_nonterminal = value
        elif self._leftmost_nonterminal:
            self._right_values.append(value)
        else:
            self._left_values.append(value)

    def has_nonterminal(self):
        return self._leftmost_nonterminal is not None

    def get_leftmost_nonterminal(self):
        assert self.has_nonterminal(), "Trying to get a NT but none present in this derivation."
        return self._leftmost_nonterminal

    def expand_leftmost_nonterminal(self, rule, expandables):
        new_derivation = Template()
        new_derivation_symbols = self._left_values + rule.rhs + self._right_values
        new_derivation.rules = self.rules.copy()
        new_derivation.rules.append(rule)
        for value in new_derivation_symbols:
            if value in expandables:
                new_derivation.add_value(value, expandable=True)
            else:
                new_derivation.add_value(value, expandable=False)
        return new_derivation

    def to_derivation(self):
        assert not self.has_nonterminal(), "Trying to write a non-terminal to a string."
        self.rules.reverse()
        return self._left_values, self.rules


class Grammar(object):
    RULES = {
        "conjunction": [Root(), RootConj(max_recursion=2), VpWrapper(), VpIntransitive(), VpTransitive(), Dp(),
                        NpWrapper(max_recursion=2), Np()],
        "adverb": [Root(), VpWrapper(), VpIntransitive(), VpTransitive(), Dp(),
                   NpWrapper(max_recursion=2), Np()],
        "normal": [Root(), VpIntransitive(), VpTransitive(), Dp(), NpWrapper(max_recursion=2), Np()],
        "simple_trans": [Root(), VpTransitive(), Dp(), NpWrapper(max_recursion=1), Np()],
        "simple_intrans": [Root(), VpIntransitive(), Dp(), NpWrapper(max_recursion=1), Np()]
    }

    def __init__(self, vocabulary: ClassVar, max_recursion=1, type_grammar="normal"):
        """
        Defines a grammar of NT -> NT rules and NT -> T rules depending on the vocabulary.
        :param vocabulary: an instance of class Vocabulary filled with different types of words.
        :param max_recursion: Maximum recursion to be allowed in generation of examples.
        :param type_grammar: options are 'full', 'adverb', 'normal' and 'simple'. Determines which set of common rules
        is chosen.
        """
        assert type_grammar in self.RULES, "Specified unsupported type grammar {}".format(type_grammar)
        self.type_grammar = type_grammar
        if type_grammar == "simple_intrans":
            assert len(vocabulary.get_intransitive_verbs()) > 0, "Please specify intransitive verbs."
        elif type_grammar == "simple_trans":
            assert len(vocabulary.get_transitive_verbs()) > 0, "Please specify transitive verbs."
        self.rule_list = self.RULES[type_grammar] + self.lexical_rules(vocabulary.get_intransitive_verbs(),
                                                                       vocabulary.get_transitive_verbs(),
                                                                       vocabulary.get_adverbs(),
                                                                       vocabulary.get_nouns(),
                                                                       vocabulary.get_color_adjectives(),
                                                                       vocabulary.get_size_adjectives())
        nonterminals = {rule.lhs for rule in self.rule_list}
        self.rules = {nonterminal: [] for nonterminal in nonterminals}
        self.nonterminals = {nt.name: nt for nt in nonterminals}
        self.terminals = {}

        self.vocabulary = vocabulary
        self.rule_str_to_rules = {}
        for rule in self.rule_list:
            self.rules[rule.lhs].append(rule)
            self.rule_str_to_rules[str(rule)] = rule
        self.expandables = set(rule.lhs for rule in self.rule_list if not isinstance(rule, LexicalRule))
        self.categories = {
            "manner": set(vocabulary.get_adverbs()),
            "shape": {n for n in vocabulary.get_nouns()},
            "color": set([v for v in vocabulary.get_color_adjectives()]),
            "size": set([v for v in vocabulary.get_size_adjectives()])
        }
        self.word_to_category = {}
        for category, words in self.categories.items():
            for word in words:
                self.word_to_category[word] = category

        self.max_recursion = max_recursion
        self.all_templates = []
        self.all_derivations = {}
        self.command_statistics = self.empty_command_statistics()

    @staticmethod
    def empty_command_statistics():
        return {
            VV_intransitive: {},
            VV_transitive: {},
            NN: {},
            JJ: {},
            RB: {}
        }

    def reset_grammar(self):
        self.command_statistics = self.empty_command_statistics()
        self.all_templates.clear()
        self.all_derivations.clear()

    def lexical_rules(self, verbs_intrans: List[str], verbs_trans: List[str], adverbs: List[str], nouns: List[str],
                      color_adjectives: List[str], size_adjectives: List[str]) -> list:
        """
        Instantiate the lexical rules with the sampled words from the vocabulary.
        """
        assert size_adjectives or color_adjectives, "Please specify words for at least one of size_adjectives or "\
                                                    "color_adjectives."
        all_rules = []
        vv_intrans_rules = [
            LexicalRule(lhs=VV_intransitive, word=verb, sem_type=EVENT, specs=Weights(action=verb, is_transitive=False))
            for verb in verbs_intrans
        ]
        all_rules += vv_intrans_rules
        if self.type_grammar != "simple":
            vv_trans_rules = [
                LexicalRule(lhs=VV_transitive, word=verb, sem_type=EVENT, specs=Weights(action=verb, is_transitive=True))
                for verb in verbs_trans
            ]
            all_rules += vv_trans_rules
        if self.type_grammar == "adverb" or self.type_grammar == "full":
            rb_rules = [LexicalRule(lhs=RB, word=word, sem_type=EVENT, specs=Weights(manner=word)) for word in adverbs]
            all_rules += rb_rules
        nn_rules = [LexicalRule(lhs=NN, word=word, sem_type=ENTITY, specs=Weights(noun=word)) for word in nouns]
        all_rules += nn_rules
        jj_rules = []
        if color_adjectives:
            jj_rules.extend([LexicalRule(lhs=JJ, word=word, sem_type=ENTITY, specs=Weights(adjective_type=COLOR))
                            for word in color_adjectives])
        if size_adjectives:
            jj_rules.extend([LexicalRule(lhs=JJ, word=word, sem_type=ENTITY, specs=Weights(adjective_type=SIZE))
                            for word in size_adjectives])
        all_rules += jj_rules
        return all_rules

    def sample(self, symbol=ROOT, last_rule=None, recursion=0):
        """
        Sample a command from the grammar by recursively sampling rules for each symbol.
        :param symbol: current node in constituency tree.
        :param last_rule:  previous rule sampled.
        :param recursion: recursion depth (increases if sample ruled is applied twice).
        :return: Derivation
        """
        # If the current symbol is a Terminal, close current branch and return.
        if isinstance(symbol, Terminal):
            return symbol
        nonterminal_rules = self.rules[symbol]

        # Filter out last rule if max recursion depth is reached.
        if recursion == self.max_recursion - 1:
            nonterminal_rules = [rule for rule in nonterminal_rules if rule != last_rule]

        # Sample a random rule.
        next_rule = nonterminal_rules[np.random.randint(len(nonterminal_rules))]
        next_recursion = recursion + 1 if next_rule == last_rule else 0
        return Derivation(
            next_rule,
            tuple(self.sample(next_symbol, next_rule, next_recursion) for next_symbol in next_rule.rhs),
            meta={"recursion": recursion}
        )

    def generate_all(self, current_template: Template, all_templates: list, rule_use_counter: dict):

        # If the template contains no non-terminals, we close this branch.
        if not current_template.has_nonterminal():
            all_templates.append(current_template.to_derivation())
            return

        # Retrieve the leftmost non-terminal to expand.
        leftmost_nonterminal = current_template.get_leftmost_nonterminal()

        # Get all possible RHS replacements and start a new derivation branch for each of them.
        rules_leftmost_nonterminal = self.rules[leftmost_nonterminal]
        for rule_leftmost_nonterminal in rules_leftmost_nonterminal:

            # Lexical rules are not expandable
            if isinstance(rule_leftmost_nonterminal, LexicalRule):
                continue

            # Each branch gets its own rule usage counter.
            rule_use_counter_copy = rule_use_counter.copy()

            # If this rule has already been applied in the current branch..
            if rule_leftmost_nonterminal in rule_use_counter_copy.keys():

                # ..do not use it again if it has been applied more than a maximum allowed number of times.
                if rule_use_counter[rule_leftmost_nonterminal] >= rule_leftmost_nonterminal.max_recursion:
                    continue
                rule_use_counter_copy[rule_leftmost_nonterminal] += 1
            else:
                rule_use_counter_copy[rule_leftmost_nonterminal] = 1

            # Get the next derivation by replacing the leftmost NT with its RHS.
            next_template = current_template.expand_leftmost_nonterminal(rule_leftmost_nonterminal,
                                                                         self.expandables)

            # Start a new derivation branch for this RHS.
            self.generate_all(next_template, all_templates, rule_use_counter_copy)

    def form_commands_from_template(self, derivation_template: list, derivation_rules: list):
        """
        Replaces all NT's in a template with the possible T's and forms all possible commands with those.
        If multiple the same NT's follow each other, e.g. a JJ JJ JJ NN, for each following JJ the possible words
        will be halved over the possibilities, meaning no words will repeat themselves (e.g. the red red circle),
        this does mean that whenever the max. recursion depth for a rule is larger than the log(n) where n is the number
        of words for that particular rule, this does not have an effect.
        :param derivation_template: list of NT's, e.g. [VV_intrans, 'to', 'a', JJ, JJ, NN, RB]
        :param derivation_rules: list of rules that build up the constituency tree for this template
        :return: all possible combinations where all NT's are replaced by the words from the lexicon.
        """

        # In the templates, replace each lexical rule with the possible words from the lexicon
        replaced_template = []
        previous_symbol = None
        lexicon = {}
        for symbol in derivation_template:
            if isinstance(symbol, Nonterminal):
                possible_words = [s.name for s in self.rules[symbol]]
                for rule in self.rules[symbol]:
                    lexicon[rule.name] = rule
                if previous_symbol == symbol:
                    previous_words = replaced_template.pop()
                    first_words, second_words = self.split_on_category(previous_words)
                    replaced_template.append(first_words)
                    replaced_template.append(second_words)
                else:
                    replaced_template.append(possible_words)
            else:
                lexicon[symbol.name] = symbol
                replaced_template.append([symbol.name])
            previous_symbol = symbol

        # Generate all possible commands from the templates.
        all_commands = [command for command in product(*replaced_template)]
        all_derivations = []
        for command in all_commands:
            command_lexicon = {}
            for word, symbol in zip(command, derivation_template):
                if symbol not in command_lexicon:
                    command_lexicon[symbol] = [lexicon[word]]
                else:
                    command_lexicon[symbol] = [lexicon[word]] + command_lexicon[symbol]
                if isinstance(symbol, Nonterminal):
                    if word not in self.command_statistics[symbol].keys():
                        self.command_statistics[symbol][word] = 1
                    else:
                        self.command_statistics[symbol][word] += 1
            derivation = Derivation.from_rules(derivation_rules.copy(), symbol=ROOT, lexicon=command_lexicon)
            assert ' '.join(derivation.words()) == ' '.join(command), "Derivation and command not the same."
            all_derivations.append(derivation)
        return all_derivations

    def generate_all_commands(self):

        # Generate all possible templates from the grammar.
        initial_template = Template()
        initial_template.add_value(value=ROOT, expandable=True)
        self.generate_all(current_template=initial_template, all_templates=self.all_templates,
                          rule_use_counter={})

        # For each template, form all possible commands by combining it with the lexicon.
        for i, (derivation_template, derivation_rules) in enumerate(self.all_templates):
            derivations = self.form_commands_from_template(derivation_template, derivation_rules)
            self.all_derivations[i] = derivations

    def split_on_category(self, words_list):
        first_category_words = [words_list[0]]
        second_category_words = []
        first_category = self.category(words_list[0])
        for word in words_list[1:]:
            if self.category(word) == first_category:
                first_category_words.append(word)
            else:
                second_category_words.append(word)
        return first_category_words, second_category_words

    def category(self, function):
        return self.word_to_category.get(function)

    def is_coherent(self, logical_form):
        """
        Returns true for coherent logical forms, false otherwise. A command's logical form is coherent the
        arguments of a variable have all unique categories. E.g. in coherent would be: 'the red blue circle'.
        """
        for variable in logical_form.variables:
            functions = [term.function for term in logical_form.terms if variable in term.arguments]
            categories = [self.category(function) for function in functions]
            categories = [c for c in categories if c is not None]
            if len(categories) != len(set(categories)):
                return False
        return True

    def __str__(self):
        output_str = ""
        for rule in self.rule_list:
            output_str += rule.__str__() + ';'
        return output_str

