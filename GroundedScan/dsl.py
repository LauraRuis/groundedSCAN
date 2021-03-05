from collections import namedtuple
from typing import List, Union, Set, NewType
from copy import deepcopy
import random
import itertools

from GroundedScan.world import Direction, DIR_TO_INT, NORTH, SOUTH, WEST, EAST
from GroundedScan.world import Position, DIR_TO_VEC, INT_TO_DIR, Situation
from GroundedScan.gym_minigrid.minigrid import Grid


Nonterminal = namedtuple("Nonterminal", "name")
Terminal = namedtuple("Terminal", "name")
Symbol = NewType("Symbol", Union[Nonterminal, Terminal])

Walk = Terminal("Walk")
Tl = Terminal("Tl")
Tr = Terminal("Tr")
ACTION = Nonterminal("ACTION")
DIR1 = Nonterminal("DIR1")
DIR2 = Nonterminal("DIR2")
Push = Terminal("Push")
Pull = Terminal("Pull")
Stay = Terminal("Stay")
East = Terminal("East")
South = Terminal("South")
North = Terminal("North")
West = Terminal("West")
EMPTY = Nonterminal("EMPTY")

SYMBOL_CONVERT = {
    East: EAST,
    North: NORTH,
    South: SOUTH,
    West: WEST
}

GET_OPPOSITE_DIR = {
    EAST: WEST,
    WEST: EAST,
    NORTH: SOUTH,
    SOUTH: NORTH
}


class Node(object):
    """A doubly linked list Node with L-system symbols."""

    def __init__(self, value: Symbol):
        self._value = value
        self._next = None
        self._previous = None

        # The replaced property keeps track of whether a node
        # has just been replaced or added by a rule.
        self._replaced = False

    def set(self, value: Symbol):
        self._value = value

    def set_replaced(self):
        self._replaced = True

    def unset_replaced(self):
        self._replaced = False

    def is_replaced(self):
        return self._replaced

    def get(self) -> Symbol:
        return self._value

    def is_none(self) -> bool:
        return not self._value

    def set_next(self, next_node: "Node"):
        self._next = next_node

    def set_previous(self, previous_node: "Node"):
        self._previous = previous_node

    def get_next(self) -> "Node":
        return self._next

    def get_previous(self) -> "Node":
        return self._previous


class Sequence(object):
    """A sequence of L-System symbols implemented as a linked list."""

    def __init__(self):
        self._sequence_start = None
        self._sequence_end = None
        self._sequence_size = 0
        self._current_node_ptr = None

    def append(self, symbol: Union[Nonterminal, Terminal]):
        if len(self) == 0:
            self._sequence_start = Node(value=symbol)
            self._sequence_end = self._sequence_start
        else:
            next_node = Node(value=symbol)
            next_node.set_previous(self._sequence_end)
            self._sequence_end.set_next(next_node)
            self._sequence_end = next_node
        self._sequence_size += 1

    def extend(self, symbols: List[Union[Nonterminal, Terminal]]):
        for symbol in symbols:
            self.append(symbol)

    def unset_replaced(self):
        current_node = self._sequence_start
        while current_node:
            current_node.unset_replaced()
            current_node = current_node.get_next()

    def iterate(self):
        current_node = self._sequence_start
        while current_node:
            node = current_node
            current_node = current_node.get_next()
            yield node

    def replace_node(self, current_node: Node,
                     replace_with: Symbol) -> Node:
        """Replaces the `current_node` with `replace_with` in the LL."""
        new_node = Node(value=replace_with)
        new_node.set_replaced()
        previous_node = current_node.get_previous()
        next_node = current_node.get_next()
        new_node.set_previous(previous_node)
        new_node.set_next(next_node)
        if current_node == self._sequence_start:
            self._sequence_start = new_node
        if current_node == self._sequence_end:
            self._sequence_end = new_node
        if previous_node:
            previous_node.set_next(new_node)
        if next_node:
            next_node.set_previous(new_node)
        return next_node

    def delete_node(self, current_node: Node):
        previous_node = current_node.get_previous()
        next_node = current_node.get_next()
        if current_node == self._sequence_start:
            # Current node is sequence start
            if next_node:
                self._sequence_start = next_node
                next_node.set_previous(None)
            else:
                self._sequence_start = None
                self._sequence_end = None
                self._sequence_size = 0
        elif current_node == self._sequence_end:
            previous_node.set_next(None)
            self._sequence_end = previous_node
        else:
            previous_node.set_next(next_node)
            next_node.set_previous(previous_node)
        return next_node

    def insert_node(self, current_node: Node,
                    insert_node: Symbol):
        """Inserts `insert_node` before `current_node` in the LL."""
        new_node = Node(value=insert_node)
        new_node.set_replaced()

        # If current_node is None, the previous node is the last.
        if not current_node:
            previous_node = self._sequence_end
        else:
            previous_node = current_node.get_previous()

        if current_node != self._sequence_start:
            previous_node.set_next(new_node)
        new_node.set_next(current_node)
        new_node.set_previous(previous_node)
        if not current_node:
            self._sequence_end = new_node
        else:
            current_node.set_previous(new_node)
        if current_node == self._sequence_start:
            self._sequence_start = new_node
        self._sequence_size += 1
        return current_node

    def replace(self, current_symbol_ptr: Node, lhs_length: int, rhs: "Sequence"):
        """
        Replace a part of the LL starting at `current_symbol_ptr`
        with `replace_with`. If the lhs was equally long as the
        values to replace it with, replace everything, if the RHS
        is longer, also insert.

        :return: the next node in the LL
        e.g., if the sequence is W W A TL, we call replace with
        `current_symbol_ptr` A and `rhs` A W, we return the final
        node TL and the sequence is now W W A W TL.
        TODO: what if rhs < lhs.
        """
        for i, symbol_node in enumerate(rhs.iterate()):
            if i < lhs_length:
                current_symbol_ptr = self.replace_node(current_symbol_ptr,
                                                       symbol_node.get())
            else:
                current_symbol_ptr = self.insert_node(current_symbol_ptr,
                                                      symbol_node.get())
        return current_symbol_ptr

    def get_start(self):
        return self._sequence_start

    def get_end(self):
        return self._sequence_end

    def get_gscan_actions(self) -> List[str]:
        gscan_actions = []
        for i, node in enumerate(self.iterate()):
            action = node.get().name.lower()
            if action == "tl":
                action = "turn left"
            elif action == "tr":
                action = "turn right"
            gscan_actions.append(action)
        return gscan_actions

    def __len__(self):
        return self._sequence_size

    def __eq__(self, sequence):
        if len(self) != len(sequence):
            return False
        for symbol_1, symbol_2 in zip(self.iterate(), sequence.iterate()):
            if symbol_1 != symbol_2:
                return False
        return True

    def __repr__(self):
        output_str = " ".join(s.get().name for s in self.iterate() if s.get().name != "EMPTY")
        return output_str


class Lhs(Sequence):
    """
    Left-hand-side used for an L-system rule.
    Can be made unconditional or conditional rule.
    E.g., W -> A W (W lhs), or {A}A A{A} -> TL TL ({A}A A{A} lhs).
    """

    def __init__(self, symbols: List[Symbol]):
        super().__init__()
        self.extend(symbols)
        self._lhs_no_conditional = Sequence()
        self._lhs_no_conditional.extend(symbols)
        self._left_conditional = None
        self._right_conditional = None
        self._full_lhs = self

    def set_conditional(self, left_conditional: List[Symbol],
                        right_conditional: List[Symbol]):
        """Initializes the conditionals."""
        self._left_conditional = Sequence()
        self._left_conditional.extend(left_conditional)
        self._right_conditional = Sequence()
        self._right_conditional.extend(right_conditional)
        sequence_start = self.get_start()
        for symbol in left_conditional:
            self.insert_node(sequence_start, symbol)
        for symbol in right_conditional:
            self.append(symbol)

    def num_left_conditional(self):
        """Number of symbols in the left conditional."""
        if self._left_conditional:
            return len(self._left_conditional)
        else:
            return 0

    def num_right_conditional(self):
        """Number of symbols in the right conditional."""
        if self._right_conditional:
            return len(self._right_conditional)
        else:
            return 0

    def __repr__(self):
        left_conditional_str = ""
        if self._left_conditional:
            left_conditional = [s.get().name for s in self._left_conditional.iterate()]
            left_conditional_str = "{%s}" % " ".join(left_conditional)
        right_conditional_str = ""
        if self._right_conditional:
            right_conditional = [s.get().name for s in self._right_conditional.iterate()]
            right_conditional_str = "{%s}" % " ".join(right_conditional)
        middle_lhs = [s.get().name for s in self._lhs_no_conditional.iterate()]
        lhs_str = "%s%s%s" % (left_conditional_str,
                              " ".join(middle_lhs),
                              right_conditional_str)
        return lhs_str


class Rhs(Sequence):
    """
    Right-hand-side used for an L-system rule.
    E.g., W -> A W (A W rhs), or {A}A A{A} -> TL TL (TL TL rhs).
    """

    def __init__(self, symbols: List[Union[Nonterminal, Terminal]]):
        super().__init__()
        self.extend(symbols)
        self._rhs_list = symbols

    def __repr__(self):
        return " ".join(s.get().name for s in self.iterate())


class LSystemRule(object):
    """
    Base class for a rewrite rule in an L-system.
    A rule operates on all symbols in a sequence
    in parallel.
    E.g., F -> W or {W}W{W} -> W W.
    """

    def __init__(self, lhs: Lhs, rhs: Rhs):
        self.lhs = lhs
        self.rhs = rhs
        self.recursive = False

    def set_recursive(self):
        self.recursive = True


class LRule(LSystemRule):
    """
    An L-system rewrite rule without conditionals.
    E.g., F -> W or W -> WW.
    """

    def __init__(self, lhs: Lhs, rhs: Rhs):
        super().__init__(lhs=lhs, rhs=rhs)

    def __repr__(self):
        lhs_str = " ".join(s.get().name for s in self.lhs.iterate())
        rhs_str = " ".join(s.get().name for s in self.rhs.iterate())
        return "%s -> %s" % (lhs_str, rhs_str)


class CLRule(LSystemRule):
    """
    A conditional rewrite rule.
    E.g., {W}W{W} -> WW
    """

    def __init__(self, lhs: Lhs, rhs: Rhs,
                 left_condition: List[Symbol],
                 right_condition: List[Symbol]):
        super().__init__(lhs=lhs, rhs=rhs)
        self.lhs.set_conditional(left_condition, right_condition)
        self._left_conditional = left_condition
        self._right_conditional = right_condition

    def __repr__(self):
        lhs_str = str(self.lhs)
        rhs_str = str(self.rhs)
        return "%s -> %s" % (lhs_str, rhs_str)


class MetaGrammar(object):
    """
    A meta-grammar holding a set of L-system rules that can
    be obtained deterministically or sampled to
    combine into an L-system.
    """

    def __init__(self):
        self.rules = {
            "ACTION": {},
            "Walk": {},
            "{ACTION}ACTION ACTION{ACTION}": {},
            "Push": {},
            "Pull": {},
            "East": {},
            "North": {},
            "South": {},
            "West": {},
            "East South": {},
            "East North": {},
            "West South": {},
            "West North": {},
            "South East": {},
            "North East": {},
            "South West": {},
            "North West": {}
        }
        # LHS of rules like while zigzagging
        self._movement_rewrite_rules = {}

        # LHS of rules like while zigzagging
        self._movement_rules = {}

        # LHS of rules like while spinning
        self._nonmovement_direction_rules = {}

        # LHS of rules like cautiously, hesitantly
        self._nonmovement_first_person_rules = {}

        self.add_rules()
        print()

    def get_movement_rewrite_rules(self):
        return self._movement_rewrite_rules.copy()

    def get_movement_rules(self):
        return self._movement_rules.copy()

    def get_nonmovement_direction_rules(self):
        return self._nonmovement_direction_rules.copy()

    def get_nonmovement_first_person_rules(self):
        return self._nonmovement_first_person_rules.copy()

    def get_rule(self, lhs_str: str, rhs_str: str):
        if self.rules.get(lhs_str):
            if self.rules[lhs_str].get(rhs_str):
                return deepcopy(self.rules[lhs_str][rhs_str])
        raise ValueError("Rule doesn't exist.")

    def add_rule(self, rule, movement_rule=False, nonmovement_dir_rule=False,
                 nonmovement_fp_rule=False, movement_rewrite=False):
        self.rules[str(rule.lhs)][str(rule.rhs)] = rule
        if movement_rewrite:
            if str(rule.lhs) not in self._movement_rewrite_rules:
                self._movement_rewrite_rules[(str(rule.lhs))] = []
            self._movement_rewrite_rules[(str(rule.lhs))].append(rule)
        if movement_rule:
            if str(rule.lhs) not in self._movement_rules:
                self._movement_rules[(str(rule.lhs))] = []
            self._movement_rules[(str(rule.lhs))].append(rule)
        if nonmovement_dir_rule:
            if str(rule.lhs) not in self._nonmovement_direction_rules:
                self._nonmovement_direction_rules[(str(rule.lhs))] = []
            self._nonmovement_direction_rules[(str(rule.lhs))].append(rule)
        if nonmovement_fp_rule:
            if str(rule.lhs) not in self._nonmovement_first_person_rules:
                self._nonmovement_first_person_rules[(str(rule.lhs))] = []
            self._nonmovement_first_person_rules[(str(rule.lhs))].append(rule)

    def add_rules(self):
        rule = LRule(lhs=Lhs(symbols=[Walk]),
                     rhs=Rhs(symbols=[ACTION, Walk]))
        rule.set_recursive()
        self.add_rule(rule, nonmovement_fp_rule=True)
        rule = LRule(lhs=Lhs(symbols=[Walk]),
                     rhs=Rhs(symbols=[Walk, ACTION]))
        rule.set_recursive()
        self.add_rule(rule, nonmovement_fp_rule=True)
        rule = LRule(lhs=Lhs(symbols=[ACTION]),
                     rhs=Rhs(symbols=[Tl]))
        self.add_rule(rule)
        rule = LRule(lhs=Lhs(symbols=[ACTION]),
                     rhs=Rhs(symbols=[Tr]))
        self.add_rule(rule)
        rule = LRule(lhs=Lhs(symbols=[ACTION]),
                     rhs=Rhs(symbols=[Stay]))
        self.add_rule(rule)
        rule = CLRule(lhs=Lhs(symbols=[ACTION, ACTION]),
                      rhs=Rhs(symbols=[Tl, Tl]),
                      left_condition=[ACTION],
                      right_condition=[ACTION])
        self.add_rule(rule)
        rule = CLRule(lhs=Lhs(symbols=[ACTION, ACTION]),
                      rhs=Rhs(symbols=[Tr, Tr]),
                      left_condition=[ACTION],
                      right_condition=[ACTION])
        self.add_rule(rule)
        rule = LRule(lhs=Lhs(symbols=[Push]),
                     rhs=Rhs(symbols=[ACTION, Push]))
        rule.set_recursive()
        self.add_rule(rule, nonmovement_dir_rule=True,
                      nonmovement_fp_rule=True)
        rule = LRule(lhs=Lhs(symbols=[Pull]),
                     rhs=Rhs(symbols=[ACTION, Pull]))
        rule.set_recursive()
        self.add_rule(rule, nonmovement_dir_rule=True,
                      nonmovement_fp_rule=True)
        rule = LRule(lhs=Lhs(symbols=[East]),
                     rhs=Rhs(symbols=[ACTION, East]))
        rule.set_recursive()
        self.add_rule(rule, nonmovement_dir_rule=True)
        rule = LRule(lhs=Lhs(symbols=[North]),
                     rhs=Rhs(symbols=[ACTION, North]))
        rule.set_recursive()
        self.add_rule(rule, nonmovement_dir_rule=True)
        rule = LRule(lhs=Lhs(symbols=[South]),
                     rhs=Rhs(symbols=[ACTION, South]))
        rule.set_recursive()
        self.add_rule(rule, nonmovement_dir_rule=True)
        rule = LRule(lhs=Lhs(symbols=[West]),
                     rhs=Rhs(symbols=[ACTION, West]))
        rule.set_recursive()
        self.add_rule(rule, nonmovement_dir_rule=True)
        rule = LRule(lhs=Lhs(symbols=[East]),
                     rhs=Rhs(symbols=[East, ACTION]))
        rule.set_recursive()
        self.add_rule(rule, nonmovement_dir_rule=True)
        rule = LRule(lhs=Lhs(symbols=[North]),
                     rhs=Rhs(symbols=[North, ACTION]))
        rule.set_recursive()
        self.add_rule(rule, nonmovement_dir_rule=True)
        rule = LRule(lhs=Lhs(symbols=[South]),
                     rhs=Rhs(symbols=[South, ACTION]))
        rule.set_recursive()
        self.add_rule(rule, nonmovement_dir_rule=True)
        rule = LRule(lhs=Lhs(symbols=[West]),
                     rhs=Rhs(symbols=[West, ACTION]))
        rule.set_recursive()
        self.add_rule(rule, nonmovement_dir_rule=True)
        rule = LRule(lhs=Lhs(symbols=[Push]),
                     rhs=Rhs(symbols=[Push, ACTION]))
        rule.set_recursive()
        self.add_rule(rule, nonmovement_dir_rule=True,
                      nonmovement_fp_rule=True)
        rule = LRule(lhs=Lhs(symbols=[Pull]),
                     rhs=Rhs(symbols=[Pull, ACTION]))
        rule.set_recursive()
        self.add_rule(rule, nonmovement_dir_rule=True,
                      nonmovement_fp_rule=True)
        rule = LRule(lhs=Lhs(symbols=[East]),
                     rhs=Rhs(symbols=[North, East, South]))
        rule.set_recursive()
        self.add_rule(rule, movement_rule=True)
        rule = LRule(lhs=Lhs(symbols=[East]),
                     rhs=Rhs(symbols=[South, East, North]))
        rule.set_recursive()
        self.add_rule(rule, movement_rule=True)
        rule = LRule(lhs=Lhs(symbols=[North]),
                     rhs=Rhs(symbols=[East, North, West]))
        rule.set_recursive()
        self.add_rule(rule, movement_rule=True)
        rule = LRule(lhs=Lhs(symbols=[North]),
                     rhs=Rhs(symbols=[West, North, East]))
        rule.set_recursive()
        self.add_rule(rule, movement_rule=True)
        rule = LRule(lhs=Lhs(symbols=[South]),
                     rhs=Rhs(symbols=[East, South, West]))
        rule.set_recursive()
        self.add_rule(rule, movement_rule=True)
        rule = LRule(lhs=Lhs(symbols=[South]),
                     rhs=Rhs(symbols=[West, South, East]))
        rule.set_recursive()
        self.add_rule(rule, movement_rule=True)
        rule = LRule(lhs=Lhs(symbols=[West]),
                     rhs=Rhs(symbols=[North, West, South]))
        rule.set_recursive()
        self.add_rule(rule, movement_rule=True)
        rule = LRule(lhs=Lhs(symbols=[West]),
                     rhs=Rhs(symbols=[South, West, North]))
        rule.set_recursive()
        self.add_rule(rule, movement_rule=True)
        rule = LRule(lhs=Lhs(symbols=[East, South]),
                     rhs=Rhs(symbols=[South, East]))
        self.add_rule(rule, movement_rewrite=True)
        rule = LRule(lhs=Lhs(symbols=[East, North]),
                     rhs=Rhs(symbols=[North, East]))
        self.add_rule(rule, movement_rewrite=True)
        rule = LRule(lhs=Lhs(symbols=[West, North]),
                     rhs=Rhs(symbols=[North, West]))
        self.add_rule(rule, movement_rewrite=True)
        rule = LRule(lhs=Lhs(symbols=[West, South]),
                     rhs=Rhs(symbols=[South, West]))
        self.add_rule(rule, movement_rewrite=True)
        rule = LRule(lhs=Lhs(symbols=[North, East]),
                     rhs=Rhs(symbols=[East, North]))
        self.add_rule(rule, movement_rewrite=True)
        rule = LRule(lhs=Lhs(symbols=[North, West]),
                     rhs=Rhs(symbols=[West, North]))
        self.add_rule(rule, movement_rewrite=True)
        rule = LRule(lhs=Lhs(symbols=[South, East]),
                     rhs=Rhs(symbols=[East, South]))
        self.add_rule(rule, movement_rewrite=True)
        rule = LRule(lhs=Lhs(symbols=[South, West]),
                     rhs=Rhs(symbols=[West, South]))
        self.add_rule(rule, movement_rewrite=True)

    def __repr__(self):
        output_str = ""
        for key, values in self.rules.items():
            for k, value in values.items():
                output_str += str(value) + '\n'
            if values.values():
                output_str += '\n'
        return output_str


class LSystem(object):
    """
    An L-system (https://en.wikipedia.org/wiki/L-system)
    with rewrite rules and terminal rules.
    The rules are stored sorted by LHS length, e.g. if
    there are the rules W W -> A W and W -> A W,
    the former ordered first. A terminal rule is a rule
    with only Terminal symbols in its RHS.
    When iterated over this class the rules are presented
    in order.
    An L-system can be applied to a Sequence (see below).
    """

    def __init__(self):
        self._rewrite_rules = {}
        self._lhs_rule_lengths = []
        self._terminal_rules = {}
        self._lhs_terminal_lengths = []
        self._rules_in_order = []
        self._terminal_rules_in_order = []
        self._num_total_rules = 0
        self._finished = False

    def add_rule(self, rule: LSystemRule, terminal_rule=False):
        lhs = str(rule.lhs)
        if lhs in self._rewrite_rules or lhs in self._terminal_rules:
            raise ValueError("LHS {} already in L-System.".format(lhs))
        self._num_total_rules += 1
        lhs_length = len(lhs)
        lhs_lengths = self._lhs_rule_lengths
        rules = self._rewrite_rules
        if terminal_rule:
            rules = self._terminal_rules
            lhs_lengths = self._lhs_terminal_lengths
        if lhs_length not in rules:
            rules[lhs_length] = {}
            lhs_lengths.append(lhs_length)
            lhs_lengths.sort(reverse=True)
        rules[lhs_length][str(rule.lhs)] = rule

    def nonterminal_rules(self):
        return self._rules_in_order.copy()

    def terminal_rules(self):
        return self._terminal_rules_in_order.copy()

    def finish_l_system(self):
        for lhs_length in self._lhs_rule_lengths:
            for lhs, rule in self._rewrite_rules[lhs_length].items():
                self._rules_in_order.append(rule)
        for lhs_length in self._lhs_terminal_lengths:
            for lhs, rule in self._terminal_rules[lhs_length].items():
                self._terminal_rules_in_order.append(rule)
        self._finished = True

    @property
    def is_finished(self):
        if self._finished:
            return True
        else:
            return False

    def __iter__(self):
        if not self._finished:
            raise ValueError(
                "Trying to iterate over an unfinished L-System."
                " Call .finish_l_system()")
        self._current_rule_ptr = 0
        self._current_terminal_ptr = 0
        return self

    def __next__(self):
        if self._current_rule_ptr < len(self._rules_in_order):
            next_rule = self._rules_in_order[self._current_rule_ptr]
            self._current_rule_ptr += 1
            return next_rule, False
        elif self._current_terminal_ptr < len(self._terminal_rules_in_order):
            next_rule = self._terminal_rules_in_order[self._current_terminal_ptr]
            self._current_terminal_ptr += 1
            return next_rule, True
        else:
            raise StopIteration

    def __repr__(self):
        if not self._finished:
            raise ValueError(
                "Trying to iterate over an unfinished L-System."
                " Call .finish_l_system()")
        output_str = ""
        for rule in reversed(self._rules_in_order):
            output_str += str(rule) + '\n'
        output_str += '\n'
        for rule in reversed(self._terminal_rules_in_order):
            output_str += str(rule) + '\n'
        return output_str


def apply_rule(sequence: Sequence, rule: LSystemRule):
    """
    Apply an L-system rule to a sequence by looping over the
    sequence symbol by symbol and finding a match with the LHS
    of the rule, if found, replace with rhs.
    :param sequence: The sequence to apply the rule on, e.g., W Tl W Tl.
    :param rule: The rule to apply to the sequence, e.g. W -> A W.
    """
    # Copy the LHS into a sequence so we can compare it to the
    # current sequence.
    lhs_matching_sequence = rule.lhs
    lhs_matching_length = len(lhs_matching_sequence)
    if lhs_matching_length > len(sequence):
        return
    else:
        # Loop over the current sequence.
        current_sequence_ptr = sequence.get_start()

        # If a symbol has been replaced by another rule in the same
        # iteration, it should not be used for LHS checking again.
        while current_sequence_ptr.is_replaced():
            current_sequence_ptr = current_sequence_ptr.get_next()
            if not current_sequence_ptr:
                break
        while current_sequence_ptr:

            lhs_iterator = lhs_matching_sequence.iterate()

            # Get a part of the sequence to compare to the LHS.
            current_ptr = deepcopy(current_sequence_ptr)
            num_compared = 0
            match = True
            while num_compared < lhs_matching_length and current_ptr and match:
                next_lhs_symbol = next(lhs_iterator)
                if current_ptr.get() != next_lhs_symbol.get() or current_ptr.is_replaced():
                    match = False
                current_ptr = current_ptr.get_next()
                num_compared += 1
            if match and num_compared == lhs_matching_length:
                lhs_replace_length = lhs_matching_length - \
                                     rule.lhs.num_left_conditional() - \
                                     rule.lhs.num_right_conditional()
                # Advance the ptr past the conditional part of the LHS.
                # We don't want to replace that part.
                for _ in range(rule.lhs.num_left_conditional()):
                    current_sequence_ptr = current_sequence_ptr.get_next()
                current_sequence_ptr = sequence.replace(current_sequence_ptr,
                                                        lhs_replace_length,
                                                        rule.rhs)
                # Advance the ptr past the conditional part of the RHS.
                # TODO: couldn't the right part of the conditional still be matched?
                for _ in range(rule.lhs.num_right_conditional()):
                    current_sequence_ptr = current_sequence_ptr.get_next()
            else:
                current_sequence_ptr = current_sequence_ptr.get_next()


def simulate_planner(start_position: Position,
                     end_position: Position) -> Sequence:
    """Returns a sequence of East, West, North, South actions that takes
    you from start_position to end_position on a grid."""

    sequence = Sequence()

    cols_diff = end_position.column - start_position.column
    # If positive, go to the right, if negative, left
    if cols_diff > 0:
        col_dir = East
    elif cols_diff < 0:
        col_dir = West

    rows_diff = end_position.row - start_position.row
    # If positive, go down, if negative, up
    if rows_diff > 0:
        row_dir = South
    elif rows_diff < 0:
        row_dir = North

    for col in range(abs(cols_diff)):
        sequence.append(col_dir)
    for row in range(abs(rows_diff)):
        sequence.append(row_dir)

    return sequence


def apply_lsystem(sequence: Sequence,
                  l_system: LSystem, recursion: int, max_recursion: int):
    """
    Apply's an L-system to the passed sequence for `max_recursion` times.
    :param sequence: The sequence to apply the LSystem on.
    :param l_system: a finished L-system with a set of rules and terminal rules.
    :param recursion: the current recursion.
    :param max_recursion: the maximum recursion.
    """
    if not l_system.is_finished:
        raise ValueError("Cannot apply an unfinished L-system.")
    if max_recursion < 0:
        return
    # Apply the non-terminal rules, from longest to shortest.
    if recursion < max_recursion:
        for rule in l_system.nonterminal_rules():
            apply_rule(sequence, rule)
        # The next iteration, all symbols can be replaced again.
        sequence.unset_replaced()
        return apply_lsystem(sequence, l_system, recursion + 1, max_recursion)
    # If all non-terminal rules are applied max-recursion times,
    # apply terminal rules, longest to shortest.
    else:
        for rule in l_system.terminal_rules():
            apply_rule(sequence, rule)


def apply_recursion(l_system: LSystem, max_recursion: int) -> LSystem:
    """
    Apply's a recursion to each rule in the L-system, for `max_recursion` times.
    :param sequence: The sequence to apply the LSystem on.
    :param l_system: a finished L-system with a set of rules and terminal rules.
    :param recursion: the current recursion.
    :param max_recursion: the maximum recursion.

    Returns: a new L-system with the rules replaced with the recursively changed ones.
    """
    if not l_system.is_finished:
        raise ValueError("Cannot apply an unfinished L-system.")
    if max_recursion <= 0:
        return l_system

    nonterminal_rules = deepcopy(l_system.nonterminal_rules())
    # Apply the non-terminal rules, from longest to shortest.
    for recursion in range(max_recursion - 1):
        for i, rule in enumerate(l_system.nonterminal_rules()):
            if rule.recursive:
                apply_rule(nonterminal_rules[i].rhs, rule)
                nonterminal_rules[i].rhs.unset_replaced()
    for i, rule in enumerate(l_system.terminal_rules()):
        for nt_rule in nonterminal_rules:
            apply_rule(nt_rule.rhs, rule)
            nt_rule.rhs.unset_replaced()
            apply_rule(nt_rule.lhs, rule)
            nt_rule.lhs.unset_replaced()
    new_l_system = LSystem()
    for rule in nonterminal_rules:
        new_l_system.add_rule(rule)
    new_l_system.finish_l_system()
    return new_l_system


class MetaGrammarFunctions(object):

    def __init__(self):

        pass

    def min(self, x: int, y: int) -> int:
        return min(x, y)

    def abs(self, x: int) -> int:
        return abs(x)

    def max(self, x: int, y: int) -> int:
        return max(x, y)

    def concat(self, sequence_1: Sequence, sequence_2: Sequence) -> Sequence:
        # TODO: do inplace? append to sequence_1
        new_sequence = Sequence()
        for symbol in sequence_1.iterate():
            new_sequence.append(symbol)

        for symbol in sequence_2.iterate():
            new_sequence.append(symbol)
        return new_sequence

    def is_empty(self, sequence: Sequence) -> bool:
        return len(sequence) == 0

    def set(self, directions: List[Direction]) -> Set[Direction]:
        return set(directions)

    def is_left(self, direction_1: Direction, direction_2: Direction):
        """Returns whether direction_1 is to the left of direction_2"""
        int_dir_1 = DIR_TO_INT[direction_1]
        int_dir_2 = DIR_TO_INT[direction_2]
        if int_dir_1 - int_dir_2 == 1 or int_dir_1 - int_dir_2 == -3:
            return True
        else:
            return False

    def turn_to_dir(self, current_dir: Direction, desired_dir: Direction,
                    left_first=True) -> List[Symbol]:
        if current_dir == desired_dir:
            return []
        sequence = []
        given_dirs = self.set([current_dir, desired_dir])
        if given_dirs == self.set([NORTH, SOUTH]) or given_dirs == self.set([WEST, EAST]):
            if left_first:
                sequence.extend([Tl, Tl])
            else:
                sequence.extend([Tr, Tr])
            return sequence
        else:
            if self.is_left(current_dir, desired_dir):
                sequence.append(Tl)
                return sequence
            else:
                sequence.append(Tr)
                return sequence

    def new_dir(self, direction: Direction, action: str):
        if direction == EAST and action == "Tl":
            return NORTH
        if direction == EAST and action == "Tr":
            return SOUTH
        if direction == SOUTH and action == "Tl":
            return EAST
        if direction == SOUTH and action == "Tr":
            return WEST
        if direction == WEST and action == "Tl":
            return SOUTH
        if direction == WEST and action == "Tr":
            return NORTH
        if direction == NORTH and action == "Tl":
            return WEST
        if direction == NORTH and action == "Tr":
            return EAST

    def is_turn(self, action: str):
        if action == "Tr" or action == "Tl":
            return True
        else:
            return False

    def step(self, direction: Direction, position: Position):
        if direction == EAST:
            position = Position(column=position.column + 1,
                                row=position.row)
            return position
        if direction == NORTH:
            position = Position(column=position.column,
                                row=position.row - 1)
            return position
        if direction == SOUTH:
            position = Position(column=position.column,
                                row=position.row + 1)
            return position
        if direction == WEST:
            position = Position(column=position.column - 1,
                                row=position.row)
            return position


def convert_sequence_to_actions(sequence: Sequence,
                                agent_start_dir: Direction) -> Sequence:
    """
    Converts a sequence containing East, West, North, South
    to a sequence of actions from an agents perspective
    :param sequence: a sequence of steps in the form of East, South, North, West symbols.
    :param agent_start_dir: direction that the agent starts facing.
    :return: a sequence of actions in the form of Tl, Walk, Tr symbols.
    """
    meta_functions = MetaGrammarFunctions()
    new_sequence = Sequence()
    current_agent_dir = agent_start_dir
    for symbol_node in sequence.iterate():
        symbol = symbol_node.get()
        if symbol in SYMBOL_CONVERT.keys():
            turn_actions = meta_functions.turn_to_dir(current_agent_dir,
                                                      SYMBOL_CONVERT[symbol])
            new_sequence.extend(turn_actions)
            new_sequence.append(Walk)
            current_agent_dir = SYMBOL_CONVERT[symbol]
        else:
            if meta_functions.is_turn(symbol.name):
                current_agent_dir = meta_functions.new_dir(current_agent_dir, symbol.name)
            new_sequence.append(symbol)
    return new_sequence


def get_free_cells_in_direction(from_position: Position, direction: Direction,
                                grid: Grid) -> int:
    """

    :param from_position: the position in the grid to start from
    :param direction: the direction to look for free cells
    :param grid: the class with the initialized grid with objects
    :return: an integer with the numnber of free cells
    """
    num_actions = 0
    grid_size = grid.width
    target_pos = (from_position.column, from_position.row)
    current_pos = target_pos
    object_at_next_cell = False
    while not object_at_next_cell:
        current_pos = current_pos + DIR_TO_VEC[DIR_TO_INT[direction]]
        if 0 <= current_pos[1] < grid_size and 0 <= current_pos[0] < grid_size:
            object_at_next_cell = grid.get(*current_pos)
            if not object_at_next_cell:
                num_actions += 1
        else:
            object_at_next_cell = True
    return num_actions


def get_num_push_actions(last_move_direction: Direction, target_position: Position,
                         grid: Grid) -> int:
    """

    :param last_move_direction: last direction the agent moved in
    :param target_position: the column and row of the target
    :param grid: the class with the initialized grid with objects
    :return: an integer denoting how many free space there is between
    the target and the direction it can move in.
    """
    num_actions = get_free_cells_in_direction(target_position, last_move_direction,
                                              grid)
    return num_actions


def get_num_pull_actions(last_move_direction: Direction, target_position: Position,
                         grid: Grid) -> int:
    """

    :param last_move_direction: last direction the agent moved in
    :param target_position: the column and row of the target
    :param grid: the class with the initialized grid with objects
    :return: an integer denoting how many free space there is between
    the target and the direction it can move in.
    """
    opposite_direction = INT_TO_DIR[(DIR_TO_INT[last_move_direction] + 2) % 4]
    num_actions = get_free_cells_in_direction(target_position, opposite_direction,
                                              grid)
    return num_actions


class World(object):

    def __init__(self, num_adverbs: int, seed: int):
        random.seed(seed)
        self.meta_grammar = MetaGrammar()
        self.functions = MetaGrammarFunctions()

    def get_rules(self, type: str):
        if type == "movement_rewrite":
            return self.meta_grammar.get_movement_rewrite_rules()
        elif type == "movement":
            return self.meta_grammar.get_movement_rules()
        elif type == "nonmovement_direction":
            return self.meta_grammar.get_nonmovement_direction_rules()
        elif type == "nonmovement_first_person":
            return self.meta_grammar.get_nonmovement_first_person_rules()
        else:
            raise ValueError("Unknown type {}".format(type))

    def sample_adverb(self):
        # Timing 0,1,2 work on direction sequences (like while zigzagging and while spinning)
        # Timing 1 doesn't exist in gSCAN (applied to directions but not to push/pull)
        # Timing 3 works on first-person sequences (like cautiously and hesitantly)
        application_timing_options = [0, 1, 2, 3]
        sampled_timing = random.sample(application_timing_options)
        pass

    def generate_all_adverbs(self, type: str):
        rules = self.get_rules(type)
        rule_list = list(rules.values())

        # Make sure not adding the rule is represented in the choices.
        nonterminal_set = set()
        all_nonterminal_rules = []
        new_rule_list = []
        for rules in rule_list:
            new_rules = []
            for rule in rules:
                new_rule = [rule]
                has_nonterminal = False
                non_terminal = None
                for symbol_node in rule.rhs.iterate():
                    symbol = symbol_node.get()
                    if isinstance(symbol, Nonterminal):
                        has_nonterminal = True
                        non_terminal = symbol
                        if symbol not in nonterminal_set:
                            nonterminal_set.add(symbol)
                            nonterminal_rules = self.meta_grammar.rules[symbol.name]
                            all_nonterminal_rules.append(list(nonterminal_rules.values()))
                new_rule.append(has_nonterminal)
                new_rule.append(non_terminal)
                new_rules.append(new_rule)
            new_rules.append([None, False, None])
            rules.append([None, False, None])
            new_rule_list.append(new_rules)

        # TODO: hier gebleven
        all_l_systems_combinations = itertools.product(*new_rule_list)
        all_l_systems = []
        for combinations in all_l_systems_combinations:
            has_nonterminals = set([has_nonterminal for _, has_nonterminal, _ in combinations])
            if True in has_nonterminals:
                nonterminals = set([nonterminal for _, _, nonterminal in combinations])
                for nt in nonterminals:
                    if nt:
                        nt_rules = self.meta_grammar.rules[nt.name]
                        for nt_rule in nt_rules.values():
                            new_combinations = list(deepcopy(combinations))
                            new_combinations.append([nt_rule, False, None])
                            adverb_l_system = LSystem()
                            for rule_repr, has_nonterminal, nt in new_combinations:
                                if rule_repr:
                                    rule = self.meta_grammar.get_rule(lhs_str=str(rule_repr.lhs),
                                                                      rhs_str=str(rule_repr.rhs))
                                    terminal_rule = not has_nonterminal
                                    adverb_l_system.add_rule(rule, terminal_rule=terminal_rule)
                            adverb_l_system.finish_l_system()
                            all_l_systems.append(adverb_l_system)
            else:
                adverb_l_system = LSystem()
                for rule_repr, has_nonterminal, nt in combinations:
                    if rule_repr:
                        rule = self.meta_grammar.get_rule(lhs_str=str(rule_repr.lhs),
                                                          rhs_str=str(rule_repr.rhs))
                        adverb_l_system.add_rule(rule)
                        # for symbol in rule_repr.rhs.iterate():
                        #     if symbol in self.meta_grammar.rules.keys():
                        #         potential_rules = self.meta_grammar.rules[symbol.name]
                        #         for
                adverb_l_system.finish_l_system()
                all_l_systems.append(adverb_l_system)
        return all_l_systems

    def planner(self, start_position: Position, end_position: Position) -> Sequence:
        planned_sequence = simulate_planner(start_position, end_position)
        return planned_sequence

    def apply_adverb(self, adverb: LSystem, sequence: Sequence,
                     recursion_depth: int) -> Sequence:
        apply_lsystem(sequence, adverb,
                      recursion=0, max_recursion=recursion_depth)
        return sequence

    def remove_out_of_grid(self, direction_sequence: Sequence,
                           grid_size: int, start_position: Position):
        start_pos = (start_position.column, start_position.row)
        current_pos = start_pos
        to_delete = []
        direction_node = direction_sequence.get_start()
        while direction_node:
            symbol = direction_node.get()
            if symbol in SYMBOL_CONVERT.keys():
                direction = SYMBOL_CONVERT[symbol]
                if direction in to_delete:
                    direction_node = direction_sequence.delete_node(direction_node)
                    to_delete.remove(direction)
                else:
                    next_pos = current_pos + DIR_TO_VEC[DIR_TO_INT[direction]]
                    next_col, next_row = next_pos
                    if not 0 <= next_col <= grid_size - 1 or not 0 <= next_row <= grid_size - 1:
                        opposite_direction = GET_OPPOSITE_DIR[direction]
                        to_delete.append(opposite_direction)
                        direction_node = direction_sequence.delete_node(direction_node)
                    else:
                        current_pos = next_pos
                        direction_node = direction_node.get_next()
            else:
                direction_node = direction_node.get_next()

    def reject_first_person_adverb(self, l_system: LSystem) -> bool:
        start_direction = EAST
        current_direction = start_direction
        end_directions = set()
        for rule in l_system.nonterminal_rules():
            rhs = rule.rhs
            current_direction = start_direction
            for symbol_node in rhs.iterate():
                symbol = symbol_node.get()
                if self.functions.is_turn(symbol.name):
                    current_direction = self.functions.new_dir(current_direction,
                                                               symbol.name)
            end_directions.add(current_direction)
        if len(end_directions) == 0:
            return False
        elif len(end_directions) > 1:
            return True
        elif start_direction == end_directions.pop():
            return False
        else:
            return True

    def generate_example(self, start_position: Position,
                         start_direction: int, end_position: Position,
                         manner: LSystem, recursion_depth_system: int,
                         recursion_depth_sequence: int,
                         grid_size: int, type_adverb: str) -> List[str]:
        start_to_end_sequence = self.planner(start_position, end_position)
        start_direction = INT_TO_DIR[start_direction]
        if recursion_depth_system >= 0:
            manner = apply_recursion(manner, recursion_depth_system)
        if type_adverb != "nonmovement_first_person":
            start_to_end_sequence = self.apply_adverb(manner,
                                                      start_to_end_sequence,
                                                      recursion_depth_sequence)
        self.remove_out_of_grid(start_to_end_sequence,
                                grid_size,
                                start_position)

        action_sequence = convert_sequence_to_actions(start_to_end_sequence,
                                                      start_direction)
        if type_adverb == "nonmovement_first_person":
            reject_lsystem = self.reject_first_person_adverb(manner)
            if reject_lsystem:
                return action_sequence.get_gscan_actions(), True
            action_sequence = self.apply_adverb(manner,
                                                action_sequence,
                                                recursion_depth_sequence)
        return action_sequence.get_gscan_actions(), False

    def sample_l_system(self):
        pass


class Gscan(object):

    def __init__(self):
        meta_grammar = MetaGrammar()
        self._hesitantly = LSystem()
        self._hesitantly.add_rule(meta_grammar.get_rule(lhs_str="Walk",
                                                        rhs_str="Walk ACTION"))
        self._hesitantly.add_rule(meta_grammar.get_rule(lhs_str="Push",
                                                        rhs_str="Push ACTION"))
        self._hesitantly.add_rule(meta_grammar.get_rule(lhs_str="Pull",
                                                        rhs_str="Pull ACTION"))
        self._hesitantly.add_rule(meta_grammar.get_rule(lhs_str="ACTION",
                                                        rhs_str="Stay"),
                                  terminal_rule=True)
        self._hesitantly.finish_l_system()
        self._hesitantly = apply_recursion(self._hesitantly, max_recursion=1)

        self._cautiously = LSystem()
        self._cautiously.add_rule(meta_grammar.get_rule(lhs_str="Walk",
                                                        rhs_str="ACTION Walk"))
        self._cautiously.add_rule(meta_grammar.get_rule(lhs_str="Push",
                                                        rhs_str="ACTION Push"))
        self._cautiously.add_rule(meta_grammar.get_rule(lhs_str="Pull",
                                                        rhs_str="ACTION Pull"))
        self._cautiously.add_rule(meta_grammar.get_rule(lhs_str="{ACTION}ACTION ACTION{ACTION}",
                                                        rhs_str="Tr Tr"),
                                  terminal_rule=True)
        self._cautiously.add_rule(meta_grammar.get_rule(lhs_str="ACTION",
                                                        rhs_str="Tl"),
                                  terminal_rule=True)
        self._cautiously.finish_l_system()
        self._cautiously = apply_recursion(self._cautiously, max_recursion=4)

        self._while_spinning = LSystem()
        self._while_spinning.add_rule(meta_grammar.get_rule(lhs_str="East",
                                                            rhs_str="ACTION East"))
        self._while_spinning.add_rule(meta_grammar.get_rule(lhs_str="North",
                                                            rhs_str="ACTION North"))
        self._while_spinning.add_rule(meta_grammar.get_rule(lhs_str="South",
                                                            rhs_str="ACTION South"))
        self._while_spinning.add_rule(meta_grammar.get_rule(lhs_str="West",
                                                            rhs_str="ACTION West"))
        self._while_spinning.add_rule(meta_grammar.get_rule(lhs_str="Pull",
                                                            rhs_str="ACTION Pull"))
        self._while_spinning.add_rule(meta_grammar.get_rule(lhs_str="ACTION",
                                                            rhs_str="Tl"),
                                      terminal_rule=True)
        self._while_spinning.add_rule(meta_grammar.get_rule(lhs_str="Push",
                                                            rhs_str="ACTION Push"))
        self._while_spinning.finish_l_system()
        self._while_spinning = apply_recursion(self._while_spinning, max_recursion=4)

        self._while_zigzagging = LSystem()
        self._while_zigzagging.add_rule(meta_grammar.get_rule(lhs_str="East South",
                                                              rhs_str="South East"))
        self._while_zigzagging.add_rule(meta_grammar.get_rule(lhs_str="East North",
                                                              rhs_str="North East"))
        self._while_zigzagging.add_rule(meta_grammar.get_rule(lhs_str="West North",
                                                              rhs_str="North West"))
        self._while_zigzagging.add_rule(meta_grammar.get_rule(lhs_str="West South",
                                                              rhs_str="South West"))
        self._while_zigzagging.finish_l_system()

    def parse_gscan_example(self, verb_in_command: str, start_position: Position,
                            start_direction: int,
                            end_position: Position, manner: str, grid: Grid,
                            heavy: bool) -> List[str]:
        start_to_end_sequence = self.walk(start_position, end_position)
        start_direction = INT_TO_DIR[start_direction]
        if manner == "while zigzagging":
            recursion_depth = abs(end_position.column - start_position.column) - 1
            start_to_end_sequence = self.zigzag(start_to_end_sequence, recursion_depth)
        if verb_in_command == "push":
            self.push(start_to_end_sequence,
                      grid,
                      end_position, heavy)
        elif verb_in_command == "pull":
            self.pull(start_to_end_sequence,
                      grid,
                      end_position, heavy)
        if manner == "while spinning":
            self.spin(start_to_end_sequence, recursion_depth=1)
        action_sequence = convert_sequence_to_actions(start_to_end_sequence,
                                                      start_direction)
        if manner == "cautiously":
            self.cautiously(action_sequence, recursion_depth=1)
        if manner == "hesitantly":
            self.hesitantly(action_sequence, recursion_depth=1)
        return action_sequence.get_gscan_actions()

    def walk(self, start_position: Position, end_position: Position) -> Sequence:
        planned_sequence = simulate_planner(start_position, end_position)
        return planned_sequence

    def spin(self, sequence: Sequence, recursion_depth: int) -> Sequence:
        apply_lsystem(sequence, self._while_spinning,
                      recursion=0, max_recursion=recursion_depth)
        return sequence

    def zigzag(self, sequence: Sequence, recursion_depth: int) -> Sequence:
        apply_lsystem(sequence, self._while_zigzagging,
                      recursion=0, max_recursion=recursion_depth)
        return sequence

    def cautiously(self, sequence: Sequence, recursion_depth: int) -> Sequence:
        apply_lsystem(sequence, self._cautiously,
                      recursion=0, max_recursion=recursion_depth)
        return sequence

    def hesitantly(self, sequence: Sequence, recursion_depth: int) -> Sequence:
        apply_lsystem(sequence, self._hesitantly,
                      recursion=0, max_recursion=recursion_depth)
        return sequence

    def push(self, sequence: Sequence, grid: Grid,
             target_position: Position, heavy: bool) -> Sequence:
        # TODO: here or outside of this class and just use num_actions?
        last_direction = sequence.get_end().get()
        last_direction = SYMBOL_CONVERT[last_direction]
        num_push = get_num_push_actions(last_direction, target_position, grid)
        if heavy:
            num_push *= 2
        for _ in range(num_push):
            sequence.append(Push)
        return sequence

    def pull(self, sequence: Sequence, grid: Grid,
             target_position: Position, heavy: bool) -> Sequence:
        # TODO: here or outside of this class and just use num_actions?
        last_direction = sequence.get_end().get()
        last_direction = SYMBOL_CONVERT[last_direction]
        num_pull = get_num_pull_actions(last_direction, target_position, grid)
        if heavy:
            num_pull *= 2
        for _ in range(num_pull):
            sequence.append(Pull)
        return sequence
