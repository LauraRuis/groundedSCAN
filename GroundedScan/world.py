from collections import namedtuple
import itertools
import os
import numpy as np
from typing import Tuple
from typing import List
from typing import Dict
import random
from itertools import product

from GroundedScan.gym_minigrid.minigrid import MiniGridEnv
from GroundedScan.gym_minigrid.minigrid import Grid
from GroundedScan.gym_minigrid.minigrid import IDX_TO_OBJECT
from GroundedScan.gym_minigrid.minigrid import OBJECT_TO_IDX
from GroundedScan.gym_minigrid.minigrid import Circle
from GroundedScan.gym_minigrid.minigrid import Square
from GroundedScan.gym_minigrid.minigrid import Cylinder
from GroundedScan.gym_minigrid.minigrid import DIR_TO_VEC
from GroundedScan.helpers import one_hot
from GroundedScan.helpers import generate_possible_object_names
from GroundedScan.helpers import numpy_array_to_image


SemType = namedtuple("SemType", "name")
Position = namedtuple("Position", "column row")
Object = namedtuple("Object", "size color shape")
PositionedObject = namedtuple("PositionedObject", "object position vector", defaults=(None, None, None))
Variable = namedtuple("Variable", "name sem_type")
fields = ("action", "is_transitive", "manner", "adjective_type", "noun")
Weights = namedtuple("Weights", fields, defaults=(None, ) * len(fields))

ENTITY = SemType("noun")
COLOR = SemType("color")
SIZE = SemType("size")
EVENT = SemType("verb")

Direction = namedtuple("Direction", "name")
NORTH = Direction("north")
SOUTH = Direction("south")
WEST = Direction("west")
EAST = Direction("east")
FORWARD = Direction("forward")

DIR_TO_INT = {
    NORTH: 3,
    SOUTH: 1,
    WEST: 2,
    EAST: 0
}

INT_TO_DIR = {direction_int: direction for direction, direction_int in DIR_TO_INT.items()}

SIZE_TO_INT = {
    "small": 1,
    "average": 2,
    "big": 3
}

# TODO put somewhere different

ACTIONS_DICT = {
    "light": "push",
    "heavy": "push push"
}

DIR_STR_TO_DIR = {
    "n": NORTH,
    "e": EAST,
    "s": SOUTH,
    "w": WEST,
}

DIR_VEC_TO_DIR = {
    (1, 0): "e",
    (0, 1): "n",
    (-1, 0): "w",
    (0, -1): "s",
    (1, 1): "ne",
    (1, -1): "se",
    (-1, -1): "sw",
    (-1, 1): "nw"
}


Command = namedtuple("Command", "action event")
UNK_TOKEN = 'UNK'


class Term(object):
    """
    Holds terms that can be parts of logical forms and take as arguments variables that the term can operate over.
    E.g. for the phrase 'Brutus stabs Caesar' the term is stab(B, C) which will be represented by the string
    "(stab B:noun C:noun)".
    """

    def __init__(self, function: str, args: tuple, weights=None, meta=None, specs=None):
        self.function = function
        self.arguments = args
        self.weights = weights
        self.meta = meta
        self.specs = specs

    def replace(self, var_to_find: Variable, replace_by_var: Variable):
        """Find a variable `var_to_find` the arguments and replace it by `replace_by_var`."""
        return Term(
            function=self.function,
            args=tuple(replace_by_var if variable == var_to_find else variable for variable in self.arguments),
            specs=self.specs,
            meta=self.meta
        )

    def to_predicate(self, predicate: dict):
        assert self.specs is not None
        output = self.function
        if self.specs.noun:
            predicate["noun"] = output
        elif self.specs.adjective_type == SIZE:
            predicate["size"] = output
        elif self.specs.adjective_type == COLOR:
            predicate["color"] = output

    def __repr__(self):
        parts = [self.function]
        for variable in self.arguments:
            parts.append("{}:{}".format(variable.name, variable.sem_type.name))
        return "({})".format(" ".join(parts))


class LogicalForm(object):
    """
    Holds neo-Davidsonian-like logical forms (http://ling.umd.edu//~alxndrw/LectureNotes07/neodavidson_intro07.pdf).
    An object LogicalForm(variables=[x, y, z], terms=[t1, t2]) may represent
    lambda x, y, z: and(t1(x, y, z), t2(x, y, z)) (depending on which terms involve what variables).
    """

    def __init__(self, variables: Tuple[Variable], terms: Tuple[Term]):
        self.variables = variables
        self.terms = terms
        if len(variables) > 0:
            self.head = variables[0]

    def bind(self, bind_var: Variable):
        """
        Bind a variable to its head, e.g for 'kick the ball', 'kick' is the head and 'the ball' will be bind to it.
        Or in the case of NP -> JJ NP, bind the JJ (adjective) to the head of the noun-phrase.
        E.g. 'the big red square', bind 'big' to 'square'.
        :param bind_var:
        :return:
        """
        sub_var, variables_out = self.variables[0], self.variables[1:]
        # assert sub_var.sem_type == bind_var.sem_type
        terms_out = [term.replace(sub_var, bind_var) for term in self.terms]
        return LogicalForm(variables=(bind_var,) + variables_out, terms=tuple(terms_out))

    def select(self, variables: list, exclude=frozenset()):
        """Select and return the sub-logical form of the variables in the variables list."""
        queue = list(variables)
        used_vars = set()
        terms_out = []
        while len(queue) > 0:
            var = queue.pop()
            deps = [term for term in self.terms if term.function not in exclude and term.arguments[0] == var]
            for term in deps:
                terms_out.append(term)
                used_vars.add(var)
                for variable in term.arguments[1:]:
                    if variable not in used_vars:
                        queue.append(variable)

        vars_out = [var for var in self.variables if var in used_vars]
        terms_out = list(set(terms_out))
        return LogicalForm(tuple(vars_out), tuple(terms_out))

    def to_predicate(self):
        assert len(self.variables) == 1
        predicate = {"noun": "", "size": "", "color": ""}
        [term.to_predicate(predicate) for term in self.terms]
        object_str = ""
        if predicate["color"]:
            object_str += ' ' + predicate["color"]
        object_str += ' ' + predicate["noun"]
        object_str = object_str.strip()
        return object_str, predicate

    def __repr__(self):
        return "LF({})".format(" ^ ".join([repr(term) for term in self.terms]))


def object_to_repr(object: Object) -> dict:
    return {
        "shape": object.shape,
        "color": object.color,
        "size": str(object.size)
    }


def position_to_repr(position: Position) -> dict:
    return {
        "row": str(position.row),
        "column": str(position.column)
    }


def positioned_object_to_repr(positioned_object: PositionedObject) -> dict:
    return {
        "vector": ''.join([str(idx) for idx in positioned_object.vector]),
        "position": position_to_repr(positioned_object.position),
        "object": object_to_repr(positioned_object.object)
    }


def parse_object_repr(object_repr: dict) -> Object:
    return Object(shape=object_repr["shape"], color=object_repr["color"], size=int(object_repr["size"]))


def parse_position_repr(position_repr: dict) -> Position:
    return Position(column=int(position_repr["column"]), row=int(position_repr["row"]))


def parse_object_vector_repr(object_vector_repr: str) -> np.ndarray:
    return np.array([int(idx) for idx in object_vector_repr])


def parse_positioned_object_repr(positioned_object_repr: dict):
    return PositionedObject(object=parse_object_repr(positioned_object_repr["object"]),
                            position=parse_position_repr(positioned_object_repr["position"]),
                            vector=parse_object_vector_repr(positioned_object_repr["vector"]))


class Situation(object):
    """
    Specification of a situation that can be used for serialization as well as initialization of a world state.
    """
    def __init__(self, grid_size: int, agent_position: Position, agent_direction: Direction,
                 target_object: PositionedObject, placed_objects: List[PositionedObject], carrying=None):
        self.grid_size = grid_size
        self.agent_pos = agent_position  # position is [col, row] (i.e. [x-axis, y-axis])
        self.agent_direction = agent_direction
        self.placed_objects = placed_objects
        self.carrying = carrying  # The object the agent is carrying
        self.target_object = target_object

    @property
    def distance_to_target(self):
        """Number of grid steps to take to reach the target position from the agent position."""
        return abs(self.agent_pos.column - self.target_object.position.column) + \
               abs(self.agent_pos.row - self.target_object.position.row)

    @property
    def direction_to_target(self):
        """Direction to the target in terms of north, east, south, north-east, etc. Needed for a grounded scan split."""
        column_distance = self.target_object.position.column - self.agent_pos.column
        column_distance = min(max(-1, column_distance), 1)
        row_distance = self.agent_pos.row - self.target_object.position.row
        row_distance = min(max(-1, row_distance), 1)
        return DIR_VEC_TO_DIR[(column_distance, row_distance)]

    def to_dict(self) -> dict:
        """Represent this situation in a dictionary."""
        return {
            "agent_position": Position(column=self.agent_pos[0], row=self.agent_pos[1]),
            "agent_direction": self.agent_direction,
            "target_object": self.target_object,
            "grid_size": self.grid_size,
            "objects": self.placed_objects,
            "carrying": self.carrying
        }

    def to_representation(self) -> dict:
        """Represent this situation in serializable dict that can be written to a file."""
        return {
            "grid_size": self.grid_size,
            "agent_position": position_to_repr(self.agent_pos),
            "agent_direction": DIR_TO_INT[self.agent_direction],
            "target_object": positioned_object_to_repr(self.target_object) if self.target_object else None,
            "distance_to_target": str(self.distance_to_target) if self.target_object else None,
            "direction_to_target": self.direction_to_target if self.target_object else None,
            "placed_objects":  {str(i): positioned_object_to_repr(placed_object) for i, placed_object
                                in enumerate(self.placed_objects)},
            "carrying_object": object_to_repr(self.carrying) if self.carrying else None
        }

    @classmethod
    def from_representation(cls, situation_representation: dict):
        """Initialize this class by some situation as represented by .to_representation()."""
        target_object = situation_representation["target_object"]
        carrying_object = situation_representation["carrying_object"]
        placed_object_reps = situation_representation["placed_objects"]
        placed_objects = []
        for placed_object_rep in placed_object_reps.values():
            placed_objects.append(parse_positioned_object_repr(placed_object_rep))
        situation = cls(grid_size=situation_representation["grid_size"],
                        agent_position=parse_position_repr(situation_representation["agent_position"]),
                        agent_direction=INT_TO_DIR[situation_representation["agent_direction"]],
                        target_object=parse_positioned_object_repr(target_object) if target_object else None,
                        placed_objects=placed_objects,
                        carrying=parse_object_repr(carrying_object) if carrying_object else None)
        return situation

    def __eq__(self, other) -> bool:
        """Recursive function to compare this situation to another and determine if they are equivalent."""
        representation_other = other.to_representation()
        representation_self = self.to_representation()

        def compare_nested_dict(value_1, value_2, unequal_values):
            if len(unequal_values) > 0:
                return
            if isinstance(value_1, dict):
                for k, v_1 in value_1.items():
                    v_2 = value_2.get(k)
                    if not v_2 and v_1:
                        unequal_values.append(False)
                    compare_nested_dict(v_1, v_2, unequal_values)
            else:
                if value_1 != value_2:
                    unequal_values.append(False)
            return
        result = []
        compare_nested_dict(representation_self, representation_other, result)
        return not len(result) > 0


class ObjectVocabulary(object):
    """
    Constructs an object vocabulary. Each object will be calculated by the following:
    [size color shape] and where size is on an ordinal scale of 1 (smallest) to 4 (largest),
    colors and shapes are orthogonal vectors [0 1] and [1 0] and the result is a concatenation:
    e.g. the biggest red circle: [4 0 1 0 1], the smallest blue square: [1 1 0 1 0]
    """
    SIZES = list(range(1, 5))

    def __init__(self, shapes: List[str], colors: List[str], min_size: int, max_size: int):
        """
        :param shapes: a list of string names for nouns.
        :param colors: a list of string names for colors.
        :param min_size: minimum object size
        :param max_size: maximum object size
        """
        assert self.SIZES[0] <= min_size <= max_size <= self.SIZES[-1], \
            "Unsupported object sizes (min: {}, max: {}) specified.".format(min_size, max_size)
        self._min_size = min_size
        self._max_size = max_size

        # Translation from shape nouns to shapes.
        self._shapes = set(shapes)
        self._n_shapes = len(self._shapes)
        self._colors = set(colors)
        self._n_colors = len(self._colors)
        self._idx_to_shapes_and_colors = shapes + colors
        self._shapes_and_colors_to_idx = {token: i for i, token in enumerate(self._idx_to_shapes_and_colors)}
        self._sizes = list(range(min_size, max_size + 1))

        # Also size specification for 'average' size, e.g. if adjectives are small and big, 3 sizes exist.
        self._n_sizes = len(self._sizes)
        assert (self._n_sizes % 2) == 0, "Please specify an even amount of sizes "\
                                         " (needs to be split in 2 classes.)"
        self._middle_size = (max_size + min_size) // 2

        # Make object classes.
        self._object_class = {i: "light" for i in range(min_size, self._middle_size + 1)}
        self._heavy_weights = {i: "heavy" for i in range(self._middle_size + 1, max_size + 1)}
        self._object_class.update(self._heavy_weights)

        # Prepare object vectors.
        self._object_vector_size = self._n_shapes + self._n_colors + self._n_sizes
        self._object_vectors = self.generate_objects()
        self._possible_colored_objects = set([color + ' ' + shape for color, shape in itertools.product(self._colors,
                                                                                                        self._shapes)])

    def has_object(self, shape: str, color: str, size: int):
        return shape in self._shapes and color in self._colors and size in self._sizes

    def object_in_class(self, size: int):
        return self._object_class[size]

    @property
    def num_object_attributes(self):
        """Dimension of object vectors is one hot for shapes and colors + 1 ordinal dimension for size."""
        return len(self._idx_to_shapes_and_colors) + self._n_sizes

    @property
    def smallest_size(self):
        return self._min_size

    @property
    def largest_size(self):
        return self._max_size

    @property
    def object_shapes(self):
        return self._shapes.copy()

    @property
    def object_sizes(self):
        return self._sizes.copy()

    @property
    def object_colors(self):
        return self._colors.copy()

    @property
    def all_objects(self):
        return product(self.object_sizes, self.object_colors, self.object_shapes)

    def sample_size(self):
        return random.choice(self._sizes)

    def sample_color(self):
        return random.choice(list(self._colors))

    def get_object_vector(self, shape: str, color: str, size: int) -> np.ndarray:
        assert self.has_object(shape, color, size), "Trying to get an unavailable object vector from the vocabulary/"
        return self._object_vectors[shape][color][size]

    def generate_objects(self) -> Dict[str, Dict[str, Dict[str, np.ndarray]]]:
        """
        An object vector is built as follows: the first entry is an ordinal entry defining the size (from 1 the smallest
        to 4 the largest), then 2 entries define a one-hot vector over shape, the last two entries define a one-hot
        vector over color. A red circle of size 1 could then be: [1 0 1 0 1], meaning a blue square of size 2 would be
        [2 1 0 1 0].
        """
        object_to_object_vector = {}
        for size, color, shape in itertools.product(self._sizes, self._colors, self._shapes):
            object_vector = one_hot(self._object_vector_size, size - 1) + \
                            one_hot(self._object_vector_size, self._shapes_and_colors_to_idx[color] + self._n_sizes) + \
                            one_hot(self._object_vector_size, self._shapes_and_colors_to_idx[shape] + self._n_sizes)
            # object_vector = np.concatenate(([size], object_vector))
            if shape not in object_to_object_vector.keys():
                object_to_object_vector[shape] = {}
            if color not in object_to_object_vector[shape].keys():
                object_to_object_vector[shape][color] = {}
            object_to_object_vector[shape][color][size] = object_vector

        return object_to_object_vector


class World(MiniGridEnv):
    """
    Wrapper class to execute actions in a world state. Connected to minigrid.py in gym_minigrid for visualizations.
    Every time actions are executed, the commands and situations are saved in self._observed_commands and
    self._observed_situations, which can then be retrieved with get_current_observations().
    The world can be cleared with clear_situation().
    """

    AVAILABLE_SHAPES = {"circle", "square", "cylinder"}
    AVAILABLE_COLORS = {"red", "blue", "green", "yellow"}

    def __init__(self, grid_size: int, shapes: List[str], colors: List[str], object_vocabulary: ObjectVocabulary,
                 save_directory: str):
        # Some checks on the input
        for shape, color in zip(shapes, colors):
            assert shape in self.AVAILABLE_SHAPES, "Specified shape {} not implemented in minigrid env.".format(shape)
            assert color in self.AVAILABLE_COLORS, "Specified color {}, not implemented in minigrid env.".format(color)

        # Define the grid world.
        self.grid_size = grid_size

        # Column, row
        self.agent_start_pos = (0, 0)
        self.agent_start_dir = DIR_TO_INT[EAST]
        self.mission = None

        # Generate the object vocabulary.
        self._object_vocabulary = object_vocabulary
        self.num_available_objects = len(IDX_TO_OBJECT.keys())
        self.available_objects = set(OBJECT_TO_IDX.keys())

        # Data structures for keeping track of the current state of the world.
        self._placed_object_list = []
        self._target_object = None
        self._observed_commands = []
        self._observed_situations = []
        self._occupied_positions = set()
        # Hash table for looking up locations of objects based on partially formed references (e.g. find the location(s)
        # of a red cylinder when the grid has both a big red cylinder and a small red cylinder.)
        self._object_lookup_table = {}
        self.save_directory = save_directory
        super().__init__(grid_size=grid_size, max_steps=4 * grid_size * grid_size)

    def _gen_grid(self, width, height):
        # Create an empty grid
        self.grid = Grid(width, height, depth=self._object_vocabulary.num_object_attributes)

        # Place the agent
        if self.agent_start_pos is not None:
            self.agent_pos = self.agent_start_pos
            self.agent_dir = self.agent_start_dir
        else:
            position = self.place_agent()
            self._occupied_positions.add(position)

    def initialize(self, objects: List[Tuple[Object, Position]], agent_position: Position, agent_direction: Direction,
                   target_object: PositionedObject, carrying: Object=None):
        """
        Create a grid world by placing the objects that are passed as an argument at the specified locations and the
        agent at the specified location.
        """
        self.clear_situation()
        self.agent_dir = DIR_TO_INT[agent_direction]
        self.place_agent_at(agent_position)
        self._target_object = target_object
        for current_object, current_position in objects:
            target = False
            if target_object:
                if target_object.position == current_position:
                    target = True
            self.place_object(current_object, current_position, target=target)
        if carrying:
            carrying_object = self.create_object(carrying,
                                                 self._object_vocabulary.get_object_vector(carrying.shape,
                                                                                           carrying.color,
                                                                                           carrying.size))
            self.carrying = carrying_object
            self.carrying.cur_pos = np.array([-1, -1])
            self.carrying.cur_pos = self.agent_pos

    def create_object(self, object_spec: Object, object_vector: np.ndarray, target=False):
        if object_spec.shape == "circle":
            return Circle(object_spec.color, size=object_spec.size, vector_representation=object_vector,
                          object_representation=object_spec, target=target,
                          weight=self._object_vocabulary.object_in_class(object_spec.size))
        elif object_spec.shape == "square":
            return Square(object_spec.color, size=object_spec.size, vector_representation=object_vector,
                          object_representation=object_spec, target=target,
                          weight=self._object_vocabulary.object_in_class(object_spec.size))
        elif object_spec.shape == "cylinder":
            return Cylinder(object_spec.color, size=object_spec.size, vector_representation=object_vector,
                            object_representation=object_spec,
                            weight=self._object_vocabulary.object_in_class(object_spec.size))
        else:
            raise ValueError("Trying to create an object shape {} that is not implemented.".format(object_spec.shape))

    def position_taken(self, position: Position):
        return self.grid.get(position.column, position.row) is not None

    def within_grid(self, position: Position):
        if 0 <= position.row < self.grid_size and 0 <= position.column < self.grid_size:
            return True
        else:
            return False

    def place_agent_at(self, position: Position):
        if not self.position_taken(position):
            self.place_agent(top=(position.column, position.row), size=(1, 1), rand_dir=False)
            self._occupied_positions.add((position.column, position.row))
        else:
            raise ValueError("Trying to place agent on cell that is already taken.")

    def sample_position(self) -> Position:
        available_positions = [(row, col) for row, col in itertools.product(list(range(self.grid_size)),
                                                                            list(range(self.grid_size)))
                               if (col, row) not in self._occupied_positions]
        sampled_position = random.sample(available_positions, 1).pop()
        return Position(row=sampled_position[0], column=sampled_position[1])

    def min_distance_from_edge(self, position: Position):
        row_distance = min(self.grid_size - position.row, position.row)
        column_distance = min(self.grid_size - position.column, position.column)
        return min(row_distance, column_distance)

    def sample_position_steps_from_edge(self, distance_from_edge=1) -> Position:
        available_positions = [(row, col) for row, col in itertools.product(list(range(self.grid_size)),
                                                                            list(range(self.grid_size)))
                               if (row, col) not in self._occupied_positions]
        actual_available_positions = []
        for row, col in available_positions:
            if self.min_distance_from_edge(Position(row=row, column=col)) <= distance_from_edge:
                actual_available_positions.append((row, col))
        sampled_position = random.sample(actual_available_positions, 1).pop()
        return Position(row=sampled_position[0], column=sampled_position[1])

    def sample_position_conditioned(self, north, east, south, west):
        """
        Specify for each direction how many steps should be free (i.e. before hitting wall) in that direction.
        """
        assert north == 0 or south == 0, "Can't take steps in both North and South direction"
        assert east == 0 or west == 0, "Can't take steps in both East and West direction"

        max_col = self.grid_size - east if east > 0 else self.grid_size - 1
        min_col = west - 1 if west > 0 else 0
        max_row = self.grid_size - south if south > 0 else self.grid_size - 1
        min_row = north - 1 if north > 0 else 0
        available_positions = []
        for col in range(min_col, max_col + 1):
            for row in range(min_row, max_row + 1):
                available_positions.append((row, col))
        sampled_position = random.sample(available_positions, 1).pop()
        return Position(row=sampled_position[0], column=sampled_position[1])

    def place_object(self, object_spec: Object, position: Position, target=False):
        if not self.within_grid(position):
            raise IndexError("Trying to place object '{}' outside of grid of size {}.".format(
                object_spec.shape, self.grid_size))
        # Object already placed at this location
        if self.position_taken(position):
            print("WARNING: attempt to place two objects at location ({}, {}), but overlapping objects not "
                  "supported. Skipping object.".format(position.row, position.column))
        else:
            object_vector = self._object_vocabulary.get_object_vector(shape=object_spec.shape, color=object_spec.color,
                                                                      size=object_spec.size)
            positioned_object = PositionedObject(object=object_spec, position=position, vector=object_vector)
            self.place_obj(self.create_object(object_spec, object_vector, target=target),
                           top=(position.column, position.row), size=(1, 1))

            # Add to list that keeps track of all objects currently positioned on the grid.
            self._placed_object_list.append(positioned_object)

            # Adjust the object lookup table accordingly.
            self._add_object_to_lookup_table(positioned_object)

            # Add to occupied positions:
            self._occupied_positions.add((position.column, position.row))

            if target:
                self._target_object = positioned_object

    def _add_object_to_lookup_table(self, positioned_object: PositionedObject):
        object_size = positioned_object.object.size
        object_color = positioned_object.object.color
        object_shape = positioned_object.object.shape

        # Generate all possible names
        object_names = generate_possible_object_names(color=object_color, shape=object_shape)
        for possible_object_name in object_names:
            if possible_object_name not in self._object_lookup_table.keys():
                self._object_lookup_table[possible_object_name] = {}

            # This part allows for multiple exactly the same objects (e.g. 2 small red circles) to be on the grid.
            if positioned_object.object.size not in self._object_lookup_table[possible_object_name].keys():
                self._object_lookup_table[possible_object_name] = {
                    size: [] for size in self._object_vocabulary.object_sizes}
            self._object_lookup_table[possible_object_name][object_size].append(
                positioned_object.position)

    def _remove_object(self, target_position: Position) -> PositionedObject:
        # remove from placed_object_list
        target_object = None
        for i, positioned_object in enumerate(self._placed_object_list):
            if positioned_object.position == target_position:
                target_object = self._placed_object_list[i]
                del self._placed_object_list[i]
                break

        # remove from object_lookup Table
        self._remove_object_from_lookup_table(target_object)

        # remove from gym grid
        self.grid.get(target_position.column, target_position.row)
        self.grid.set(target_position.column, target_position.row, None)

        self._occupied_positions.remove((target_position.column, target_position.row))

        return target_object

    def _remove_object_from_lookup_table(self, positioned_object: PositionedObject):
        possible_object_names = generate_possible_object_names(positioned_object.object.color,
                                                               positioned_object.object.shape)
        for possible_object_name in possible_object_names:
            self._object_lookup_table[possible_object_name][positioned_object.object.size].remove(
                positioned_object.position)

    def move_object(self, old_position: Position, new_position: Position):
        # Remove object from old position
        old_positioned_object = self._remove_object(old_position)
        if not old_positioned_object:
            raise ValueError("Trying to move an object from an empty grid location (row {}, col {})".format(
                old_position.row, old_position.column))

        # Add object at new position
        self.place_object(old_positioned_object.object, new_position)

    def pull(self, position: Position):
        self.agent_pos = (position.column, position.row)
        self._observed_commands.append("pull")
        self._observed_situations.append(self.get_current_situation())

    def pick_up_object(self):
        """
        Picking up an object in gym-minigrid means removing it and saying the agent is carrying it.
        :return:
        """
        assert self.grid.get(*self.agent_pos) is not None, "Trying to pick up an object at an empty cell."
        self.step(self.actions.pickup)
        if self.carrying:
            self._remove_object(Position(column=self.agent_pos[0], row=self.agent_pos[1]))
            self._observed_commands.append("PICK UP")
            self._observed_situations.append(self.get_current_situation())

    def drop_object(self):
        assert self.carrying is not None, "Trying to drop something but not carrying anything."
        self.place_object(self.carrying.object_representation, Position(column=self.agent_pos[0],
                                                                        row=self.agent_pos[1]))
        self.carrying = None
        self._observed_commands.append("DROP")
        self._observed_situations.append(self.get_current_situation())

    def push_or_pull_object(self, direction: Direction, primitive_command: str):
        current_object = self.grid.get(*self.agent_pos)
        if not current_object:
            self._observed_commands.append(primitive_command)
            self._observed_situations.append(self.get_current_situation())
        else:
            assert current_object.can_push(), "Trying to push an object that cannot be pushed"
            if current_object.push():
                new_position = self.agent_pos + DIR_TO_VEC[DIR_TO_INT[direction]]
                new_position = Position(column=new_position[0], row=new_position[1])
                # If the new position isn't occupied by another object, push it forward.
                if self.within_grid(new_position):
                    if not self.grid.get(new_position[0], new_position[1]):
                        self.move_object(Position(column=self.agent_pos[0], row=self.agent_pos[1]), new_position)
                        if primitive_command == "push":
                            self.take_step_in_direction(direction, primitive_command)
                        else:
                            self.pull(position=new_position)

            else:
                # Pushing an object that won't move just yet (because it's heavy).
                self._observed_commands.append(primitive_command)
                self._observed_situations.append(self.get_current_situation())

    def move_object_to_wall(self, action: str, manner: str):
        if action == "push":
            direction = INT_TO_DIR[self.agent_dir]
        else:
            direction = INT_TO_DIR[(self.agent_dir + 2) % 4]
        while self.empty_cell_in_direction(direction=direction):
            if manner == "while spinning":
                self.spin()
            elif manner == "cautiously":
                self.look_left_and_right()
            self.push_or_pull_object(direction=direction, primitive_command=action)
            if manner == "hesitantly":
                self.hesitate()

    @staticmethod
    def get_direction(direction_str: str):
        return DIR_STR_TO_DIR[direction_str]

    @staticmethod
    def get_position_at(current_position: Position, direction_str: str, distance: int) -> Position:
        """Returns the column and row of a position on the grid some distance away in a particular direction."""
        assert len(DIR_STR_TO_DIR[direction_str]) == 1, "getting a position at a distance only implemented for "\
                                                        "straight directions"
        direction = DIR_STR_TO_DIR[direction_str]
        direction_vec = DIR_TO_VEC[DIR_TO_INT[direction]] * distance
        position = np.array([current_position.column, current_position.row]) + direction_vec
        return Position(column=position[0], row=position[1])

    def direction_to_goal(self, goal: Position):
        difference_vec = np.array([goal.column - self.agent_pos[0], goal.row - self.agent_pos[1]])
        difference_vec[difference_vec < 0] = 0
        col_difference = difference_vec[0]
        row_difference = difference_vec[1]
        if col_difference and row_difference:
            return "SE", self.actions.left
        elif col_difference and not row_difference:
            return "NE", self.actions.right
        elif row_difference and not col_difference:
            return "SW", self.actions.right
        else:
            return "NW", self.actions.left

    def execute_command(self, command_str: str):
        command_list = command_str.split()
        verb = command_list[0]
        if len(command_list) > 1 and verb == "turn":
            direction = command_list[1]
            if direction == "left":
                self.take_step(self.actions.left, "turn left")
            elif direction == "right":
                self.take_step(self.actions.right, "turn right")
            else:
                raise ValueError("Trying to turn in an unknown direction")
        elif verb == "walk" or verb == "run" or verb == "jump":
            self.take_step_in_direction(direction=DIR_STR_TO_DIR[INT_TO_DIR[self.agent_dir].name[0]],
                                        primitive_command=verb)
        elif verb == "push" or verb == "pull":
            self.push_or_pull_object(direction=DIR_STR_TO_DIR[INT_TO_DIR[self.agent_dir].name[0]],
                                     primitive_command=verb)
        elif verb == "stay":
            return
        else:
            raise ValueError("Incorrect command {}.".format(command_str))

    def empty_cell_in_direction(self, direction: Direction):
        next_cell = self.agent_pos + DIR_TO_VEC[DIR_TO_INT[direction]]
        if self.within_grid(Position(column=next_cell[0], row=next_cell[1])):
            next_cell_object = self.grid.get(*next_cell)
            return not next_cell_object
        else:
            return False

    def look_left_and_right(self):
        self.take_step(self.actions.left, "turn left")
        self.take_step(self.actions.right, "turn right")
        self.take_step(self.actions.right, "turn right")
        self.take_step(self.actions.left, "turn left")

    def hesitate(self):
        self._observed_commands.append("stay")
        self._observed_situations.append(self.get_current_situation())

    def spin(self):
        for _ in range(4):
            self.take_step(self.actions.left, "turn left")

    def move_with_manners(self, direction: Direction, manner: str, primitive_command: str):
        # Spin to the left
        if manner == "while spinning":
            self.spin()
            self.take_step_in_direction(direction=direction, primitive_command=primitive_command)
        # Look left and right if cautious
        elif manner == "cautiously":
            self.turn_to_direction(direction=direction)
            self.look_left_and_right()
            self.take_step_in_direction(direction=direction, primitive_command=primitive_command)
        else:
            self.take_step_in_direction(direction=direction, primitive_command=primitive_command)

        # Stop after each step
        if manner == "hesitantly":
            self.hesitate()

    def go_to_position(self, position: Position, manner: str, primitive_command: str):
        """Move to the position denoted in the argument. Adds an action for each step to self._observed_commands
        and self._observed_situations. If a manner is specified, the sequence of actions will be transformed to
        represent the specified manner."""
        # Zigzag somewhere until in line with the goal, then just go straight for the goal
        if manner == "while zigzagging" and not self.agent_in_line_with_goal(position):
            # Find direction of goal.
            direction_to_goal, first_move = self.direction_to_goal(position)
            previous_step = first_move
            if direction_to_goal == "NE" or direction_to_goal == "SE":
                self.take_step_in_direction(direction=EAST, primitive_command=primitive_command)
            else:
                self.take_step_in_direction(direction=WEST, primitive_command=primitive_command)
            while not self.agent_in_line_with_goal(position):
                # turn in opposite direction of previous step and take take step
                if previous_step == self.actions.left:
                    self.take_step(self.actions.right, "turn right")
                    previous_step = self.actions.right
                else:
                    self.take_step(self.actions.left, "turn left")
                    previous_step = self.actions.left
                self.take_step(self.actions.forward, primitive_command)

            # Finish the route not zigzagging
            while self.agent_pos[0] > position.column:
                self.take_step_in_direction(direction=WEST, primitive_command=primitive_command)
            while self.agent_pos[0] < position.column:
                self.take_step_in_direction(direction=EAST, primitive_command=primitive_command)
            while self.agent_pos[1] > position.row:
                self.take_step_in_direction(direction=NORTH, primitive_command=primitive_command)
            while self.agent_pos[1] < position.row:
                self.take_step_in_direction(direction=SOUTH, primitive_command=primitive_command)
        else:
            # Calculate the route to the object on the grid.
            while self.agent_pos[0] > position.column:
                self.move_with_manners(direction=WEST, manner=manner, primitive_command=primitive_command)
            while self.agent_pos[0] < position.column:
                self.move_with_manners(direction=EAST, manner=manner, primitive_command=primitive_command)
            while self.agent_pos[1] > position.row:
                self.move_with_manners(direction=NORTH, manner=manner, primitive_command=primitive_command)
            while self.agent_pos[1] < position.row:
                self.move_with_manners(direction=SOUTH, manner=manner, primitive_command=primitive_command)

    def has_object(self, object_str: str) -> bool:
        if object_str not in self._object_lookup_table.keys():
            return False
        else:
            return True

    def object_positions(self, object_str: str, object_size=None) -> List[Position]:
        assert self.has_object(object_str), "Trying to get an object's position that is not placed in the world."
        object_locations = self._object_lookup_table[object_str]
        if object_size:
            present_object_sizes = [size for size, objects in object_locations.items() if objects]
            present_object_sizes.sort()
            assert len(present_object_sizes) >= 2, "referring to a {} object but only one of its size present.".format(
                object_size)
            # Perhaps just keep track of smallest and largest object in world
            if object_size == "small":
                object_locations = object_locations[present_object_sizes[0]]
            elif object_size == "big":
                object_locations = object_locations[present_object_sizes[-1]]
            else:
                raise ValueError("Wrong size in term specifications.")
        else:
            object_locations = object_locations.items()
        return object_locations

    def agent_in_line_with_goal(self, goal: Position):
        return goal.column == self.agent_pos[0] or goal.row == self.agent_pos[1]

    def take_step(self, action, observed_command: str):
        self.step(action=action)
        self._observed_situations.append(self.get_current_situation())
        self._observed_commands.append(observed_command)

    def turn_to_direction(self, direction: Direction) -> {}:
        """Turn to some direction."""
        current_direction = self.agent_dir
        target_direction = DIR_TO_INT[direction]
        if current_direction == target_direction:
            return
        assert current_direction != target_direction, "Trying to turn to a direction that is the current direction."
        difference_vector = DIR_TO_VEC[target_direction] - DIR_TO_VEC[self.agent_dir]
        difference_norm = np.linalg.norm(difference_vector, ord=2)
        if difference_norm >= 2:
            self.take_step(self.actions.left, "turn left")
            self.take_step(self.actions.left, "turn left")
        else:
            if current_direction == 0:  # East
                if target_direction == 1:
                    self.take_step(self.actions.right, "turn right")
                else:
                    self.take_step(self.actions.left, "turn left")
            elif current_direction == 3:  # North
                if target_direction == 0:
                    self.take_step(self.actions.right, "turn right")
                else:
                    self.take_step(self.actions.left, "turn left")
            else:  # South and West
                if target_direction > current_direction:
                    self.take_step(self.actions.right, "turn right")
                else:
                    self.take_step(self.actions.left, "turn left")

    def take_step_in_direction(self, direction: Direction, primitive_command: str):
        """
        Turn to some direction and take a step forward.
        """
        if DIR_TO_INT[direction] != self.agent_dir:
            self.turn_to_direction(direction)
        if self.within_grid(Position(column=self.front_pos[0], row=self.front_pos[1])):
            self.step(action=self.actions.forward)
            self._observed_commands.append(primitive_command)
            self._observed_situations.append(self.get_current_situation())

    def save_situation(self, file_name, attention_weights=[]) -> str:
        save_location = os.path.join(self.save_directory, file_name)
        assert save_location.endswith('.png'), "Invalid file name passed to save_situation, must end with .png."
        success = self.render(mode="human", attention_weights=attention_weights).save(save_location)
        if not success:
            print("WARNING: image with name {} failed to save.".format(file_name))
            return ''
        else:
            return save_location

    def get_current_situation_image(self) -> np.ndarray:
        return self.render().getArray()

    def get_current_situation_grid_repr(self) -> np.ndarray:
        return self.grid.encode(agent_row=self.agent_pos[1], agent_column=self.agent_pos[0],
                                agent_direction=self.agent_dir)

    def save_current_situation_image(self, image_name: str):
        save_path = os.path.join(self.save_directory, image_name)
        current_situation_array = self.get_current_situation_image()
        numpy_array_to_image(current_situation_array, save_path)

    def get_current_situation(self) -> Situation:
        if self.carrying:
            carrying = self.carrying.object_representation
        else:
            carrying = None
        return Situation(grid_size=self.grid_size,
                         agent_position=Position(column=self.agent_pos[0], row=self.agent_pos[1]),
                         target_object=self._target_object,
                         agent_direction=INT_TO_DIR[self.agent_dir], placed_objects=self._placed_object_list.copy(),
                         carrying=carrying)

    def get_current_observations(self):
        return self._observed_commands.copy(), self._observed_situations.copy()

    def clear_situation(self):
        self._object_lookup_table.clear()
        self._placed_object_list.clear()
        self._observed_commands.clear()
        self._observed_situations.clear()
        self._occupied_positions.clear()
        self.reset()

    def set_mission(self, mission: str):
        self.mission = mission

