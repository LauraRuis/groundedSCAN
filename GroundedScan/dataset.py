from GroundedScan.world import Situation, EVENT, Object, Position, World, ObjectVocabulary
from GroundedScan.grammar import Grammar, Derivation
from GroundedScan.vocabulary import Vocabulary
from GroundedScan.helpers import topo_sort, save_counter, bar_plot, grouped_bar_plot

import json
from collections import Counter, defaultdict
from typing import List, Tuple, Union, Dict
import numpy as np
import os
import imageio
import random
import itertools
import logging
from copy import deepcopy
import xlwt
from xlwt import Workbook

logger = logging.getLogger("GroundedScan")


class GroundedScan(object):
    """
    A dataset for systematic generalization in language, grounded in a gridworld.
    """

    def __init__(self, intransitive_verbs: Union[Dict[str, str], List[str], int],
                 transitive_verbs: Union[Dict[str, str], List[str], int],
                 adverbs: Union[Dict[str, str], List[str], int], nouns: Union[Dict[str, str], List[str], int],
                 color_adjectives: Union[Dict[str, str], List[str], int],
                 size_adjectives: Union[Dict[str, str], List[str], int],
                 grid_size: int, min_object_size: int,
                 max_object_size: int, type_grammar: str, sample_vocabulary: str, percentage_train: float,
                 percentage_dev=0.01,
                 save_directory=os.getcwd(), max_recursion=1):
        if sample_vocabulary == 'sample':
            needed_type = int
        elif sample_vocabulary == 'load':
            needed_type = dict
        elif sample_vocabulary == 'default':
            needed_type = list
        else:
            raise ValueError("Unknown value specified for sample_vocabulary: {}".format(sample_vocabulary))
        assert (isinstance(intransitive_verbs, needed_type) and isinstance(transitive_verbs, needed_type) and
                isinstance(adverbs, needed_type) and isinstance(nouns, needed_type) and
                isinstance(color_adjectives, needed_type) and isinstance(size_adjectives, needed_type)), \
            "please specify correct flags for words for --sample_vocabulary={}.".format(sample_vocabulary)

        # All images, data and data statistics will be saved in this directory.
        self.save_directory = save_directory

        # Command vocabulary.
        if sample_vocabulary == 'default':
            self._vocabulary = Vocabulary.initialize(intransitive_verbs=intransitive_verbs,
                                                     transitive_verbs=transitive_verbs, adverbs=adverbs, nouns=nouns,
                                                     color_adjectives=color_adjectives,
                                                     size_adjectives=size_adjectives)
        elif sample_vocabulary == 'sample':
            self._vocabulary = Vocabulary.sample(num_intransitive=intransitive_verbs, num_transitive=transitive_verbs,
                                                 num_adverbs=adverbs, num_color_adjectives=color_adjectives,
                                                 num_size_adjectives=size_adjectives)
        # We need to be able to load the saved vocabulary from file to map nonce words to their original meaning.
        elif sample_vocabulary == 'load':
            self._vocabulary = Vocabulary(intransitive_verbs=intransitive_verbs,
                                          transitive_verbs=transitive_verbs,
                                          adverbs=adverbs, nouns=nouns, color_adjectives=color_adjectives,
                                          size_adjectives=size_adjectives)
        else:
            raise ValueError("Unknown value for sample_vocabulary.")

        # Object vocabulary.
        self._object_vocabulary = ObjectVocabulary(shapes=self._vocabulary.get_semantic_shapes(),
                                                   colors=self._vocabulary.get_semantic_colors(),
                                                   min_size=min_object_size, max_size=max_object_size)

        # Initialize the world.
        self._world = World(grid_size=grid_size, colors=self._vocabulary.get_semantic_colors(),
                            object_vocabulary=self._object_vocabulary,
                            shapes=self._vocabulary.get_semantic_shapes(),
                            save_directory=self.save_directory)
        self._relative_directions = {"n", "e", "s", "w", "ne", "se", "sw", "nw"}
        self._straight_directions = {"n", "e", "s", "w"}
        self._combined_directions = {"ne", "se", "sw", "nw"}

        # Generate the grammar.
        self._type_grammar = type_grammar
        self.max_recursion = max_recursion
        self._grammar = Grammar(vocabulary=self._vocabulary, type_grammar=type_grammar, max_recursion=max_recursion)

        # Structures for data and statistics.
        self._percentage_train = percentage_train
        self._percentage_dev = percentage_dev
        self._possible_splits = ["train", "dev", "test", "visual", "situational_1", "situational_2", "contextual",
                                 "adverb_1", "adverb_2", "visual_easier", "target_lengths"]
        self._data_pairs = self.get_empty_split_dict()
        self._template_identifiers = self.get_empty_split_dict()
        self._examples_to_visualize = []
        self._k_shot_examples_in_train = Counter()
        self._data_statistics = {split: self.get_empty_data_statistics() for split in self._possible_splits}

    def reset_dataset(self):
        self._grammar.reset_grammar()
        self._data_pairs = self.get_empty_split_dict()
        self._template_identifiers = self.get_empty_split_dict()
        self._examples_to_visualize.clear()
        self._data_statistics = {split: self.get_empty_data_statistics() for split in self._possible_splits}

    def get_empty_split_dict(self):
        return {split: [] for split in self._possible_splits}

    def make_test_set(self, type_set: str, percentage: float):
        num_examples = int(percentage * len(self._data_pairs["train"]))
        k_random_indices = random.sample(range(0, len(self._data_pairs["train"])), k=num_examples)
        for example_idx in k_random_indices:
            example = deepcopy(self._data_pairs["train"][example_idx])
            template_identifier = self._template_identifiers["train"][example_idx]
            self._data_pairs[type_set].append(example)
            self._template_identifiers[type_set].append(template_identifier)
        for example_idx in sorted(k_random_indices, reverse=True):
            del self._data_pairs["train"][example_idx]
            del self._template_identifiers["train"][example_idx]

    def move_k_examples_to_train(self, k: int, split: str):
        if len(self._data_pairs[split]) < k + 1:
            logger.info("Not enough examples in split {} for k(k={})-shot generalization".format(split, k))
        k_random_indices = random.sample(range(0, len(self._data_pairs[split])), k=k)
        for example_idx in k_random_indices:
            example = deepcopy(self._data_pairs[split][example_idx])
            template_identifier = self._template_identifiers[split][example_idx]
            self._data_pairs["train"].append(example)
            self._template_identifiers["train"].append(template_identifier)
            self._k_shot_examples_in_train[split] += 1
        for example_idx in sorted(k_random_indices, reverse=True):
            del self._data_pairs[split][example_idx]
            del self._template_identifiers[split][example_idx]

    def get_examples_with_image(self, split="train", simple_situation_representation=False) -> dict:
        """
        Get data pairs with images in the form of np.ndarray's with RGB values or with 1 pixel per grid cell
        (see encode in class Grid of minigrid.py for details on what such representation looks like).
        :param split: string specifying which split to load.
        :param simple_situation_representation:  whether to get the full RGB image or a simple representation.
        :return: data examples.
        """
        for example in self._data_pairs[split]:
            command = self.parse_command_repr(example["command"])
            if example.get("meaning"):
                meaning = example["meaning"]
            else:
                meaning = example["command"]
            meaning = self.parse_command_repr(meaning)
            situation = Situation.from_representation(example["situation"])
            self._world.clear_situation()
            self.initialize_world(situation)
            if simple_situation_representation:
                situation_image = self._world.get_current_situation_grid_repr()
            else:
                situation_image = self._world.get_current_situation_image()
            target_commands = self.parse_command_repr(example["target_commands"])
            yield {"input_command": command, "input_meaning": meaning,
                   "derivation_representation": example.get("derivation"),
                   "situation_image": situation_image, "situation_representation": example["situation"],
                   "target_command": target_commands}

    @property
    def situation_image_dimension(self):
        return self._world.get_current_situation_image().shape[0]

    def num_examples(self, split="train"):
        return len(self._data_pairs[split])

    def count_equivalent_examples(self, split_1="train", split_2="test"):
        """Count the number of equivalent examples between two specified splits."""
        logger.info("WARNING: about to compare a maximum of {} examples.".format(
            len(self._data_pairs[split_1]) * len(self._data_pairs[split_2])))
        equivalent_examples = 0
        for i, example_1 in enumerate(self._data_pairs[split_1]):
            template_identifier_1 = self._template_identifiers[split_1][i]
            for j, example_2 in enumerate(self._data_pairs[split_2]):
                template_identifier_2 = self._template_identifiers[split_2][j]
                if template_identifier_2 == template_identifier_1:
                    if self.compare_examples(example_1, example_2):
                        equivalent_examples += 1
        return equivalent_examples

    def discard_equivalent_examples(self, split="test") -> int:
        """Go over the specified split and discard any examples that are already found in the training set."""
        equivalent_examples = 0
        to_delete = []
        for i, example in enumerate(self._data_pairs[split]):
            template_identifier = self._template_identifiers[split][i]
            if self.has_equivalent_example(example, template_identifier, split="train"):
                equivalent_examples += 1
                to_delete.append(i)
        for i_to_delete in sorted(to_delete, reverse=True):
            del self._data_pairs[split][i_to_delete]
            del self._template_identifiers[split][i_to_delete]
        return equivalent_examples

    def has_equivalent_example(self, example: dict, template_identifier: int, split="train"):
        """Go over the matching templates in the specified split and compare for equivalent with the passed example."""
        for i, example_1 in enumerate(self._data_pairs[split]):
            template_identifier_1 = self._template_identifiers[split][i]
            if template_identifier_1 == template_identifier:
                if self.compare_examples(example_1, example):
                    return True
        return False

    def meaning_command(self, input_command: List[str]):
        return [self._vocabulary.translate_word(word) for word in input_command]

    def fill_example(self, command: List[str], derivation: Derivation, situation: Situation, target_commands: List[str],
                     verb_in_command: str, target_predicate: dict, visualize: bool, adverb: str, splits: List[str]):
        """Add an example to the list of examples for the specified split."""
        example = {
            "command": self.command_repr(command),
            "meaning": self.command_repr(self.meaning_command(command)),
            "derivation": self.derivation_repr(derivation),
            "situation": situation.to_representation(),
            "target_commands": self.command_repr(target_commands),
            "verb_in_command": self._vocabulary.translate_word(verb_in_command),
            "manner": self._vocabulary.translate_word(adverb),
            "referred_target": ' '.join([self._vocabulary.translate_word(target_predicate["size"]),
                                         self._vocabulary.translate_word(target_predicate["color"]),
                                         self._vocabulary.translate_word(target_predicate["noun"])])
        }
        for split in splits:
            self._data_pairs[split].append(example)
        if visualize:
            self._examples_to_visualize.append(example)
        return example

    @staticmethod
    def compare_examples(example_1: dict, example_2: dict) -> bool:
        """An example is regarded the same if the command, situation, target commands are the same."""
        if example_1["command"] != example_2["command"]:
            return False
        if example_1["target_commands"] != example_2["target_commands"]:
            return False
        target_position_1 = example_1['situation']['target_object']['position']
        target_position_2 = example_2['situation']['target_object']['position']
        if (target_position_1['row'] != target_position_2['row'] or
                target_position_1['column'] != target_position_2['column']):
            return False
        return True

    def parse_example(self, data_example: dict):
        """Take an example as written in a file and parse it to its internal representations such that we can interact
        with it."""
        command = self.parse_command_repr(data_example["command"])
        meaning = self.parse_command_repr(data_example["meaning"])
        situation = Situation.from_representation(data_example["situation"])
        target_commands = self.parse_command_repr(data_example["target_commands"])
        derivation = self.parse_derivation_repr(data_example["derivation"])
        assert self.derivation_repr(derivation) == data_example["derivation"]
        actual_target_commands, target_demonstration, action = self.demonstrate_command(derivation, situation)
        assert self.command_repr(actual_target_commands) == self.command_repr(target_commands)
        return command, meaning, derivation, situation, actual_target_commands, target_demonstration, action

    @staticmethod
    def get_empty_situation():
        return {
            "distance_to_target": None,
            "direction_to_target": None,
            "target_shape": None,
            "target_color": None,
            "target_size": None,
            "target_position": None,
            "agent_position": None
        }

    def get_empty_data_statistics(self):
        empty_dict = {
            "distance_to_target": Counter(),
            "direction_to_target": Counter(),
            "input_length": Counter(),
            "target_length": Counter(),
            "target_shape": Counter(),
            "target_color": Counter(),
            "target_size": Counter(),
            "target_position": Counter(),
            "agent_position": Counter(),
            "verbs_in_command": defaultdict(int),
            "manners_in_command": defaultdict(int),
            "verb_adverb_combinations": defaultdict(lambda: defaultdict(int)),
            "verb_target_combinations": defaultdict(lambda: defaultdict(int)),
            "referred_targets": defaultdict(lambda: defaultdict(int)),
            "placed_targets": defaultdict(int),
            "situations": {
                "shape": {"objects_in_world": defaultdict(int), "num_objects_placed": Counter()},
                "color,shape": {"objects_in_world": defaultdict(int), "num_objects_placed": Counter()},
                "size,shape": {"objects_in_world": defaultdict(int), "num_objects_placed": Counter()},
                "size,color,shape": {"objects_in_world": defaultdict(int), "num_objects_placed": Counter()},
                "all": {"objects_in_world": defaultdict(int), "num_objects_placed": Counter()},
            },
            "examples_in_train": 0
        }
        for target_object in self._object_vocabulary.all_objects:
            target_object_str = ' '.join([str(target_object[0]), target_object[1], target_object[2]])
            for key in empty_dict["situations"].keys():
                empty_dict["situations"][key][target_object_str] = 0
            empty_dict["placed_targets"][target_object_str] = 0
        return empty_dict

    def update_data_statistics(self, data_example, split="train"):
        """Keeps track of certain statistics regarding the data pairs generated."""

        # Update the statistics regarding the situation.
        self._data_statistics[split]["distance_to_target"][int(data_example["situation"]["distance_to_target"])] += 1
        self._data_statistics[split]["direction_to_target"][data_example["situation"]["direction_to_target"]] += 1
        target_size = data_example["situation"]["target_object"]["object"]["size"]
        target_color = data_example["situation"]["target_object"]["object"]["color"]
        target_shape = data_example["situation"]["target_object"]["object"]["shape"]
        self._data_statistics[split]["target_shape"][target_shape] += 1
        self._data_statistics[split]["target_color"][target_color] += 1
        self._data_statistics[split]["target_size"][target_size] += 1
        self._data_statistics[split]["target_position"][
            (data_example["situation"]["target_object"]["position"]["column"],
             data_example["situation"]["target_object"]["position"]["row"])] += 1
        self._data_statistics[split]["agent_position"][(data_example["situation"]["agent_position"]["column"],
                                                        data_example["situation"]["agent_position"]["row"])] += 1
        placed_target = ' '.join([str(target_size), target_color, target_shape])
        self._data_statistics[split]["placed_targets"][placed_target] += 1

        # Update the statistics regarding the command.
        self._data_statistics[split]["verbs_in_command"][data_example["verb_in_command"]] += 1
        manner = data_example.get("manner")
        self._data_statistics[split]["manners_in_command"][manner] += 1
        self._data_statistics[split]["referred_targets"][data_example.get("referred_target")][placed_target] += 1
        self._data_statistics[split]["verb_adverb_combinations"][manner][data_example["verb_in_command"]] += 1
        self._data_statistics[split]["verb_target_combinations"][data_example["verb_in_command"]][placed_target] += 1
        self._data_statistics[split]["input_length"][len(data_example["command"].split(','))] += 1

        self._data_statistics[split]["target_length"][len(data_example["target_commands"].split(','))] += 1
        referred_target = data_example.get("referred_target")
        if referred_target:
            referred_target = referred_target.split()
        else:
            referred_target = [""]
        if len(referred_target) == 3:
            referred_categories = "size,color,shape"
        elif len(referred_target) == 1:
            referred_categories = "shape"
        else:
            if referred_target[0] in self._object_vocabulary.object_colors:
                referred_categories = "color,shape"
            else:
                referred_categories = "size,shape"
        num_placed_objects = len(data_example['situation']['placed_objects'].keys())
        self._data_statistics[split]["situations"][referred_categories]["num_objects_placed"][num_placed_objects] += 1
        self._data_statistics[split]["situations"]["all"]["num_objects_placed"][num_placed_objects] += 1
        for placed_object in data_example['situation']['placed_objects'].values():
            placed_object = ' '.join([placed_object['object']['size'], placed_object['object']['color'],
                                      placed_object['object']['shape']])
            self._data_statistics[split]["situations"][referred_categories]["objects_in_world"][placed_object] += 1
            self._data_statistics[split]["situations"]["all"]["objects_in_world"][placed_object] += 1

    def save_position_counts(self, position_counts, file) -> {}:
        """
        Prints a grid with at each position a count of something occurring at that position in the dataset
        (e.g. the agent or the target object.)
        """
        file.write("Columns\n")
        for row in range(self._world.grid_size):
            row_print = "Row {}".format(row)
            file.write(row_print)
            file.write((8 - len(row_print)) * ' ')
            for column in range(self._world.grid_size):
                if (str(column), str(row)) in position_counts:
                    count = position_counts[(str(column), str(row))]
                else:
                    count = 0
                count_print = "({}, {}): {}".format(column, row, count)
                fill_spaces = 20 - len(count_print)
                file.write(count_print + fill_spaces * ' ')
            file.write("\n")
            file.write("\n")

    def save_dataset_statistics(self, split="train") -> {}:
        """
        Summarizes the statistics and saves and prints them.
        """
        examples = self._data_pairs[split]
        for example in examples:
            self.update_data_statistics(example, split)
        with open(os.path.join(self.save_directory, split + "_dataset_stats.txt"), 'w') as infile:
            # General statistics
            number_of_examples = len(self._data_pairs[split])
            if number_of_examples == 0:
                logger.info("WARNING: trying to save dataset statistics for an empty split {}.".format(split))
                return
            infile.write("Number of examples: {}\n".format(number_of_examples))
            infile.write("Number of examples of this split in train: {}\n".format(
                str(self._k_shot_examples_in_train[split])))
            # Situation statistics.
            mean_distance_to_target = 0
            for distance_to_target, count in self._data_statistics[split]["distance_to_target"].items():
                mean_distance_to_target += count * distance_to_target
            mean_distance_to_target /= sum(self._data_statistics[split]["distance_to_target"].values())
            infile.write("Mean walking distance to target: {}\n".format(mean_distance_to_target))
            infile.write("Agent positions:\n")
            self.save_position_counts(self._data_statistics[split]["agent_position"], infile)
            infile.write("Target positions:\n")
            self.save_position_counts(self._data_statistics[split]["target_position"], infile)

            infile.write("Verbs:\n")
            verb_target_combinations = self._data_statistics[split]["verb_target_combinations"]
            infile.write("Verb target combinations:\n")
            for key, values in verb_target_combinations.items():
                save_counter(" " + key, values, infile)
            infile.write("\n")

            infile.write("Adverbs:\n")
            manners_in_command = self._data_statistics[split]["manners_in_command"]
            infile.write("Adverb occurrences:\n")
            save_counter("Adverbs", manners_in_command, infile)
            infile.write("\n")
            verb_adverb_combinations = self._data_statistics[split]["verb_adverb_combinations"]
            infile.write("Verb adverb combinations:\n")
            for key, values in verb_adverb_combinations.items():
                save_counter(" " + key, values, infile)
            infile.write("\n")
            referred_targets = self._data_statistics[split]["referred_targets"]
            infile.write("\nReferred Targets: \n")
            for key, values in referred_targets.items():
                save_counter("  " + key, values, infile)
            placed_targets = self._data_statistics[split]["placed_targets"]
            infile.write("\n")
            save_counter("placed_targets", placed_targets, infile)
            situation_stats = self._data_statistics[split]["situations"]
            infile.write("\nObjects placed in the world for particular referenced objects: \n")
            for key, values in situation_stats.items():
                save_counter("  " + key, values["num_objects_placed"], infile)
                save_counter("  " + key, values["objects_in_world"], infile)

        for key, values in self._data_statistics[split]["situations"].items():
            if len(values["objects_in_world"]):
                bar_plot(values["objects_in_world"], key, os.path.join(self.save_directory, split + "_" + key + ".png"))

        for key in self.get_empty_situation().keys():
            occurrence_counter = self._data_statistics[split][key]
            if key != "agent_position" and key != "target_position" and key != "distance_to_target":
                bar_plot(occurrence_counter, key, os.path.join(self.save_directory, split + "_" + key + ".png"))

        # Command statistics.
        verbs_in_command = self._data_statistics[split]["verbs_in_command"]
        bar_plot(verbs_in_command, "verbs_in_command", os.path.join(self.save_directory,
                                                                    split + "_" + "verbs_in_command.png"))
        manners_in_command = self._data_statistics[split]["manners_in_command"]
        bar_plot(manners_in_command, "manners_in_command", os.path.join(self.save_directory,
                                                                        split + "_" + "manners_in_command.png"))
        bar_plot(self._data_statistics[split]["target_length"], "target_lengths",
                 os.path.join(self.save_directory, split + "_" + "target_lengths.png"))
        bar_plot(self._data_statistics[split]["input_length"], "input_lengths",
                 os.path.join(self.save_directory, split + "_" + "input_lengths.png"))

    def save_dataset(self, file_name: str) -> str:
        """
        Saves the current generated data to a file in a particular format that is readable by load_examples_from_file.
        :param file_name: file name to save the dataset in. Will get saved in self.save_directory
        :return: path to saved file.
        """
        assert len(self._data_pairs) > 0, "No data to save, call .get_data_pairs()"
        output_path = os.path.join(self.save_directory, file_name)
        with open(output_path, 'w') as outfile:
            dataset_representation = {
                "grid_size": self._world.grid_size,
                "type_grammar": self._type_grammar,
                "grammar": self._grammar.__str__(),
                "min_object_size": self._object_vocabulary.smallest_size,
                "max_object_size": self._object_vocabulary.largest_size,
                "max_recursion": self.max_recursion,
                "percentage_train": self._percentage_train,
                "examples": {key: values for key, values in self._data_pairs.items()}
            }
            dataset_representation.update(self._vocabulary.to_representation())
            if self._type_grammar == "simple_intrans":
                dataset_representation["transitive_verbs"] = {}
            if self._type_grammar == "simple_trans":
                dataset_representation["intransitive_verbs"] = {}
            if not (self._type_grammar == "adverb" or self._type_grammar == "conjunction"):
                dataset_representation["adverbs"] = {}
            json.dump(dataset_representation, outfile, indent=4)
        return output_path

    @classmethod
    def load_dataset_from_file(cls, file_path: str, save_directory: str, k=0):
        with open(file_path, 'r') as infile:
            all_data = json.load(infile)
            percentage_train = all_data.get("percentage_train")
            if not percentage_train:
                percentage_train = 0.8
            dataset = cls(all_data["intransitive_verbs"], all_data["transitive_verbs"], all_data["adverbs"],
                          all_data["nouns"], all_data["color_adjectives"], all_data["size_adjectives"],
                          all_data["grid_size"], all_data["min_object_size"], all_data["max_object_size"],
                          type_grammar=all_data["type_grammar"], save_directory=save_directory,
                          percentage_train=percentage_train,
                          max_recursion=all_data["max_recursion"], sample_vocabulary='load')
            for split, examples in all_data["examples"].items():
                if split == "adverb_1":
                    num_examples = len(examples)
                    k_random_indices = random.sample(range(0, num_examples), k=k)
                else:
                    k_random_indices = []
                for i, example in enumerate(examples):
                    if i in k_random_indices:
                        dataset._data_pairs["train"].append(example)
                        dataset.update_data_statistics(example, "train")
                        dataset._data_pairs["dev"].append(example)
                        dataset.update_data_statistics(example, "dev")
                    else:
                        dataset._data_pairs[split].append(example)
                        dataset.update_data_statistics(example, split)
                    if split == "train" and dataset._vocabulary.translate_meaning("cautiously") in example["command"]:
                        dataset._data_pairs["dev"].append(example)
                        dataset.update_data_statistics(example, "dev")
            return dataset

    def generate_all_commands(self) -> {}:
        self._grammar.generate_all_commands()

    def sample_command(self) -> Tuple[Derivation, list]:
        coherent = False
        while not coherent:
            command = self._grammar.sample()
            arguments = []
            meaning = command.meaning(arguments)
            if not self._grammar.is_coherent(meaning):
                continue
            else:
                return command, arguments

    def demonstrate_target_commands(self, command: str, initial_situation: Situation,
                                    target_commands: List[str]) -> Tuple[List[str], List[Situation], int, int]:
        """Executes a sequence of commands starting from initial_situation."""
        current_situation = self._world.get_current_situation()
        current_mission = self._world.mission

        # Initialize the world based on the initial situation and the command.
        self.initialize_world(initial_situation, mission=command)

        for target_command in target_commands:
            self._world.execute_command(target_command)

        target_commands, target_demonstration = self._world.get_current_observations()
        end_column, end_row = self._world.agent_pos
        self._world.clear_situation()

        # Re-initialize the world as before the command
        self.initialize_world(current_situation, mission=current_mission)
        return target_commands, target_demonstration, end_column, end_row

    def demonstrate_command(self, derivation: Derivation, initial_situation: Situation) -> Tuple[List[str],
                                                                                                 List[Situation], str]:
        """
        Demonstrate a command derivation and situation pair. Done by extracting the events from the logical form
        of the command derivation, extracting the arguments of each event. The argument of the event gets located in the
        situation of the world and the path to that target gets calculated. Based on whether the verb in the command is
        transitive or not, the agent interacts with the object.
        :param derivation:
        :param initial_situation:
        :returns
        """
        command = ' '.join(derivation.words())
        arguments = []
        logical_form = derivation.meaning(arguments)
        current_situation = self._world.get_current_situation()
        current_mission = self._world.mission

        # Initialize the world based on the initial situation and the command.
        self.initialize_world(initial_situation, mission=command)

        # Extract all present events in the current command and order them by constraints.
        events = [variable for variable in logical_form.variables if variable.sem_type == EVENT]
        seq_constraints = [term.arguments for term in logical_form.terms if term.function == "seq"]
        ordered_events = topo_sort(events, seq_constraints)

        # Loop over the events to get the demonstrations.
        action = None
        for event in ordered_events:

            # Get the logical form of the current event
            sub_logical_form = logical_form.select([event], exclude={"seq"})
            event_lf = sub_logical_form.select([event], exclude={"patient"})
            args = [term.arguments[1] for term in sub_logical_form.terms if term.function == "patient"]

            # Find the action verb if it exists.
            is_transitive = False
            if event_lf.head.sem_type == EVENT:
                for term in event_lf.terms:
                    if term.specs.action:
                        action = term.specs.action
                        is_transitive = term.specs.is_transitive

            # Find the manner adverb if it exists.
            manner = [term.specs.manner for term in event_lf.terms if term.specs.manner]
            manner = manner.pop() if manner else None
            assert len(args) <= 1, "Only one target object supported, but two arguments parsed in a derivation."
            if len(args) > 0:
                # Find the logical form of the argument of the verb and find its location.
                arg_logical_form = sub_logical_form.select([args[0]])
                object_str, object_predicate = arg_logical_form.to_predicate()

                # If no location is passed, find the target object there.
                if not initial_situation.target_object:
                    translated_object_str = ' '.join([self._vocabulary.translate_word(word) for word in object_str.split()])
                    translated_object_size = self._vocabulary.translate_word(object_predicate["size"])
                    if self._world.has_object(translated_object_str):
                        object_locations = self._world.object_positions(translated_object_str,
                                                                        object_size=translated_object_size)
                    else:
                        object_locations = {}
                # Else we have saved the target location when we generated the situation.
                else:
                    object_locations = [initial_situation.target_object.position]

                if len(object_locations) > 1:
                    logger.info("WARNING: {} possible target locations.".format(len(object_locations)))
                if not object_locations:
                    continue
                goal = random.sample(object_locations, 1).pop()
                if not is_transitive:
                    primitive_command = self._vocabulary.translate_word(action)
                else:
                    primitive_command = "walk"

                self._world.go_to_position(position=goal, manner=manner, primitive_command=primitive_command)

                # Interact with the object for transitive verbs.
                if is_transitive:
                    semantic_action = self._vocabulary.translate_word(action)
                    self._world.move_object_to_wall(action=semantic_action, manner=manner)

        target_commands, target_demonstration = self._world.get_current_observations()
        self._world.clear_situation()

        # Re-initialize the world as before the command
        self.initialize_world(current_situation, mission=current_mission)
        return target_commands, target_demonstration, action

    def initialize_world(self, situation: Situation, mission="") -> {}:
        """
        Initializes the world with the passed situation.
        :param situation: class describing the current situation in the world, fully determined by a grid size,
        agent position, agent direction, list of placed objects, an optional target object and optional carrying object.
        :param mission: a string defining a command (e.g. "Walk to a green circle.")
        """
        objects = []
        for positioned_object in situation.placed_objects:
            objects.append((positioned_object.object, positioned_object.position))
        self._world.initialize(objects, agent_position=situation.agent_pos, agent_direction=situation.agent_direction,
                               target_object=situation.target_object, carrying=situation.carrying)
        if mission:
            self._world.set_mission(mission)

    def visualize_attention(self, input_commands: List[str], target_commands: List[str], situation: Situation,
                            attention_weights_commands: List[List[int]], attention_weights_situation: List[List[int]]):
        raise NotImplementedError()

    def error_analysis(self, predictions_file: str, output_file: str, save_directory: str):
        assert os.path.exists(predictions_file), "Trying to open a non-existing predictions file."
        error_analysis = {
            "target_length": defaultdict(lambda: {"accuracy": [], "exact_match": []}),
            "input_length": defaultdict(lambda: {"accuracy": [], "exact_match": []}),
            "verb_in_command": defaultdict(lambda: {"accuracy": [], "exact_match": []}),
            "manner": defaultdict(lambda: {"accuracy": [], "exact_match": []}),
            "referred_target": defaultdict(lambda: {"accuracy": [], "exact_match": []}),
            "referred_size": defaultdict(lambda: {"accuracy": [], "exact_match": []}),
            "distance_to_target": defaultdict(lambda: {"accuracy": [], "exact_match": []}),
            "direction_to_target": defaultdict(lambda: {"accuracy": [], "exact_match": []}),
            "actual_target": defaultdict(lambda: {"accuracy": [], "exact_match": []}),
        }
        all_accuracies = []
        exact_matches = []
        workbook = Workbook()
        with open(predictions_file, 'r') as infile:
            data = json.load(infile)
            logger.info("Running error analysis on {} examples.".format(len(data)))
            for predicted_example in data:

                # Get the scores of the current example.
                accuracy = predicted_example["accuracy"]
                exact_match = predicted_example["exact_match"]
                all_accuracies.append(accuracy)
                exact_matches.append(exact_match)

                # Get the information about the current example.
                example_information = {
                    "input_length": len(predicted_example["input"]),
                    "verb_in_command": self._vocabulary.translate_word(predicted_example["input"][0])}
                derivation = self.parse_derivation_repr(predicted_example["derivation"][0])
                arguments = []
                derivation.meaning(arguments)
                target_str, target_predicate = arguments.pop().to_predicate()
                adverb = ""
                for word in derivation.words():
                    if word in self._vocabulary.get_adverbs():
                        adverb = word
                manner = self._vocabulary.translate_word(adverb)
                example_information["referred_target"] = ' '.join([
                    self._vocabulary.translate_word(target_predicate["size"]),
                    self._vocabulary.translate_word(target_predicate["color"]),
                    self._vocabulary.translate_word(target_predicate["noun"])])
                if target_predicate["size"]:
                    example_information["referred_size"] = self._vocabulary.translate_word(target_predicate["size"])
                else:
                    example_information["referred_size"] = "None"
                example_information["target_length"] = len(predicted_example["target"])
                situation_repr = predicted_example["situation"]
                situation = Situation.from_representation(situation_repr[0])
                example_information["actual_target"] = ' '.join([str(situation.target_object.object.size),
                                                                 situation.target_object.object.color,
                                                                 situation.target_object.object.shape])
                example_information["direction_to_target"] = situation.direction_to_target
                example_information["distance_to_target"] = situation.distance_to_target
                example_information["manner"] = manner

                # Add that information to the analysis.
                for key in error_analysis.keys():
                    error_analysis[key][example_information[key]]["accuracy"].append(accuracy)
                    error_analysis[key][example_information[key]]["exact_match"].append(exact_match)

        # Write the information to a file and make plots
        with open(output_file, 'w') as outfile:
            outfile.write("Error Analysis\n\n")
            outfile.write(" Mean accuracy: {}\n".format(np.mean(np.array(all_accuracies))))
            exact_matches_counter = Counter(exact_matches)
            outfile.write(" Num. exact matches: {}\n".format(exact_matches_counter[True]))
            outfile.write(" Num not exact matches: {}\n\n".format(exact_matches_counter[False]))
            for key, values in error_analysis.items():
                sheet = workbook.add_sheet(key)
                sheet.write(0, 0, key)
                sheet.write(0, 1, "Num examples")
                sheet.write(0, 2, "Mean accuracy")
                sheet.write(0, 3, "Std. accuracy")
                sheet.write(0, 4, "Exact Match")
                sheet.write(0, 5, "Not Exact Match")
                sheet.write(0, 6, "Exact Match Percentage")
                outfile.write("\nDimension {}\n\n".format(key))
                means = {}
                standard_deviations = {}
                num_examples = {}
                exact_match_distributions = {}
                exact_match_relative_distributions = {}
                for i, (item_key, item_values) in enumerate(values.items()):
                    outfile.write("  {}:{}\n\n".format(key, item_key))
                    accuracies = np.array(item_values["accuracy"])
                    mean_accuracy = np.mean(accuracies)
                    means[item_key] = mean_accuracy
                    num_examples[item_key] = len(item_values["accuracy"])
                    standard_deviation = np.std(accuracies)
                    standard_deviations[item_key] = standard_deviation
                    exact_match_distribution = Counter(item_values["exact_match"])
                    exact_match_distributions[item_key] = exact_match_distribution
                    exact_match_relative_distributions[item_key] = exact_match_distribution[True] / (
                            exact_match_distribution[False] + exact_match_distribution[True])
                    outfile.write("    Num. examples: {}\n".format(len(item_values["accuracy"])))
                    outfile.write("    Mean accuracy: {}\n".format(mean_accuracy))
                    outfile.write("    Min. accuracy: {}\n".format(np.min(accuracies)))
                    outfile.write("    Max. accuracy: {}\n".format(np.max(accuracies)))
                    outfile.write("    Std. accuracy: {}\n".format(standard_deviation))
                    outfile.write("    Num. exact match: {}\n".format(exact_match_distribution[True]))
                    outfile.write("    Num. not exact match: {}\n\n".format(exact_match_distribution[False]))
                    sheet.write(i + 1, 0, item_key)
                    sheet.write(i + 1, 1, len(item_values["accuracy"]))
                    sheet.write(i + 1, 2, mean_accuracy)
                    sheet.write(i + 1, 3, standard_deviation)
                    sheet.write(i + 1, 4, exact_match_distribution[True])
                    sheet.write(i + 1, 5, exact_match_distribution[False])
                    sheet.write(i + 1, 6, exact_match_distribution[True] / (
                            exact_match_distribution[False] + exact_match_distribution[True]))
                outfile.write("\n\n\n")
                bar_plot(means, title=key, save_path=os.path.join(save_directory, key + '_accuracy'),
                         errors=standard_deviations, y_axis_label="accuracy")
                bar_plot(exact_match_relative_distributions, title=key, save_path=os.path.join(
                    save_directory, key + '_exact_match_rel'),
                         errors={}, y_axis_label="Exact Match Percentage")
                grouped_bar_plot(values=exact_match_distributions, group_one_key=True, group_two_key=False,
                                 title=key + ' Exact Matches', save_path=os.path.join(save_directory,
                                                                                      key + '_exact_match'),
                                 sort_on_key=True)
            outfile_excel = output_file.split(".txt")[0] + ".xls"
            workbook.save(outfile_excel)

    def visualize_prediction(self, predictions_file: str, only_save_errors=True, workbook=None) -> List[str]:
        """For each prediction in a file visualizes it in a gif and writes to self.save_directory."""
        assert os.path.exists(predictions_file), "Trying to open a non-existing predictions file."
        with open(predictions_file, 'r') as infile:
            data = json.load(infile)
            save_dirs = []
            sheet = workbook.add_sheet(predictions_file.split("/")[1].split(".")[0].split("_")[-1])
            sheet.write(0, 0, "Col Matches")
            sheet.write(0, 1, "Row Matches")
            sheet.write(0, 2, "Full Match")
            sheet.write(0, 3, "No Match")
            sheet.write(2, 0, "pred col")
            sheet.write(2, 1, "actual col")
            sheet.write(2, 2, "match")
            sheet.write(2, 3, "pred row")
            sheet.write(2, 4, "actual row")
            sheet.write(2, 5, "match")
            sheet.write(2, 6, "full match")
            sheet.write(2, 7, "no match")
            col_matches = 0
            row_matches = 0
            full_matches = 0
            no_matches = 0
            for i, predicted_example in enumerate(data):
                command = predicted_example["input"]
                prediction = predicted_example["prediction"]
                target = predicted_example["target"]
                meaning = [self._vocabulary.translate_word(word) for word in command]
                situation_repr = predicted_example["situation"]
                situation = Situation.from_representation(situation_repr[0])
                (predicted_commands, predicted_demonstration,
                 predicted_end_column, predicted_end_row) = self.demonstrate_target_commands(
                    command, situation, target_commands=prediction)
                (target_commands, target_demonstration,
                 actual_end_column, actual_end_row) = self.demonstrate_target_commands(
                    command, situation, target_commands=target)
                col_match = predicted_end_column == actual_end_column
                row_match = predicted_end_row == actual_end_row
                full_match = col_match and row_match
                no_match = not col_match and not row_match
                if i < 2000:
                    sheet.write(i + 3, 0, int(predicted_end_column))
                    sheet.write(i + 3, 1, int(actual_end_column))
                    sheet.write(i + 3, 2, int(col_match))
                    sheet.write(i + 3, 3, int(predicted_end_row))
                    sheet.write(i + 3, 4, int(actual_end_row))
                    sheet.write(i + 3, 5, int(row_match))
                    sheet.write(i + 3, 6, int(full_match))
                    sheet.write(i + 3, 7, int(no_match))
                if col_match:
                    col_matches += 1
                if row_match:
                    row_matches += 1
                if full_match:
                    full_matches += 1
                if no_match:
                    no_matches += 1

                # str_command = ' '.join(command)
                # mission = ' '.join(["Command:", str_command, "\nMeaning:"] + meaning
                #                    + ["\nPrediction"] + predicted_example["prediction"]
                #                    + ["\n      Target:"] + target_commands)
                # if predicted_example["exact_match"]:
                #     if only_save_errors:
                #         continue
                #     parent_save_dir = "exact_matches"
                # else:
                #     parent_save_dir = "errors"
                # save_dir_prediction = self.visualize_command(
                #     situation, command, predicted_demonstration, mission=mission, parent_save_dir=parent_save_dir,
                #     attention_weights=predicted_example["attention_weights_situation"])
                # save_dirs.append(save_dir_prediction)
            sheet.write(1, 0, col_matches)
            sheet.write(1, 1, row_matches)
            sheet.write(1, 2, full_matches)
            sheet.write(1, 3, no_matches)
        return save_dirs

    def visualize_data_example(self, data_example: dict) -> str:
        command, meaning, derivation, situation, actual_target_commands, target_demonstration, _ = self.parse_example(
            data_example)
        mission = ' '.join(["Command:", ' '.join(command), "\nMeaning: ", ' '.join(meaning),
                            "\nTarget:"] + actual_target_commands)
        save_dir = self.visualize_command(situation, command, target_demonstration,
                                          mission=mission)
        return save_dir

    def visualize_data_examples(self) -> List[str]:
        if len(self._examples_to_visualize) == 0:
            logger.info("No examples to visualize.")
        save_dirs = []
        for data_example in self._examples_to_visualize:
            save_dir = self.visualize_data_example(data_example)
            save_dirs.append(save_dir)
        return save_dirs

    def visualize_command(self, initial_situation: Situation, command: List[str], demonstration: List[Situation],
                          mission: str, parent_save_dir="", attention_weights=[]) -> str:
        """
        :param initial_situation: (list of objects with their location, grid size, agent position)
        :param command: command in natural language
        :param demonstration: action sequence
        :param mission: the string you want to display in the text box of the image
        :param parent_save_dir: directory to save in, can be the instruction text separated by _'s
        :param attention_weights: if filled, colors grid cells with higher values darker grey
        :return: path_to_visualization
        """
        # Save current situation.
        current_situation = self._world.get_current_situation()
        current_mission = self._world.mission

        # Initialize directory with current command as its name.
        mission_folder = '_'.join([self._vocabulary.translate_word(word) for word in command])
        if parent_save_dir:
            mission_folder = os.path.join(parent_save_dir, mission_folder)
            if not os.path.exists(os.path.join(self.save_directory, parent_save_dir)):
                os.mkdir(os.path.join(self.save_directory, parent_save_dir))
        full_dir = os.path.join(self.save_directory, mission_folder)
        if not os.path.exists(full_dir):
            os.mkdir(full_dir)
            file_count = 0
        else:
            files_list = os.listdir(full_dir)
            file_count = len(files_list)
        mission_folder = os.path.join(mission_folder, "situation_{}".format(file_count))
        final_dir = os.path.join(full_dir, "situation_{}".format(file_count))
        if not os.path.exists(final_dir):
            os.mkdir(final_dir)

        # Visualize command.
        self.initialize_world(initial_situation, mission=mission)
        if attention_weights:
            current_attention_weights = np.array(attention_weights[0])
        else:
            current_attention_weights = []
        save_location = self._world.save_situation(os.path.join(mission_folder, 'initial.png'),
                                                   attention_weights=current_attention_weights)
        filenames = [save_location]

        for i, situation in enumerate(demonstration):
            if attention_weights:
                assert len(attention_weights) >= len(demonstration), "Unequal number of attention weights and "\
                                                                     "demonstration steps."
                current_attention_weights = np.array(attention_weights[i])
            else:
                current_attention_weights = []
            self.initialize_world(situation, mission=mission)
            save_location = self._world.save_situation(os.path.join(mission_folder, 'situation_' + str(i) + '.png'),
                                                       attention_weights=current_attention_weights)
            filenames.append(save_location)

        # Make a gif of the action sequence.
        images = []
        for filename in filenames:
            images.append(imageio.imread(filename))
        movie_dir = os.path.join(self.save_directory, mission_folder)
        imageio.mimsave(os.path.join(movie_dir, 'movie.gif'), images, fps=5)

        # Restore situation.
        self.initialize_world(current_situation, mission=current_mission)

        return movie_dir

    def generate_possible_targets(self, referred_size: str, referred_color: str, referred_shape: str):
        """
        Generate a list of possible target objects based on some target referred to in a command, e.g.
        for small red circle any sized circle but the largest can be a potential target.
        """
        if referred_size:
            if referred_size == "small":
                target_sizes = self._object_vocabulary.object_sizes[:-1]
            elif referred_size == "big":
                target_sizes = self._object_vocabulary.object_sizes[1:]
            else:
                raise ValueError("Unknown size adjective in command.")
        else:
            target_sizes = self._object_vocabulary.object_sizes
        # If no color specified, use all colors.
        if not referred_color:
            target_colors = self._object_vocabulary.object_colors
        else:
            target_colors = [referred_color]

        # Return all possible combinations of sizes and colors
        return list(itertools.product(target_sizes, target_colors, [referred_shape]))

    def all_objects_except_shape(self, shape: str) -> List[tuple]:
        all_sizes = self._object_vocabulary.object_sizes
        all_colors = self._object_vocabulary.object_colors
        all_shapes = self._object_vocabulary.object_shapes
        all_shapes.remove(shape)
        return list(itertools.product(all_sizes, all_colors, all_shapes))

    def get_larger_sizes(self, size: int) -> List[int]:
        return list(range(size + 1, self._object_vocabulary.largest_size + 1))

    def get_smaller_sizes(self, size: int) -> List[int]:
        return list(range(self._object_vocabulary.smallest_size, size))

    def generate_distinct_objects(self, referred_size: str, referred_color: str, referred_shape: str,
                                  actual_size: int, actual_color: str) -> Tuple[list, list]:
        """
        Generate a list of objects that are distinct from some referred target. E.g. if the referred target is a
        small circle, and the actual color of the target object is red, there cannot also be a blue circle of the same
        size, since then there will be 2 possible targets.
        Currently makes sure at least 2 sized objects of each group is placed whenever a size is referred to in the
        referred_size. E.g. if the command is 'walk to a big circle', make sure there are at least 2 sized circles.
        This doesn't get done for the color, e.g. if the comment is 'walk to a green circle', there are not
        necessarily also other colored circles in obligatory_objects.
        """
        objects = []
        # Initialize list that will be filled with objects that need to be present in the situation for it to make sense
        # E.g. if the referred object is 'small circle' there needs to be at least 1 larger circle.
        obligatory_objects = []
        # E.g. distinct from 'circle' -> no other circles; generate one random object of each other shape.
        if not referred_size and not referred_color:
            all_shapes = self._object_vocabulary.object_shapes
            all_shapes.remove(referred_shape)
            for shape in all_shapes:
                objects.append([(self._object_vocabulary.sample_size(), self._object_vocabulary.sample_color(), shape)])
            return objects, obligatory_objects
        # E.g. distinct from 'red circle' -> no other red circles of any size; generate one randomly size object for
        # each color, shape combination that is not a 'red circle'.
        elif not referred_size:
            for shape in self._object_vocabulary.object_shapes:
                for color in self._object_vocabulary.object_colors:
                    if not (shape == referred_shape and color == referred_color):
                        objects.append([(self._object_vocabulary.sample_size(), color, shape)])
            return objects, obligatory_objects
        else:
            if referred_size == "small":
                all_other_sizes = self.get_larger_sizes(actual_size)
            elif referred_size == "big":
                all_other_sizes = self.get_smaller_sizes(actual_size)
            else:
                raise ValueError("Unknown referred size in command")
            all_other_shapes = self._object_vocabulary.object_shapes
            all_other_shapes.remove(referred_shape)
            # E.g. distinct from 'small circle' -> no circles of size <= than target in any color; generate two
            # random sizes for each color-shape pair except for the shape that is referred generate one larger objects
            # (if referred size is small, else a smaller object)
            if not referred_color:
                for shape in self._object_vocabulary.object_shapes:
                    for color in self._object_vocabulary.object_colors:
                        if not shape == referred_shape:
                            colored_shapes = []
                            for _ in range(2):
                                colored_shapes.append((self._object_vocabulary.sample_size(), color, shape))
                            objects.append(colored_shapes)
                        else:
                            if not color == actual_color:
                                colored_shapes = []
                                for _ in range(2):
                                    colored_shapes.append((random.choice(all_other_sizes), color, shape))
                                objects.append(colored_shapes)
                            else:
                                obligatory_objects.append((random.choice(all_other_sizes), color, shape))
                return objects, obligatory_objects
            # E.g. distinct from 'small red circle' -> no red circles of size <= as target; generate for each
            # color-shape pair two random sizes, and when the pair is the referred pair, one larger size.
            else:
                for shape in self._object_vocabulary.object_shapes:
                    for color in self._object_vocabulary.object_colors:
                        if not (shape == referred_shape and color == referred_color):
                            colored_shapes = []
                            for _ in range(2):
                                colored_shapes.append((self._object_vocabulary.sample_size(), color, shape))
                            objects.append(colored_shapes)
                        else:
                            obligatory_objects.append((random.choice(all_other_sizes), color, shape))
                return objects, obligatory_objects

    def generate_situations(self, num_resampling=1):
        """
        Generate all semantically distinct situations with an agent and a target object.
        A semantically distinct situation is based on the target object (shape, color and size), the direction the
        agent has w.r.t. the target (e.g. North, South-West, etc.) and the number of steps the agent is removed from
        the target. For each of these possible situations, num_resampling defines how often other objects positions
        are resampled to create a new situation.
        :param num_resampling: how often to resample a semantically equivalent situation but with non-target objects at
        different locations.
        :return: a dictionary with situations.
        """
        # All possible target objects
        all_targets = itertools.product(self._object_vocabulary.object_sizes, self._object_vocabulary.object_colors,
                                        self._object_vocabulary.object_shapes)

        # Loop over all semantically different situation specifications
        situation_specifications = {}
        for target_size, target_color, target_shape in all_targets:
            if target_shape not in situation_specifications.keys():
                situation_specifications[target_shape] = {}
            if target_color not in situation_specifications[target_shape].keys():
                situation_specifications[target_shape][target_color] = {}
            if target_size not in situation_specifications[target_shape][target_color].keys():
                situation_specifications[target_shape][target_color][target_size] = []

            # Loop over all possible directions from agent to target (e.g. agent is facing the target from the NW).
            for direction_str in self._relative_directions:

                # For straight directions (e.g. North, East, South and West) loop over 1 to grid size number of steps.
                if direction_str in self._straight_directions:
                    for num_steps_to_target in range(1, self._world.grid_size):

                        # Don't resample too often on the edges of the grid-world.
                        if 1 < num_steps_to_target < self._world.grid_size - 1:
                            num_to_resample = num_resampling
                        else:
                            num_to_resample = 1

                        # Resample a semantically equivalent situation based on positions of non-target objects.
                        for _ in range(num_to_resample):
                            empty_situation = self.get_empty_situation()
                            target_position = Position(column=self._world.grid_size + 1, row=self._world.grid_size + 1)
                            while not self._world.within_grid(target_position):
                                condition = {"n": 0, "e": 0, "s": 0, "w": 0}
                                condition[direction_str] = num_steps_to_target
                                agent_position = self._world.sample_position_conditioned(*condition.values())
                                target_position = self._world.get_position_at(agent_position, direction_str,
                                                                              num_steps_to_target)
                            assert self._world.within_grid(target_position) and self._world.within_grid(agent_position)

                            # Save a situation.
                            empty_situation["agent_position"] = agent_position
                            empty_situation["target_position"] = target_position
                            empty_situation["distance_to_target"] = num_steps_to_target
                            empty_situation["direction_to_target"] = direction_str
                            empty_situation["target_shape"] = target_shape
                            empty_situation["target_color"] = target_color
                            empty_situation["target_size"] = target_size
                            situation_specifications[target_shape][target_color][target_size].append(empty_situation)

                # For combined dirs (e.g. North-East, South-West, etc.) loop over 1 to 2 * grid size number of steps
                elif direction_str in self._combined_directions:
                    for number_of_steps_in_direction in range(2, 2 * (self._world.grid_size - 1) + 1):
                        if 1 < number_of_steps_in_direction < 2 * (self._world.grid_size - 1):
                            num_to_resample = num_resampling
                        else:
                            num_to_resample = 1
                        for _ in range(num_to_resample):
                            empty_situation = self.get_empty_situation()

                            # Randomly divide the number of steps over each direction of the combination
                            random_divide = random.randint(max(1, number_of_steps_in_direction - self._world.grid_size + 1),
                                                           min(number_of_steps_in_direction - 1, self._world.grid_size - 1))
                            steps_in_first_direction = random_divide
                            steps_in_second_direction = number_of_steps_in_direction - random_divide
                            assert (steps_in_second_direction + steps_in_first_direction) == number_of_steps_in_direction
                            assert (steps_in_first_direction and steps_in_second_direction) <= self._world.grid_size - 1
                            directions = list(direction_str)
                            target_position = Position(column=self._world.grid_size + 1, row=self._world.grid_size + 1)
                            while not self._world.within_grid(target_position):
                                condition = {"n": 0, "e": 0, "s": 0, "w": 0}
                                condition[directions[0]] = steps_in_first_direction
                                condition[directions[1]] = steps_in_second_direction
                                agent_position = self._world.sample_position_conditioned(*condition.values())
                                intermediate_target_position = self._world.get_position_at(agent_position,
                                                                                           directions[0],
                                                                                           steps_in_first_direction)
                                target_position = self._world.get_position_at(intermediate_target_position,
                                                                              directions[1], steps_in_second_direction)
                            assert self._world.within_grid(target_position) and self._world.within_grid(agent_position)
                            empty_situation["agent_position"] = agent_position
                            empty_situation["target_position"] = target_position
                            empty_situation["distance_to_target"] = number_of_steps_in_direction
                            empty_situation["direction_to_target"] = direction_str
                            empty_situation["target_shape"] = target_shape
                            empty_situation["target_color"] = target_color
                            empty_situation["target_size"] = target_size
                            situation_specifications[target_shape][target_color][target_size].append(empty_situation)
        return situation_specifications

    def initialize_world_from_spec(self, situation_spec, referred_size: str, referred_color: str, referred_shape: str,
                                   actual_size: int, sample_percentage=0.5, min_other_objects=0):
        self._world.clear_situation()
        self._world.place_agent_at(situation_spec["agent_position"])
        target_shape = situation_spec["target_shape"]
        target_color = situation_spec["target_color"]
        target_size = situation_spec["target_size"]
        self._world.place_object(Object(size=target_size, color=target_color, shape=target_shape),
                                 position=situation_spec["target_position"], target=True)
        distinct_objects, obligatory_objects = self.generate_distinct_objects(
            referred_size=self._vocabulary.translate_word(referred_size),
            referred_color=self._vocabulary.translate_word(referred_color),
            referred_shape=self._vocabulary.translate_word(referred_shape),
            actual_size=actual_size,
            actual_color=target_color)
        num_to_sample = int(len(distinct_objects) * sample_percentage)
        num_to_sample = max(min_other_objects, num_to_sample)
        objects_to_place = obligatory_objects
        sampled_objects = random.sample(distinct_objects, k=num_to_sample)
        for sampled_objs in sampled_objects:
            objects_to_place.extend(sampled_objs)
        for size, color, shape in objects_to_place:
            other_position = self._world.sample_position()
            self._world.place_object(Object(size=size, color=color, shape=shape), position=other_position)

    @staticmethod
    def command_repr(command: List[str]) -> str:
        return ','.join(command)

    @staticmethod
    def parse_command_repr(command_repr: str) -> List[str]:
        return command_repr.split(',')

    @staticmethod
    def derivation_repr(derivation: Derivation) -> str:
        return str(derivation)

    def parse_derivation_repr(self, derivation_repr: str) -> Derivation:
        command_rules, command_lexicon = derivation_repr.split(';')
        return Derivation.from_str(command_rules, command_lexicon, self._grammar)

    @staticmethod
    def position_repr(position: Position):
        return ','.join([str(position.column), str(position.row)])

    @staticmethod
    def parse_position_repr(position_repr: str) -> Position:
        column, row = position_repr.split(',')
        return Position(column=int(column), row=int(row))

    def get_data_pairs(self, max_examples=None, num_resampling=1, other_objects_sample_percentage=0.5,
                       split_type="uniform", visualize_per_template=0, visualize_per_split=0, train_percentage=0.8,
                       min_other_objects=0, k_shot_generalization=0, make_dev_set=False,
                       cut_off_target_length=25) -> {}:
        """
        Generate a set of situations and generate all possible commands based on the current grammar and lexicon,
        match commands to situations based on relevance (if a command refers to a target object, it needs to be
        present in the situation) and save these pairs in a the list of data examples.
        """
        if k_shot_generalization > 0 and split_type == "uniform":
            logger.info("WARNING: k_shot_generalization set to {} but for split_type uniform this is not used.".format(
                k_shot_generalization))

        # Save current situation of the world for later restoration.
        current_situation = self._world.get_current_situation()
        current_mission = self._world.mission
        self.reset_dataset()

        # Generate all situations and commands.
        situation_specifications = self.generate_situations(num_resampling=num_resampling)
        self.generate_all_commands()
        example_count = 0
        dropped_examples = 0
        for template_num, template_derivations in self._grammar.all_derivations.items():
            visualized_per_template = 0
            visualized_per_split = {split: 0 for split in self._possible_splits}
            for derivation in template_derivations:
                arguments = []
                derivation.meaning(arguments)
                assert len(arguments) == 1, "Only one target object currently supported."
                adverb = ""
                for word in derivation.words():
                    if word in self._vocabulary.get_adverbs():
                        adverb = word
                target_str, target_predicate = arguments.pop().to_predicate()
                possible_target_objects = self.generate_possible_targets(
                    referred_size=self._vocabulary.translate_word(target_predicate["size"]),
                    referred_color=self._vocabulary.translate_word(target_predicate["color"]),
                    referred_shape=self._vocabulary.translate_word(target_predicate["noun"]))
                for target_size, target_color, target_shape in possible_target_objects:
                    relevant_situations = situation_specifications[target_shape][target_color][target_size]
                    num_relevant_situations = len(relevant_situations)
                    idx_to_visualize = random.sample([i for i in range(num_relevant_situations)], k=1).pop()

                    if split_type == "uniform":
                        idx_for_train = random.sample([i for i in range(num_relevant_situations)], k=int(
                            num_relevant_situations * train_percentage))
                        idx_for_train = set(idx_for_train)
                    for i, relevant_situation in enumerate(relevant_situations):
                        visualize = False
                        if (example_count + 1) % 10000 == 0:
                            logger.info("Number of examples: {}".format(example_count + 1))
                        if max_examples:
                            if example_count >= max_examples:
                                break
                        self.initialize_world_from_spec(relevant_situation,
                                                        referred_size=target_predicate["size"],
                                                        referred_color=target_predicate["color"],
                                                        referred_shape=target_predicate["noun"],
                                                        actual_size=target_size,
                                                        sample_percentage=other_objects_sample_percentage,
                                                        min_other_objects=min_other_objects
                                                        )
                        situation = self._world.get_current_situation()
                        assert situation.direction_to_target == relevant_situation["direction_to_target"]
                        if situation.direction_to_target not in self._combined_directions:
                            continue
                        assert situation.distance_to_target == relevant_situation["distance_to_target"]
                        if situation.distance_to_target < 2:
                            continue
                        target_commands, target_situations, target_action = self.demonstrate_command(
                            derivation, initial_situation=situation)
                        if i == idx_to_visualize:
                            visualize = True
                        if visualized_per_template >= visualize_per_template:
                            visualize = False
                        if adverb and visualized_per_template <= visualize_per_template:
                            visualize = True
                        if split_type == "uniform":
                            if i in idx_for_train:
                                splits = ["train"]
                                if adverb != "hesitantly":
                                    continue
                            else:
                                splits = ["test"]
                        elif split_type == "generalization":
                            splits = self.assign_splits(target_size, target_color, target_shape, target_action,
                                                        situation.direction_to_target, target_predicate,
                                                        self._vocabulary.translate_word(adverb))
                            if len(splits) == 0:
                                splits = ["train"]
                            elif len(splits) > 1:
                                dropped_examples += 1
                                self._world.clear_situation()
                                continue
                            else:
                                if split_type == "generalization":
                                    if visualized_per_split[splits[0]] <= visualize_per_split:
                                        visualized_per_split[splits[0]] += 1
                                        visualize = True
                                if adverb != "hesitantly":
                                    continue
                        elif split_type == "target_lengths":
                            if len(target_commands) > cut_off_target_length:
                                splits = ["test"]
                            else:
                                splits = ["train"]
                        else:
                            raise ValueError("Unknown split_type in .get_data_pairs().")
                        self.fill_example(command=derivation.words(), derivation=derivation,
                                          situation=situation, target_commands=target_commands,
                                          verb_in_command=target_action, target_predicate=target_predicate,
                                          visualize=visualize, adverb=adverb, splits=splits)
                        for split in splits:
                            self._template_identifiers[split].append(template_num)
                        example_count += 1
                        if visualize:
                            visualized_per_template += 1
                        self._world.clear_situation()
        logger.info("Dropped {} examples due to belonging to multiple splits.".format(dropped_examples))
        if split_type == "generalization":
            self.make_test_set(percentage=(1 - self._percentage_train), type_set="test")
        logger.info("Discarding equivalent examples, may take a while...")
        equivalent_examples = self.discard_equivalent_examples()
        logger.info("Discarded {} examples from the test set that were already in the training set.".format(
            equivalent_examples))

        if make_dev_set:
            self.make_test_set(percentage=self._percentage_dev, type_set="dev")

        if k_shot_generalization > 0:
            self.move_k_examples_to_train(k_shot_generalization, split="adverb_1")

        # restore situation
        self.initialize_world(current_situation, mission=current_mission)
        return

    def assign_splits(self, target_size: str, target_color: str, target_shape: str, verb_in_command: str,
                      direction_to_target: str, referred_target: dict, manner: str):
        splits = []
        # Experiment 1: visual generalization, hold out all red squares as targets.
        if target_color == "red" and target_shape == "square":
            splits.append("visual")
        # Experiment 2: situational generalization, hold out all directions of agent to target = South-West.
        if direction_to_target == "sw":
            splits.append("situational_1")
        # Experiment 3: situational generalization, hold out all situations where a circle of size 2 is referred to
        # as the small circle.
        if self._vocabulary.translate_word(referred_target["size"]) == "small" and \
                target_shape == "circle" and target_size == 2:
            splits.append("situational_2")
        # Experiment 4: contextual generalization, hold out all situations where interaction with a red square of
        # size 3 is required.
        if self._vocabulary.translate_word(verb_in_command) == "push" and target_shape == "square" and target_size == 3:
            splits.append("contextual")
        # Experiment 5: generalize adverb to new situations.
        if manner == "cautiously":
            splits.append("adverb_1")
        # Experiment 6: generalize adverb to new verb 'pull'.
        if (verb_in_command == self._vocabulary.translate_meaning("pull")
                and manner == "while spinning"):
            splits.append("adverb_2")
        # Experiment 7: visual gen. easier, hold out all yellow squares as targets if also yellow in referred target.
        if (self._vocabulary.translate_meaning("yellow") == referred_target["color"] and
                target_color == "yellow" and target_shape == "square"):
            splits.append("visual_easier")
        return splits
