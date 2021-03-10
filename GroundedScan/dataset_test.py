# TODO: use test framework instead of asserts
from GroundedScan.dataset import GroundedScan
from GroundedScan.grammar import Derivation
from GroundedScan.world import Situation
from GroundedScan.world import Position
from GroundedScan.world import Object
from GroundedScan.world import INT_TO_DIR
from GroundedScan.world import PositionedObject
from GroundedScan.helpers import numpy_array_to_image
from GroundedScan.helpers import image_to_numpy_array
import GroundedScan.dsl as dsl

import os
import time
import numpy as np
import logging
import random

logging.getLogger("PyQt5").disabled = True
logging.getLogger('matplotlib.font_manager').disabled = True
logger = logging.getLogger("GroundedScan")

TEST_DIRECTORY = "test_dir"
TEST_PATH = os.path.join(os.getcwd(), TEST_DIRECTORY)
if not os.path.exists(TEST_PATH):
    os.mkdir(TEST_PATH)

EXAMPLES_TO_TEST = 10000

intransitive_verbs = ["walk"]
transitive_verbs = ["push", "pull"]
adverbs = ["cautiously", "while spinning", "hesitantly", "while zigzagging"]
nouns = ["circle", "cylinder", "square"]
color_adjectives = ["red", "blue", "green", "yellow"]
size_adjectives = ["big", "small"]

TEST_DATASET = GroundedScan(intransitive_verbs=intransitive_verbs,
                            transitive_verbs=transitive_verbs,
                            adverbs=adverbs, nouns=nouns,
                            color_adjectives=color_adjectives,
                            size_adjectives=size_adjectives, percentage_train=0.8,
                            min_object_size=1, max_object_size=4, sample_vocabulary='default',
                            save_directory=TEST_DIRECTORY, grid_size=15, type_grammar="adverb")

TEST_DATASET_2 = GroundedScan(intransitive_verbs=intransitive_verbs,
                              transitive_verbs=transitive_verbs,
                              adverbs=adverbs, nouns=nouns,
                              color_adjectives=color_adjectives,
                              size_adjectives=size_adjectives, percentage_train=0.8,
                              min_object_size=1, max_object_size=4, sample_vocabulary='default',
                              save_directory=TEST_DIRECTORY, grid_size=6, type_grammar="adverb")


TEST_DATASET_NONCE = GroundedScan(intransitive_verbs=1,
                                  transitive_verbs=2,
                                  adverbs=1, nouns=3,
                                  color_adjectives=4,
                                  size_adjectives=2, percentage_train=0.8,
                                  min_object_size=1, max_object_size=4, sample_vocabulary='sample',
                                  save_directory=TEST_DIRECTORY, grid_size=15, type_grammar="adverb")

TEST_SITUATION_1 = Situation(grid_size=15, agent_position=Position(row=7, column=2), agent_direction=INT_TO_DIR[0],
                             target_object=PositionedObject(object=Object(size=2, color='red', shape='circle'),
                                                            position=Position(row=10, column=4),
                                                            vector=np.array([1, 0, 1])),
                             placed_objects=[PositionedObject(object=Object(size=2, color='red', shape='circle'),
                                                              position=Position(row=10, column=4),
                                                              vector=np.array([1, 0, 1])),
                                             PositionedObject(object=Object(size=4, color='green', shape='circle'),
                                                              position=Position(row=3, column=12),
                                                              vector=np.array([0, 1, 0]))], carrying=None)

TEST_SITUATION_2 = Situation(grid_size=15, agent_position=Position(row=7, column=2), agent_direction=INT_TO_DIR[0],
                             target_object=PositionedObject(object=Object(size=4, color='red', shape='circle'),
                                                            position=Position(row=10, column=4),
                                                            vector=np.array([1, 0, 1])),
                             placed_objects=[PositionedObject(object=Object(size=4, color='red', shape='circle'),
                                                              position=Position(row=10, column=4),
                                                              vector=np.array([1, 0, 1])),
                                             PositionedObject(object=Object(size=4, color='green', shape='cylinder'),
                                                              position=Position(row=3, column=12),
                                                              vector=np.array([0, 1, 0]))], carrying=None)

TEST_SITUATION_3 = Situation(grid_size=15, agent_position=Position(row=7, column=2), agent_direction=INT_TO_DIR[0],
                             target_object=None,
                             placed_objects=[PositionedObject(object=Object(size=1, color='red', shape='circle'),
                                                              position=Position(row=10, column=4),
                                                              vector=np.array([1, 0, 1])),
                                             PositionedObject(object=Object(size=2, color='green', shape='circle'),
                                                              position=Position(row=3, column=1),
                                                              vector=np.array([0, 1, 0]))], carrying=None)

TEST_SITUATION_4 = Situation(grid_size=15, agent_position=Position(row=7, column=2), agent_direction=INT_TO_DIR[0],
                             target_object=None,
                             placed_objects=[PositionedObject(object=Object(size=2, color='red', shape='circle'),
                                                              position=Position(row=10, column=4),
                                                              vector=np.array([1, 0, 1])),
                                             PositionedObject(object=Object(size=4, color='red', shape='circle'),
                                                              position=Position(row=3, column=1),
                                                              vector=np.array([0, 1, 0]))], carrying=None)


def test_save_and_load_dataset(dataset):
    start = time.time()
    dataset.get_data_pairs(max_examples=EXAMPLES_TO_TEST)
    dataset.save_dataset("test.txt")
    dataset.save_dataset_statistics(split="train")
    dataset.save_dataset_statistics(split="test")

    test_grounded_scan = GroundedScan.load_dataset_from_file(os.path.join(TEST_DIRECTORY, "test.txt"),
                                                             TEST_DIRECTORY)
    for example_one, example_two in zip(dataset.get_examples_with_image("train"),
                                        test_grounded_scan.get_examples_with_image("train")):
        assert dataset.command_repr(example_one["input_command"]) == test_grounded_scan.command_repr(
            example_two["input_command"]), "test_save_and_load_dataset FAILED"
        assert dataset.command_repr(example_one["target_command"]) == test_grounded_scan.command_repr(
            example_two["target_command"]), "test_save_and_load_dataset FAILED"
        assert np.array_equatest_save_and_load_datasetl(example_one["situation_image"], example_two["situation_image"]),\
            " FAILED"
        assert dataset.command_repr(example_one["input_meaning"]) == test_grounded_scan.command_repr(
            example_two["input_meaning"]), "test_save_and_load_dataset FAILED"
    os.remove(os.path.join(TEST_DIRECTORY, "test.txt"))
    end = time.time()
    logger.info("test_save_and_load_dataset PASSED in {} seconds".format(end - start))
    return


def test_save_and_load_dataset_nonce():
    start = time.time()
    TEST_DATASET_NONCE.get_data_pairs(max_examples=EXAMPLES_TO_TEST)
    TEST_DATASET_NONCE.save_dataset("test.txt")
    TEST_DATASET_NONCE.save_dataset_statistics(split="train")
    TEST_DATASET_NONCE.save_dataset_statistics(split="test")

    test_grounded_scan = GroundedScan.load_dataset_from_file(os.path.join(TEST_DIRECTORY, "test.txt"),
                                                             TEST_DIRECTORY)

    for example_one, example_two in zip(TEST_DATASET_NONCE.get_examples_with_image("train"),
                                        test_grounded_scan.get_examples_with_image("train")):
        assert TEST_DATASET_NONCE.command_repr(example_one["input_command"]) == test_grounded_scan.command_repr(
            example_two["input_command"]), "test_save_and_load_dataset FAILED"
        assert TEST_DATASET_NONCE.command_repr(example_one["target_command"]) == test_grounded_scan.command_repr(
            example_two["target_command"]), "test_save_and_load_dataset FAILED"
        assert np.array_equal(example_one["situation_image"], example_two["situation_image"]),\
            "test_save_and_load_dataset FAILED"
        assert TEST_DATASET_NONCE.command_repr(example_one["input_meaning"]) == test_grounded_scan.command_repr(
            example_two["input_meaning"]), "test_save_and_load_dataset FAILED"
    os.remove(os.path.join(TEST_DIRECTORY, "test.txt"))
    end = time.time()
    logger.info("test_save_and_load_dataset PASSED in {} seconds".format(end - start))
    return


def test_derivation_from_rules(dataset):
    start = time.time()
    derivation, arguments = dataset.sample_command()
    rules_list = []
    lexicon = {}
    derivation.to_rules(rules_list, lexicon)
    test = Derivation.from_rules(rules_list, lexicon=lexicon)
    assert ' '.join(test.words()) == ' '.join(derivation.words()), "test_derivation_from_rules FAILED"
    end = time.time()
    logger.info("test_derivation_from_rules PASSED in {} seconds".format(end - start))


def test_derivation_from_string(dataset):
    start = time.time()
    derivation, arguments = dataset.sample_command()
    derivation_str = derivation.__repr__()
    rules_str, lexicon_str = derivation_str.split(';')
    new_derivation = Derivation.from_str(rules_str, lexicon_str, dataset._grammar)
    assert ' '.join(new_derivation.words()) == ' '.join(derivation.words()), "test_derivation_from_string FAILED"
    end = time.time()
    logger.info("test_derivation_from_string PASSED in {} seconds".format(end - start))


def test_demonstrate_target_commands_one(dataset):
    """Test that target commands sequence resulting from demonstrate_command is the same as the one executed by
     demonstrate_target_commands"""
    start = time.time()
    rules_str = "NP -> NN,NP -> JJ NP,DP -> 'a' NP,VP -> VV_intrans 'to' DP,ROOT -> VP"
    translate_fn = dataset._vocabulary.translate_meaning
    lexicon_str = "T:{},NT:VV_intransitive -> {},T:to,T:a,T:{},NT:JJ -> {},T:{},NT:NN -> {}".format(
        translate_fn("walk"), translate_fn("walk"), translate_fn("small"), translate_fn("small"),
        translate_fn("circle"), translate_fn("circle")
    )
    derivation = Derivation.from_str(rules_str, lexicon_str, dataset._grammar)
    actual_target_commands, _, _ = dataset.demonstrate_command(derivation, TEST_SITUATION_1)
    command = ' '.join(derivation.words())
    target_commands, _, _, _ = dataset.demonstrate_target_commands(command, TEST_SITUATION_1, actual_target_commands)
    assert ','.join(actual_target_commands) == ','.join(target_commands),  \
        "test_demonstrate_target_commands_one FAILED"
    end = time.time()
    logger.info("test_demonstrate_target_commands_one PASSED in {} seconds".format(end - start))


def test_demonstrate_target_commands_two(dataset):
    """Test that target commands sequence resulting from demonstrate_command for pushing a heavy objectis the same as
     the executed one by demonstrate_target_commands"""
    start = time.time()
    rules_str = "NP -> NN,NP -> JJ NP,DP -> 'a' NP,VP -> VV_trans DP,ROOT -> VP"
    translate_fn = dataset._vocabulary.translate_meaning
    lexicon_str = "T:{},NT:VV_transitive -> {},T:a,T:{},NT:JJ -> {},T:{},NT:NN -> {}".format(
        translate_fn("push"), translate_fn("push"), translate_fn("big"), translate_fn("big"), translate_fn("circle"),
        translate_fn("circle")
    )
    derivation = Derivation.from_str(rules_str, lexicon_str, dataset._grammar)
    actual_target_commands, _, _ = dataset.demonstrate_command(derivation, initial_situation=TEST_SITUATION_2)
    command = ' '.join(derivation.words())
    target_commands, _, _, _ = dataset.demonstrate_target_commands(command, TEST_SITUATION_2, actual_target_commands)
    assert ','.join(actual_target_commands) == ','.join(target_commands), "test_demonstrate_target_commands_two FAILED"
    end = time.time()
    logger.info("test_demonstrate_target_commands_two PASSED in {} seconds".format(end - start))


def test_demonstrate_target_commands_three(dataset):
    """Test that target commands sequence resulting from demonstrate_command for pushing a light object is the same as
     the executed one by demonstrate_target_commands"""
    start = time.time()
    rules_str = "NP -> NN,NP -> JJ NP,DP -> 'a' NP,VP -> VV_trans DP,ROOT -> VP"
    translate_fn = dataset._vocabulary.translate_meaning
    lexicon_str = "T:{},NT:VV_transitive -> {},T:a,T:{},NT:JJ -> {},T:{},NT:NN -> {}".format(
        translate_fn("push"), translate_fn("push"), translate_fn("small"), translate_fn("small"),
        translate_fn("circle"), translate_fn("circle")
    )
    derivation = Derivation.from_str(rules_str, lexicon_str, dataset._grammar)
    actual_target_commands, _, _ = dataset.demonstrate_command(derivation, initial_situation=TEST_SITUATION_1)
    command = ' '.join(derivation.words())
    target_commands, _, _, _ = dataset.demonstrate_target_commands(command, TEST_SITUATION_1, actual_target_commands)
    assert ','.join(actual_target_commands) == ','.join(target_commands), "test_demonstrate_target_commands_three FAILED"
    end = time.time()
    logger.info("test_demonstrate_target_commands_three PASSED in {} seconds".format(end - start))


def test_demonstrate_command_one(dataset):
    """Test pushing a light object (where one target command of 'push <dir>' results in movement of 1 grid)."""
    start = time.time()
    rules_str = "NP -> NN,NP -> JJ NP,DP -> 'a' NP,VP -> VV_trans DP,ROOT -> VP"
    translate_fn = dataset._vocabulary.translate_meaning
    lexicon_str = "T:{},NT:VV_transitive -> {},T:a,T:{},NT:JJ -> {},T:{},NT:NN -> {}".format(
        translate_fn("push"), translate_fn("push"), translate_fn("small"), translate_fn("small"),
        translate_fn("circle"), translate_fn("circle")
    )
    derivation = Derivation.from_str(rules_str, lexicon_str, dataset._grammar)
    expected_target_commands = "walk,walk,turn right,walk,walk,walk,"\
                               "push,push,push,push"
    actual_target_commands, _, _ = dataset.demonstrate_command(derivation, initial_situation=TEST_SITUATION_1)
    assert expected_target_commands == ','.join(actual_target_commands), "test_demonstrate_command_one FAILED"
    end = time.time()
    logger.info("test_demonstrate_command_one PASSED in {} seconds".format(end - start))


def test_demonstrate_command_two(dataset):
    """Test pushing a heavy object (where one target command of 'push <dir>' results in movement of 1 grid)."""
    start = time.time()
    rules_str = "NP -> NN,NP -> JJ NP,DP -> 'a' NP,VP -> VV_trans DP,ROOT -> VP"
    translate_fn = dataset._vocabulary.translate_meaning
    lexicon_str = "T:{},NT:VV_transitive -> {},T:a,T:{},NT:JJ -> {},T:{},NT:NN -> {}".format(
        translate_fn("push"), translate_fn("push"), translate_fn("small"), translate_fn("small"),
        translate_fn("circle"), translate_fn("circle")
    )
    derivation = Derivation.from_str(rules_str, lexicon_str, dataset._grammar)
    expected_target_commands = "walk,walk,turn right,walk,walk,walk," \
                               "push,push,push,push,push,push,push,push"
    actual_target_commands, _, _ = dataset.demonstrate_command(derivation, initial_situation=TEST_SITUATION_2)
    assert expected_target_commands == ','.join(actual_target_commands), "test_demonstrate_command_two FAILED"
    end = time.time()
    logger.info("test_demonstrate_command_two PASSED in {} seconds".format(end - start))


def test_demonstrate_command_three(dataset):
    """Test walk to a small circle, tests that the function demonstrate command is able to find the target small circle
    even if that circle isn't explicitly set as the target object in the situation (which it wouldn't be at test time).
    """
    start = time.time()
    rules_str = "NP -> NN,NP -> JJ NP,DP -> 'a' NP,VP -> VV_intrans 'to' DP,ROOT -> VP"
    translate_fn = dataset._vocabulary.translate_meaning
    lexicon_str = "T:{},NT:VV_intransitive -> {},T:to,T:a,T:{},NT:JJ -> {},T:{},NT:NN -> {}".format(
        translate_fn("walk"), translate_fn("walk"), translate_fn("small"), translate_fn("small"), translate_fn("circle"),
        translate_fn("circle")
    )
    derivation = Derivation.from_str(rules_str, lexicon_str, dataset._grammar)
    expected_target_commands = "walk,walk,turn right,walk,walk,walk"
    actual_target_commands, _, _ = dataset.demonstrate_command(derivation, initial_situation=TEST_SITUATION_3)
    assert expected_target_commands == ','.join(actual_target_commands), "test_demonstrate_command_three FAILED"
    end = time.time()
    logger.info("test_demonstrate_command_three PASSED in {} seconds".format(end - start))


def test_demonstrate_command_four(dataset):
    """Test walk to a small circle, tests that the function demonstrate command is able to find the target big circle
    even if that circle isn't explicitly set as the target object in the situation (which it wouldn't be at test time).
    """
    start = time.time()
    rules_str = "NP -> NN,NP -> JJ NP,DP -> 'a' NP,VP -> VV_intrans 'to' DP,ROOT -> VP"
    translate_fn = dataset._vocabulary.translate_meaning
    lexicon_str = "T:{},NT:VV_intransitive -> {},T:to,T:a,T:{},NT:JJ -> {},T:{},NT:NN -> {}".format(
        translate_fn("walk"), translate_fn("walk"), translate_fn("big"), translate_fn("big"), translate_fn("circle"),
        translate_fn("circle")
    )
    derivation = Derivation.from_str(rules_str, lexicon_str, dataset._grammar)
    expected_target_commands = "turn left,turn left,walk,turn right,walk,walk,walk,walk"
    actual_target_commands, _, _ = dataset.demonstrate_command(derivation, initial_situation=TEST_SITUATION_3)
    assert expected_target_commands == ','.join(actual_target_commands), "test_demonstrate_command_four FAILED"
    end = time.time()
    logger.info("test_demonstrate_command_four PASSED in {} seconds".format(end - start))


def test_demonstrate_command_five(dataset):
    """Test that when referring to a small red circle and two present in the world, it finds the correct one."""
    start = time.time()
    rules_str = "NP -> NN,NP -> JJ NP,NP -> JJ NP,DP -> 'a' NP,VP -> VV_intrans 'to' DP,ROOT -> VP"
    translate_fn = dataset._vocabulary.translate_meaning
    lexicon_str = "T:{},NT:VV_intransitive -> {},T:to,T:a,T:{},NT:JJ -> {}:JJ -> {},T:{},T:{},NT:"\
                  "NN -> {}".format(translate_fn("walk"), translate_fn("walk"), translate_fn("red"),
                                    translate_fn("small"), translate_fn("red"), translate_fn("small"),
                                    translate_fn("circle"), translate_fn("circle"))
    derivation = Derivation.from_str(rules_str, lexicon_str, dataset._grammar)
    expected_target_commands = "walk,walk,turn right,walk,walk,walk"
    actual_target_commands, _, _ = dataset.demonstrate_command(derivation, initial_situation=TEST_SITUATION_4)
    assert expected_target_commands == ','.join(actual_target_commands), "test_demonstrate_command_five FAILED"
    end = time.time()
    logger.info("test_demonstrate_command_five PASSED in {} seconds".format(end - start))


def test_demonstrate_command_six(dataset):
    """Test that when referring to a small red circle but only one red circle is present, demonstrate_commands fails."""
    start = time.time()
    rules_str = "NP -> NN,NP -> JJ NP,NP -> JJ NP,DP -> 'a' NP,VP -> VV_intrans 'to' DP,ROOT -> VP"
    translate_fn = dataset._vocabulary.translate_meaning
    lexicon_str = "T:{},NT:VV_intransitive -> {},T:to,T:a,T:{},NT:JJ -> {}:JJ -> {},T:{},T:{},NT:" \
                  "NN -> {}".format(translate_fn("walk"), translate_fn("walk"), translate_fn("red"),
                                    translate_fn("small"), translate_fn("red"), translate_fn("small"),
                                    translate_fn("circle"), translate_fn("circle"))
    derivation = Derivation.from_str(rules_str, lexicon_str, dataset._grammar)
    expected_target_commands = ""
    try:
        actual_target_commands, _, _ = dataset.demonstrate_command(derivation, initial_situation=TEST_SITUATION_3)
    except AssertionError:
        actual_target_commands = ""
    assert expected_target_commands == ','.join(actual_target_commands), "test_demonstrate_command_six FAILED"
    end = time.time()
    logger.info("test_demonstrate_command_six PASSED in {} seconds".format(end - start))


def test_find_referred_target_one(dataset):
    """Test that for particular referred targets, the Derivation class identifies it correctly."""
    start = time.time()
    rules_str = "NP -> NN,NP -> JJ NP,NP -> JJ NP,DP -> 'a' NP,VP -> VV_intrans 'to' DP,ROOT -> VP"
    translate_fn = dataset._vocabulary.translate_meaning
    lexicon_str = "T:{},NT:VV_intransitive -> {},T:to,T:a,T:{},NT:JJ -> {}:JJ -> {},T:{},T:{},NT:" \
                  "NN -> {}".format(translate_fn("walk"), translate_fn("walk"), translate_fn("red"),
                                    translate_fn("small"), translate_fn("red"), translate_fn("small"),
                                    translate_fn("circle"), translate_fn("circle"))
    derivation = Derivation.from_str(rules_str, lexicon_str, dataset._grammar)
    arguments = []
    derivation.meaning(arguments)
    assert len(arguments) == 1, "test_find_referred_target_one FAILED."
    target_str, target_predicate = arguments.pop().to_predicate()
    translate_fn_word = dataset._vocabulary.translate_word
    translated_target_str = ' '.join([translate_fn_word(word) for word in target_str.split()])
    assert translated_target_str == "red circle", "test_find_referred_target FAILED."
    assert target_predicate["noun"] == translate_fn("circle"), "test_find_referred_target_one FAILED."
    assert target_predicate["size"] == translate_fn("small"), "test_find_referred_target_one FAILED."
    assert target_predicate["color"] == translate_fn("red"), "test_find_referred_target_one FAILED."
    end = time.time()
    logger.info("test_find_referred_target_one PASSED in {} seconds".format(end - start))


def test_find_referred_target_two(dataset):
    """Test that for particular referred targets, the Derivation class identifies it correctly."""
    start = time.time()
    rules_str = "NP -> NN,NP -> JJ NP,DP -> 'a' NP,VP -> VV_intrans 'to' DP,ROOT -> VP"
    translate_fn = dataset._vocabulary.translate_meaning
    lexicon_str = "T:{},NT:VV_intransitive -> {},T:to,T:a,T:{},NT:JJ -> {},T:{},NT:NN -> {}".format(
        translate_fn("walk"), translate_fn("walk"), translate_fn("big"), translate_fn("big"), translate_fn("circle"),
        translate_fn("circle")
    )
    derivation = Derivation.from_str(rules_str, lexicon_str, dataset._grammar)
    arguments = []
    derivation.meaning(arguments)
    assert len(arguments) == 1, "test_find_referred_target_two FAILED."
    target_str, target_predicate = arguments.pop().to_predicate()
    translate_fn_word = dataset._vocabulary.translate_word
    translated_target_str = ' '.join([translate_fn_word(word) for word in target_str.split()])
    assert translated_target_str == "circle", "test_find_referred_target_two FAILED."
    assert target_predicate["noun"] == translate_fn("circle"), "test_find_referred_target_two FAILED."
    assert target_predicate["size"] == translate_fn("big"), "test_find_referred_target_two FAILED."
    assert target_predicate["color"] == translate_fn(""), "test_find_referred_target_two FAILED."
    end = time.time()
    logger.info("test_find_referred_target_two PASSED in {} seconds".format(end - start))


def test_generate_possible_targets_one(dataset):
    """Test that for particular referred targets, the right possible target objects get generated."""
    start = time.time()
    translate_meaning = dataset._vocabulary.translate_meaning
    target_predicate = {"noun": translate_meaning("circle"),
                        "color": translate_meaning("red"),
                        "size": translate_meaning("big")}
    translate_word = dataset._vocabulary.translate_word
    expected_possible_targets = {(2, "red", "circle"), (3, "red", "circle"), (4, "red", "circle")}
    actual_possible_targets = dataset.generate_possible_targets(
        referred_size=translate_word(target_predicate["size"]),
        referred_color=translate_word(target_predicate["color"]),
        referred_shape=translate_word(target_predicate["noun"]))
    for actual_possible_target in actual_possible_targets:
        assert actual_possible_target in expected_possible_targets, "test_generate_possible_targets_one FAILED."
    end = time.time()
    logger.info("test_generate_possible_targets_one PASSED in {} seconds".format(end - start))


def test_generate_possible_targets_two(dataset):
    """Test that for particular referred targets, the right possible target objects get generated."""
    start = time.time()
    translate_meaning = dataset._vocabulary.translate_meaning
    target_predicate = {"noun": translate_meaning("circle"),
                        "color": translate_meaning("red"),
                        "size": translate_meaning("small")}
    translate_word = dataset._vocabulary.translate_word
    expected_possible_targets = {(1, "red", "circle"), (2, "red", "circle"), (3, "red", "circle"),
                                 (1, "blue", "circle"), (2, "blue", "circle"), (3, "blue", "circle"),
                                 (1, "green", "circle"), (2, "green", "circle"), (3, "green", "circle")}
    actual_possible_targets = dataset.generate_possible_targets(
        referred_size=translate_word(target_predicate["size"]),
        referred_color=translate_word(target_predicate["color"]),
        referred_shape=translate_word(target_predicate["noun"]))
    for expected_possible_target, actual_possible_target in zip(expected_possible_targets, actual_possible_targets):
        assert actual_possible_target in expected_possible_targets, "test_generate_possible_targets_two FAILED."
    end = time.time()
    logger.info("test_generate_possible_targets_two PASSED in {} seconds".format(end - start))


def test_generate_situations_one(dataset):
    """Test that when a small green circle is referred to there exist no smaller green circles than the target object in
    the world and at least one larger green circle."""
    start = time.time()
    translate_meaning = dataset._vocabulary.translate_meaning
    target_shape = "circle"
    target_color = "green"
    target_size = 2
    referred_size = translate_meaning("small")
    referred_color = translate_meaning("green")
    referred_shape = translate_meaning("circle")
    situation_specifications = dataset.generate_situations(num_resampling=1)
    relevant_situation = situation_specifications[target_shape][target_color][target_size].pop()
    dataset.initialize_world_from_spec(relevant_situation, referred_size=referred_size,
                                       referred_color=referred_color,
                                       referred_shape=referred_shape,
                                       actual_size=target_size,
                                       sample_percentage=0.5
                                       )
    smallest_object = dataset._world.object_positions("green circle",
                                                      object_size="small").pop()
    assert smallest_object == relevant_situation["target_position"], "test_generate_situations_one FAILED."
    other_related_objects = dataset._world.object_positions("green circle")
    larger_objects = []
    for size, sized_objects in other_related_objects:
        if size < target_size:
            assert not sized_objects, "test_generate_situations_one FAILED."
        elif size > target_size:
            larger_objects.extend(sized_objects)
    assert len(larger_objects) >= 1, "test_generate_situations_one FAILED."
    end = time.time()
    logger.info("test_generate_situations_one PASSED in {} seconds".format(end - start))


def test_generate_situations_two(dataset):
    """Test that when a big green circle is referred to there exists no larger green circles and the exists at least
    one smaller green circle."""
    start = time.time()
    translate_meaning = dataset._vocabulary.translate_meaning
    target_shape = "circle"
    target_color = "green"
    target_size = 2
    referred_size = translate_meaning("big")
    referred_color = translate_meaning("green")
    referred_shape = translate_meaning("circle")
    situation_specifications = dataset.generate_situations(num_resampling=1)
    relevant_situation = situation_specifications[target_shape][target_color][target_size].pop()
    dataset.initialize_world_from_spec(relevant_situation, referred_size=referred_size,
                                       referred_color=referred_color,
                                       referred_shape=referred_shape,
                                       actual_size=target_size,
                                       sample_percentage=0.5
                                       )
    largest_object = dataset._world.object_positions("green circle",
                                                           object_size="big").pop()
    assert largest_object == relevant_situation["target_position"], "test_generate_situations_two FAILED."
    other_related_objects = dataset._world.object_positions("green circle")
    smaller_objects = []
    for size, sized_objects in other_related_objects:
        if size > target_size:
            assert not sized_objects, "test_generate_situations_two FAILED."
        elif size < target_size:
            smaller_objects.extend(sized_objects)
    assert len(smaller_objects) >= 1, "test_generate_situations_two FAILED."
    end = time.time()
    logger.info("test_generate_situations_two PASSED in {} seconds".format(end - start))


def test_generate_situations_three(dataset):
    """Test that for particular commands the right situations get matched."""
    start = time.time()
    translate_meaning = dataset._vocabulary.translate_meaning
    target_shape = "circle"
    target_color = "green"
    target_size = 2
    referred_size = translate_meaning("big")
    referred_shape = translate_meaning("circle")
    situation_specifications = dataset.generate_situations(num_resampling=1)
    relevant_situation = situation_specifications[target_shape][target_color][target_size].pop()
    dataset.initialize_world_from_spec(relevant_situation, referred_size=referred_size,
                                       referred_color="",
                                       referred_shape=referred_shape,
                                       actual_size=target_size,
                                       sample_percentage=0.5
                                       )
    largest_object = dataset._world.object_positions("circle",
                                                          object_size="big").pop()
    assert largest_object == relevant_situation["target_position"], "test_generate_situations_three FAILED."
    other_related_objects = dataset._world.object_positions("circle")
    smaller_objects = []
    for size, sized_objects in other_related_objects:
        if size > target_size:
            assert not sized_objects, "test_generate_situations_three FAILED."
        elif size < target_size:
            smaller_objects.extend(sized_objects)
    assert len(smaller_objects) >= 1, "test_generate_situations_three FAILED."
    end = time.time()
    logger.info("test_generate_situations_three PASSED in {} seconds".format(end - start))


def test_situation_representation_eq():
    start = time.time()
    test_situations = [TEST_SITUATION_1, TEST_SITUATION_2, TEST_SITUATION_3, TEST_SITUATION_4]
    for i, test_situation_1 in enumerate(test_situations):
        for j, test_situation_2 in enumerate(test_situations):
            if i == j:
                assert test_situation_1 == test_situation_2, "test_situation_representation_eq FAILED."
            else:
                assert test_situation_1 != test_situation_2, "test_situation_representation_eq FAILED."
    end = time.time()
    logger.info("test_situation_representation_eq PASSED in {} seconds".format(end - start))


def test_example_representation_eq(dataset):
    """Test that the function for comparing examples returns true when exactly the same example is passed twice."""
    start = time.time()
    rules_str = "NP -> NN,NP -> JJ NP,DP -> 'a' NP,VP -> VV_intrans 'to' DP,ROOT -> VP"
    translate_fn = dataset._vocabulary.translate_meaning
    lexicon_str = "T:{},NT:VV_intransitive -> {},T:to,T:a,T:{},NT:JJ -> {},T:{},NT:NN -> {}".format(
        translate_fn("walk"), translate_fn("walk"), translate_fn("big"), translate_fn("big"), translate_fn("circle"),
        translate_fn("circle")
    )
    derivation = Derivation.from_str(rules_str, lexicon_str, dataset._grammar)
    arguments = []
    derivation.meaning(arguments)
    target_str, target_predicate = arguments.pop().to_predicate()
    adverb = ""
    for word in derivation.words():
        if word in dataset._vocabulary.get_adverbs():
            adverb = word
    target_commands, _, target_action = dataset.demonstrate_command(derivation, initial_situation=TEST_SITUATION_1)
    TEST_DATASET.fill_example(derivation.words(), derivation, TEST_SITUATION_1, target_commands, target_action,
                              target_predicate, visualize=False, splits=["train"], adverb=adverb)
    TEST_DATASET.get_data_pairs(max_examples=10, num_resampling=2)
    for split, examples in dataset._data_pairs.items():
        for example in examples:
            assert dataset.compare_examples(example, example), "test_example_representation_eq FAILED."
    end = time.time()
    logger.info("test_example_representation_eq PASSED in {} seconds".format(end - start))


def test_example_representation(dataset):
    """Test that when you save an example in its representation its the same if you parse it again."""
    start = time.time()
    rules_str = "NP -> NN,NP -> JJ NP,DP -> 'a' NP,VP -> VV_intrans 'to' DP,ROOT -> VP"
    translate_fn = dataset._vocabulary.translate_meaning
    lexicon_str = "T:{},NT:VV_intransitive -> {},T:to,T:a,T:{},NT:JJ -> {},T:{},NT:NN -> {}".format(
        translate_fn("walk"), translate_fn("walk"), translate_fn("big"), translate_fn("big"), translate_fn("circle"),
        translate_fn("circle")
    )
    derivation = Derivation.from_str(rules_str, lexicon_str, dataset._grammar)
    arguments = []
    derivation.meaning(arguments)
    target_str, target_predicate = arguments.pop().to_predicate()
    adverb = ""
    for word in derivation.words():
        if word in dataset._vocabulary.get_adverbs():
            adverb = word
    target_commands, _, target_action = dataset.demonstrate_command(derivation, initial_situation=TEST_SITUATION_1)
    dataset.fill_example(derivation.words(), derivation, TEST_SITUATION_1, target_commands, target_action,
                         target_predicate, visualize=False, splits=["train"], adverb=adverb)
    example = dataset._data_pairs["train"].pop()
    (parsed_command, parsed_meaning, parsed_derivation, parsed_situation,
     parsed_target_commands, _, parsed_action) = dataset.parse_example(
        example
    )
    assert example["command"] == dataset.command_repr(parsed_command), "test_example_representation FAILED."
    assert example["meaning"] == dataset.command_repr(parsed_meaning), "test_example_representation FAILED."
    assert example["derivation"] == dataset.derivation_repr(parsed_derivation), "test_example_representation "\
                                                                                     "FAILED."
    situation = Situation.from_representation(example["situation"])
    assert situation == parsed_situation, "test_example_representation FAILED."
    assert example["target_commands"] == dataset.command_repr(parsed_target_commands), \
        "test_example_representation FAILED."
    assert example["verb_in_command"] == dataset._vocabulary.translate_word(parsed_action),\
        "test_example_representation FAILED."
    assert example["referred_target"] == ' '.join([dataset._vocabulary.translate_word(target_predicate["size"]),
                                                   dataset._vocabulary.translate_word(target_predicate["color"]),
                                                   dataset._vocabulary.translate_word(target_predicate["noun"])]),\
        "test_example_representation FAILED."
    end = time.time()
    logger.info("test_example_representation PASSED in {} seconds".format(end - start))


def test_initialize_world(dataset):
    """Test that two the same situations get represented in exactly the same image by rendering.py and minigrid.py"""
    start = time.time()
    test_situations = [TEST_SITUATION_1, TEST_SITUATION_2, TEST_SITUATION_3, TEST_SITUATION_4]
    current_situation = dataset._world.get_current_situation()
    current_mission = dataset._world.mission
    for i, test_situation_1 in enumerate(test_situations):
        for j, test_situation_2 in enumerate(test_situations):
            dataset._world.clear_situation()
            dataset.initialize_world(test_situation_1)
            situation_1 = dataset._world.get_current_situation()
            dataset._world.clear_situation()
            dataset.initialize_world(test_situation_2)
            situation_2 = dataset._world.get_current_situation()
            if i == j:
                assert situation_1 == situation_2, "test_initialize_world FAILED."
            else:
                assert situation_1 != situation_2, "test_initialize_world FAILED."
    dataset.initialize_world(current_situation, mission=current_mission)
    end = time.time()
    logger.info("test_initialize_world PASSED in {} seconds".format(end - start))


def test_generate_manners():
    intransitive_verbs = ["walk"]
    transitive_verbs = ["push", "pull"]
    adverbs = ["manner1", "manner2", "manner3",
               "cautiously", "while spinning", "while zigzagging", "hesitantly"]
    nouns = ["circle", "cylinder", "square"]
    color_adjectives = ["red", "blue", "green", "yellow"]
    size_adjectives = ["big", "small"]
    situation = Situation(grid_size=6, agent_position=Position(row=1, column=2), agent_direction=INT_TO_DIR[0],
                          target_object=PositionedObject(object=Object(size=2, color='red', shape='circle'),
                                                         position=Position(row=4, column=2),
                                                         vector=np.array([1, 0, 1])),
                          placed_objects=[PositionedObject(object=Object(size=2, color='red', shape='circle'),
                                                           position=Position(row=4, column=2),
                                                           vector=np.array([1, 0, 1])),
                                          PositionedObject(object=Object(size=4, color='green', shape='circle'),
                                                           position=Position(row=5, column=5),
                                                           vector=np.array([0, 1, 0]))], carrying=None)
    dataset = GroundedScan(intransitive_verbs=intransitive_verbs,
                           transitive_verbs=transitive_verbs,
                           adverbs=adverbs, nouns=nouns,
                           color_adjectives=color_adjectives,
                           size_adjectives=size_adjectives, percentage_train=0.8,
                           min_object_size=1, max_object_size=4, sample_vocabulary='default',
                           save_directory=TEST_DIRECTORY, grid_size=6, type_grammar="adverb")
    generated_manners = dataset._world.left_over_manners.keys()
    assert len(generated_manners) == 3, "test_generate_manners() FAILED."
    for extra_manner in adverbs[:3]:
        deriv = dataset._grammar.sample_constrained(lexical_rule="RB -> {}".format(extra_manner))
        target_commands_1, target_demonstration_1, action_1 = dataset.demonstrate_command(
            deriv, initial_situation=situation)
        command = deriv.words()
        meaning = command
        manner = dataset._world.left_over_manners[extra_manner]
        mission = ' '.join(["Command:", ' '.join(command), "\nMeaning: ", ' '.join(meaning), "\nManner: ",
                            ' '.join(manner),
                            "\nTarget:"] + target_commands_1)
        save_dir = dataset.visualize_command(situation, command, target_demonstration_1,
                                             mission=mission)
    return


def test_image_representation_situations(dataset):
    """Test that situations are still the same when they need to be in image / numpy RGB array form."""
    start = time.time()
    current_situation = dataset._world.get_current_situation()
    current_mission = dataset._world.mission
    test_situations = [TEST_SITUATION_1, TEST_SITUATION_2, TEST_SITUATION_3, TEST_SITUATION_4]
    for i, test_situation_1 in enumerate(test_situations):
        for j, test_situation_2 in enumerate(test_situations):
            dataset._world.clear_situation()
            dataset.initialize_world(test_situation_1)
            np_situation_image_1 = dataset._world.render(mode='human').getArray()
            numpy_array_to_image(np_situation_image_1, os.path.join(TEST_DIRECTORY, "test_im_1.png"))
            np_situation_image_1_reread = image_to_numpy_array(os.path.join(TEST_DIRECTORY, "test_im_1.png"))
            assert np.array_equal(np_situation_image_1,
                                  np_situation_image_1_reread), "test_image_representation_situations FAILED."
            dataset._world.clear_situation()
            dataset.initialize_world(test_situation_2)
            np_situation_image_2 = dataset._world.render().getArray()
            numpy_array_to_image(np_situation_image_2, os.path.join(TEST_DIRECTORY, "test_im_2.png"))
            np_situation_image_2_reread = image_to_numpy_array(os.path.join(TEST_DIRECTORY, "test_im_2.png"))
            assert np.array_equal(np_situation_image_2,
                                  np_situation_image_2_reread), "test_image_representation_situations FAILED."
            if i == j:
                assert np.array_equal(np_situation_image_1, np_situation_image_2), \
                    "test_image_representation_situations FAILED."
            else:
                assert not np.array_equal(np_situation_image_1, np_situation_image_2), \
                    "test_image_representation_situations FAILED."
    os.remove(os.path.join(TEST_DIRECTORY, "test_im_1.png"))
    os.remove(os.path.join(TEST_DIRECTORY, "test_im_2.png"))
    dataset.initialize_world(current_situation, mission=current_mission)
    end = time.time()
    logger.info("test_image_representation_situations PASSED in {} seconds".format(end - start))


def test_encode_situation(dataset):
    start = time.time()
    current_situation = dataset._world.get_current_situation()
    current_mission = dataset._world.mission
    test_situation = Situation(grid_size=15, agent_position=Position(row=7, column=2), agent_direction=INT_TO_DIR[0],
                               target_object=PositionedObject(object=Object(size=2, color='red', shape='circle'),
                                                              position=Position(row=7, column=2),
                                                              vector=np.array([1, 0, 1])),
                               placed_objects=[PositionedObject(object=Object(size=2, color='red', shape='circle'),
                                                                position=Position(row=7, column=2),
                                                                vector=np.array([1, 0, 1])),
                                               PositionedObject(object=Object(size=4, color='green', shape='circle'),
                                                                position=Position(row=3, column=12),
                                                                vector=np.array([0, 1, 0]))], carrying=None)
    dataset._world.clear_situation()
    dataset.initialize_world(test_situation)
    expected_numpy_array = np.zeros([15, 15, dataset._world.grid._num_attributes_object + 1 + 4], dtype='uint8')
    expected_numpy_array[7, 2, -5] = 1
    expected_numpy_array[7, 2, -4:] = np.array([1, 0, 0, 0])
    expected_numpy_array[7, 2, :-5] = dataset._object_vocabulary.get_object_vector(shape='circle', color='red',
                                                                                        size=2)
    expected_numpy_array[3, 12, :-5] = dataset._object_vocabulary.get_object_vector(shape='circle', color='green',
                                                                                         size=4)
    encoded_numpy_array = dataset._world.grid.encode(agent_row=7, agent_column=2, agent_direction=0)
    assert np.array_equal(expected_numpy_array, encoded_numpy_array), "test_encode_situation FAILED."
    dataset.initialize_world(current_situation, mission=current_mission)
    end = time.time()
    logger.info("test_encode_situation PASSED in {} seconds".format(end - start))


def test_k_shot_generalization(dataset):
    start = time.time()
    current_situation = dataset._world.get_current_situation()
    current_mission = dataset._world.mission
    k_shot_generalization = 5
    dataset.get_data_pairs(max_examples=100000, num_resampling=1, other_objects_sample_percentage=0.5,
                           split_type="generalization", k_shot_generalization=k_shot_generalization)
    # Test that all the splits only contain examples related to their split.
    visual_split_examples = dataset._data_pairs["visual"]
    for example in visual_split_examples:
        target_object = example["situation"]["target_object"]["object"]
        assert target_object["shape"] == "square" and target_object["color"] == "red", \
            "test_k_shot_generalization FAILED in split visual."
    situational_split_1 = dataset._data_pairs["situational_1"]
    for example in situational_split_1:
        direction_to_target = example["situation"]["direction_to_target"]
        assert direction_to_target == "sw", "test_k_shot_generalization FAILED in split situational_1."
    situational_split_2 = dataset._data_pairs["situational_2"]
    for example in situational_split_2:
        referred_target = example["referred_target"]
        assert "small" in referred_target, \
            "test_k_shot_generalization FAILED in split situational_2."
        target_size = example["situation"]["target_object"]["object"]["size"]
        assert target_size == '2', "test_k_shot_generalization FAILED in split situational_2."
    contextual_split = dataset._data_pairs["contextual"]
    for example in contextual_split:
        assert (dataset._vocabulary.translate_meaning(example["verb_in_command"])
                in dataset._vocabulary.get_transitive_verbs()), \
            "test_k_shot_generalization FAILED in split contextual."
        target_object = example["situation"]["target_object"]["object"]
        assert target_object["shape"] == "square" and target_object["size"] == '3', \
            "test_k_shot_generalization FAILED in split contextual."

    # Test that the training set doesn't contain more than k examples of each of the test splits.
    examples_per_split = {"visual": 0, "situational_1": 0, "situational_2": 0, "contextual": 0, "adverb_1": 0}
    for example in dataset._data_pairs["train"]:
        target_object = example["situation"]["target_object"]["object"]
        target_size = target_object["size"]
        direction_to_target = example["situation"]["direction_to_target"]
        referred_target = example["referred_target"]
        if target_object["shape"] == "square" and target_object["color"] == "red":
            examples_per_split["visual"] += 1
        if direction_to_target == "sw":
            examples_per_split["situational_1"] += 1
        if "small" in referred_target and target_size == 2:
            examples_per_split["situational_2"] += 1
        if (dataset._vocabulary.translate_meaning(example["verb_in_command"]) in
                dataset._vocabulary.get_transitive_verbs() and
                target_object["shape"] == "square" and target_object["size"] == '3'):
            examples_per_split["contextual"] += 1
    for split, examples_count in examples_per_split.items():
        if split == "adverb_1":
            assert examples_count == k_shot_generalization, \
             "test_k_shot_generalization FAILED in split train for split {}.".format(split)
        else:
            assert examples_count == 0, "test_k_shot_generalization FAILED in split train for split {}.".format(split)
    dataset.initialize_world(current_situation, mission=current_mission)
    end = time.time()
    logger.info("test_k_shot_generalization PASSED in {} seconds".format(end - start))


def test_apply_while_spinning():
    start = time.time()
    meta_grammar = dsl.MetaGrammar()
    while_spinning = dsl.LSystem()
    while_spinning.add_rule(meta_grammar.get_rule(lhs_str="Walk",
                                                  rhs_str="ACTION Walk"))
    while_spinning.add_rule(meta_grammar.get_rule(lhs_str="ACTION",
                                                  rhs_str="Tl"),
                            terminal_rule=True)
    while_spinning.add_rule(meta_grammar.get_rule(lhs_str="Push",
                                                  rhs_str="ACTION Push"))
    while_spinning.finish_l_system()

    while_spinning_2 = dsl.apply_recursion(while_spinning, max_recursion=4)

    # First sequence
    sequence = dsl.Sequence()
    sequence.extend([dsl.EMPTY, dsl.Walk, dsl.Tl, dsl.Walk, dsl.EMPTY])
    dsl.apply_lsystem(sequence, while_spinning_2, 0, 1)
    expected_str_sequence = "Tl Tl Tl Tl Walk Tl Tl Tl Tl Tl Walk"
    actual_str_sequence = str(sequence)
    assert actual_str_sequence == expected_str_sequence, \
        "test_apply_while_spinning FAILED for \nexpected %s \nactual: %s" % (expected_str_sequence, str(sequence))

    # Second sequence
    sequence = dsl.Sequence()
    sequence.extend([dsl.EMPTY, dsl.Push, dsl.Tl, dsl.Walk, dsl.EMPTY])
    dsl.apply_lsystem(sequence, while_spinning, 0, 4)
    expected_str_sequence = "Tl Tl Tl Tl Push Tl Tl Tl Tl Tl Walk"
    actual_str_sequence = str(sequence)
    assert actual_str_sequence == expected_str_sequence, \
        "test_apply_while_spinning FAILED for \nexpected %s \nactual: %s" % (expected_str_sequence, str(sequence))

    # Third sequence
    sequence = dsl.Sequence()
    sequence.extend([dsl.EMPTY, dsl.Tl, dsl.Stay, dsl.EMPTY])
    dsl.apply_lsystem(sequence, while_spinning_2, 0, 1)
    expected_str_sequence = "Tl Stay"
    actual_str_sequence = str(sequence)
    assert actual_str_sequence == expected_str_sequence, \
        "test_apply_while_spinning FAILED for \nexpected %s \nactual: %s" % (expected_str_sequence, str(sequence))

    # Fourth sequence
    sequence = dsl.Sequence()
    sequence.extend([dsl.EMPTY, dsl.Tl, dsl.Tr, dsl.Walk, dsl.Push, dsl.Push, dsl.EMPTY])
    dsl.apply_lsystem(sequence, while_spinning, 0, 4)
    expected_str_sequence = "Tl Tr Tl Tl Tl Tl Walk Tl Tl Tl Tl Push Tl Tl Tl Tl Push"
    actual_str_sequence = str(sequence)
    assert actual_str_sequence == expected_str_sequence, \
        "test_apply_while_spinning FAILED for \nexpected %s \nactual: %s" % (expected_str_sequence, str(sequence))

    end = time.time()
    logger.info("test_apply_while_spinning PASSED in {} seconds".format(end - start))


def test_apply_cautiously():
    start = time.time()
    meta_grammar = dsl.MetaGrammar()
    cautiously = dsl.LSystem()
    cautiously.add_rule(meta_grammar.get_rule(lhs_str="Walk",
                                              rhs_str="ACTION Walk"))
    cautiously.add_rule(meta_grammar.get_rule(lhs_str="Push",
                                              rhs_str="ACTION Push"))
    cautiously.add_rule(meta_grammar.get_rule(lhs_str="{ACTION}ACTION ACTION{ACTION}",
                                              rhs_str="Tl Tl"),
                        terminal_rule=True)
    cautiously.add_rule(meta_grammar.get_rule(lhs_str="ACTION",
                                              rhs_str="Tr"),
                        terminal_rule=True)
    cautiously.finish_l_system()

    cautiously_2 = dsl.apply_recursion(cautiously, max_recursion=4)

    # First sequence
    sequence = dsl.Sequence()
    sequence.extend([dsl.EMPTY, dsl.Walk, dsl.Tl, dsl.Walk, dsl.EMPTY])
    dsl.apply_lsystem(sequence, cautiously_2, 0, 1)
    expected_str_sequence = "Tr Tl Tl Tr Walk Tl Tr Tl Tl Tr Walk"
    actual_str_sequence = str(sequence)
    assert actual_str_sequence == expected_str_sequence, \
        "test_apply_cautiously FAILED for \nexpected %s \nactual: %s" % (expected_str_sequence, str(sequence))

    # Second sequence
    sequence = dsl.Sequence()
    sequence.extend([dsl.Tl])
    dsl.apply_lsystem(sequence, cautiously, 0, 4)
    expected_str_sequence = "Tl"
    actual_str_sequence = str(sequence)
    assert actual_str_sequence == expected_str_sequence, \
        "test_apply_cautiously FAILED for \nexpected %s \nactual: %s" % (expected_str_sequence, str(sequence))

    # Third sequence
    sequence = dsl.Sequence()
    sequence.extend([dsl.EMPTY, dsl.Push, dsl.Tr, dsl.Tr, dsl.Push, dsl.EMPTY])
    dsl.apply_lsystem(sequence, cautiously_2, 0, 1)
    expected_str_sequence = "Tr Tl Tl Tr Push Tr Tr Tr Tl Tl Tr Push"
    actual_str_sequence = str(sequence)
    assert actual_str_sequence == expected_str_sequence, \
        "test_apply_cautiously FAILED for \nexpected %s \nactual: %s" % (expected_str_sequence, str(sequence))

    # Fourth sequence
    sequence = dsl.Sequence()
    sequence.extend([dsl.EMPTY, dsl.Walk, dsl.EMPTY])
    dsl.apply_lsystem(sequence, cautiously, 0, 4)
    expected_str_sequence = "Tr Tl Tl Tr Walk"
    actual_str_sequence = str(sequence)
    assert actual_str_sequence == expected_str_sequence, \
        "test_apply_cautiously FAILED for \nexpected %s \nactual: %s" % (expected_str_sequence, str(sequence))

    end = time.time()
    logger.info("test_apply_cautiously PASSED in {} seconds".format(end - start))


def test_apply_hesitantly():
    start = time.time()
    meta_grammar = dsl.MetaGrammar()
    hesitantly = dsl.LSystem()
    hesitantly.add_rule(meta_grammar.get_rule(lhs_str="Walk",
                                              rhs_str="Walk ACTION"))
    hesitantly.add_rule(meta_grammar.get_rule(lhs_str="Push",
                                              rhs_str="Push ACTION"))
    hesitantly.add_rule(meta_grammar.get_rule(lhs_str="ACTION",
                                              rhs_str="Stay"),
                        terminal_rule=True)
    hesitantly.finish_l_system()

    hesitantly_2 = dsl.apply_recursion(hesitantly, max_recursion=1)

    # First sequence
    sequence = dsl.Sequence()
    sequence.extend([dsl.EMPTY, dsl.Walk, dsl.Tl, dsl.Walk, dsl.EMPTY])
    dsl.apply_lsystem(sequence, hesitantly, 0, 1)
    expected_str_sequence = "Walk Stay Tl Walk Stay"
    actual_str_sequence = str(sequence)
    assert actual_str_sequence == expected_str_sequence, \
        "test_apply_hesitantly FAILED for \nexpected %s \nactual: %s" % (expected_str_sequence, str(sequence))

    # Second sequence
    sequence = dsl.Sequence()
    sequence.extend([dsl.EMPTY, dsl.Tl, dsl.Walk, dsl.Walk, dsl.EMPTY])
    dsl.apply_lsystem(sequence, hesitantly_2, 0, 1)
    expected_str_sequence = "Tl Walk Stay Walk Stay"
    actual_str_sequence = str(sequence)
    assert actual_str_sequence == expected_str_sequence, \
        "test_apply_hesitantly FAILED for \nexpected %s \nactual: %s" % (expected_str_sequence, str(sequence))

    # Third sequence
    sequence = dsl.Sequence()
    sequence.extend([dsl.EMPTY, dsl.Tl, dsl.EMPTY])
    dsl.apply_lsystem(sequence, hesitantly, 0, 1)
    expected_str_sequence = "Tl"
    actual_str_sequence = str(sequence)
    assert actual_str_sequence == expected_str_sequence, \
        "test_apply_hesitantly FAILED for \nexpected %s \nactual: %s" % (expected_str_sequence, str(sequence))

    # Third sequence
    sequence = dsl.Sequence()
    sequence.extend([dsl.EMPTY, dsl.Push, dsl.Walk, dsl.EMPTY])
    dsl.apply_lsystem(sequence, hesitantly_2, 0, 1)
    expected_str_sequence = "Push Stay Walk Stay"
    actual_str_sequence = str(sequence)
    assert actual_str_sequence == expected_str_sequence, \
        "test_apply_hesitantly FAILED for \nexpected %s \nactual: %s" % (expected_str_sequence, str(sequence))

    # Fourth sequence
    sequence = dsl.Sequence()
    sequence.extend([dsl.EMPTY])
    dsl.apply_lsystem(sequence, hesitantly, 0, 1)
    expected_str_sequence = ""
    actual_str_sequence = str(sequence)
    assert actual_str_sequence == expected_str_sequence, \
        "test_apply_hesitantly FAILED for \nexpected %s \nactual: %s" % (expected_str_sequence, str(sequence))

    end = time.time()
    logger.info("test_apply_hesitantly PASSED in {} seconds".format(end - start))


def test_apply_while_zigzagging():
    start = time.time()
    meta_grammar = dsl.MetaGrammar()
    while_zigzagging = dsl.LSystem()
    while_zigzagging.add_rule(meta_grammar.get_rule(lhs_str="East South",
                                                    rhs_str="South East"))
    while_zigzagging.add_rule(meta_grammar.get_rule(lhs_str="East North",
                                                    rhs_str="North East"))
    while_zigzagging.add_rule(meta_grammar.get_rule(lhs_str="West North",
                                                    rhs_str="North West"))
    while_zigzagging.add_rule(meta_grammar.get_rule(lhs_str="West South",
                                                    rhs_str="South West"))
    while_zigzagging.finish_l_system()

    # First sequence
    sequence = dsl.Sequence()
    # 3 cols, 2 rows
    sequence.extend([dsl.East, dsl.East, dsl.East, dsl.South, dsl.South])
    dsl.apply_lsystem(sequence, while_zigzagging, 0, 2)
    expected_str_sequence = "East South East South East"
    actual_str_sequence = str(sequence)
    assert actual_str_sequence == expected_str_sequence, \
        "test_apply_while_zigzagging FAILED for \nexpected %s \nactual: %s" % (expected_str_sequence, str(sequence))

    # Second sequence
    sequence = dsl.Sequence()
    # 5 cols, 2 rows
    sequence.extend([dsl.East, dsl.East, dsl.East, dsl.East, dsl.East, dsl.South, dsl.South])
    dsl.apply_lsystem(sequence, while_zigzagging, 0, 4)
    expected_str_sequence = "East South East South East East East"
    actual_str_sequence = str(sequence)
    assert actual_str_sequence == expected_str_sequence, \
        "test_apply_while_zigzagging FAILED for \nexpected %s \nactual: %s" % (expected_str_sequence, str(sequence))

    # Third sequence
    sequence = dsl.Sequence()
    # 2 cols, 5 rows
    sequence.extend([dsl.East, dsl.East, dsl.South, dsl.South, dsl.South, dsl.South, dsl.South])
    dsl.apply_lsystem(sequence, while_zigzagging, 0, 1)
    expected_str_sequence = "East South East South South South South"
    actual_str_sequence = str(sequence)
    assert actual_str_sequence == expected_str_sequence, \
        "test_apply_while_zigzagging FAILED for \nexpected %s \nactual: %s" % (expected_str_sequence, str(sequence))

    # Fourth sequence
    sequence = dsl.Sequence()
    # 2 cols, 5 rows
    sequence.extend([dsl.West, dsl.West, dsl.South, dsl.South, dsl.South, dsl.South, dsl.South])
    dsl.apply_lsystem(sequence, while_zigzagging, 0, 1)
    expected_str_sequence = "West South West South South South South"
    actual_str_sequence = str(sequence)
    assert actual_str_sequence == expected_str_sequence, \
        "test_apply_while_zigzagging FAILED for \nexpected %s \nactual: %s" % (expected_str_sequence, str(sequence))

    # Fifth sequence
    sequence = dsl.Sequence()
    # 3 cols
    sequence.extend([dsl.West, dsl.West, dsl.West])
    dsl.apply_lsystem(sequence, while_zigzagging, 0, 4)
    expected_str_sequence = "West West West"
    actual_str_sequence = str(sequence)
    assert actual_str_sequence == expected_str_sequence, \
        "test_apply_while_zigzagging FAILED for \nexpected %s \nactual: %s" % (expected_str_sequence, str(sequence))

    # Sixt sequence
    sequence = dsl.Sequence()
    # 3 cols, 2 rows
    sequence.extend([dsl.West, dsl.West, dsl.West, dsl.North, dsl.North])
    dsl.apply_lsystem(sequence, while_zigzagging, 0, 2)
    expected_str_sequence = "West North West North West"
    actual_str_sequence = str(sequence)
    assert actual_str_sequence == expected_str_sequence, \
        "test_apply_while_zigzagging FAILED for \nexpected %s \nactual: %s" % (expected_str_sequence, str(sequence))

    end = time.time()
    logger.info("test_apply_while_zigzagging PASSED in {} seconds".format(end - start))


def test_convert_sequence_to_actions():
    start = time.time()

    # First sequence
    sequence = dsl.Sequence()
    sequence.extend([dsl.East, dsl.East, dsl.East, dsl.South, dsl.South])
    actual_sequence = dsl.convert_sequence_to_actions(sequence, agent_start_dir=dsl.NORTH)
    expected_str_sequence = "Tr Walk Walk Walk Tr Walk Walk"
    actual_str_sequence = str(actual_sequence)
    assert actual_str_sequence == expected_str_sequence, \
        "test_convert_sequence_to_actions FAILED for \nexpected %s \nactual: %s" % (expected_str_sequence,
                                                                                    actual_str_sequence)

    # Second sequence
    sequence = dsl.Sequence()
    sequence.extend([dsl.East, dsl.South, dsl.East, dsl.South, dsl.South])
    actual_sequence = dsl.convert_sequence_to_actions(sequence, agent_start_dir=dsl.WEST)
    expected_str_sequence = "Tl Tl Walk Tr Walk Tl Walk Tr Walk Walk"
    actual_str_sequence = str(actual_sequence)
    assert actual_str_sequence == expected_str_sequence, \
        "test_convert_sequence_to_actions FAILED for \nexpected %s \nactual: %s" % (
        expected_str_sequence, actual_str_sequence)

    # Third sequence
    sequence = dsl.Sequence()
    sequence.extend([dsl.South, dsl.South, dsl.East, dsl.South, dsl.South])
    actual_sequence = dsl.convert_sequence_to_actions(sequence, agent_start_dir=dsl.WEST)
    expected_str_sequence = "Tl Walk Walk Tl Walk Tr Walk Walk"
    actual_str_sequence = str(actual_sequence)
    assert actual_str_sequence == expected_str_sequence, \
        "test_convert_sequence_to_actions FAILED for \nexpected %s \nactual: %s" % (
            expected_str_sequence, actual_str_sequence)

    # Fourth sequence
    sequence = dsl.Sequence()
    sequence.extend([dsl.North, dsl. South, dsl.North])
    actual_sequence = dsl.convert_sequence_to_actions(sequence, agent_start_dir=dsl.SOUTH)
    expected_str_sequence = "Tl Tl Walk Tl Tl Walk Tl Tl Walk"
    actual_str_sequence = str(actual_sequence)
    assert actual_str_sequence == expected_str_sequence, \
        "test_convert_sequence_to_actions FAILED for \nexpected %s \nactual: %s" % (
            expected_str_sequence, actual_str_sequence)

    # Fifth sequence
    sequence = dsl.Sequence()
    sequence.extend([dsl.West, dsl.North, dsl.North, dsl.West])
    actual_sequence = dsl.convert_sequence_to_actions(sequence, agent_start_dir=dsl.SOUTH)
    expected_str_sequence = "Tr Walk Tr Walk Walk Tl Walk"
    actual_str_sequence = str(actual_sequence)
    assert actual_str_sequence == expected_str_sequence, \
        "test_convert_sequence_to_actions FAILED for \nexpected %s \nactual: %s" % (
            expected_str_sequence, actual_str_sequence)

    # Sixth sequence
    sequence = dsl.Sequence()
    sequence.extend([dsl.West, dsl.West, dsl.West])
    actual_sequence = dsl.convert_sequence_to_actions(sequence, agent_start_dir=dsl.WEST)
    expected_str_sequence = "Walk Walk Walk"
    actual_str_sequence = str(actual_sequence)
    assert actual_str_sequence == expected_str_sequence, \
        "test_convert_sequence_to_actions FAILED for \nexpected %s \nactual: %s" % (
            expected_str_sequence, actual_str_sequence)

    end = time.time()
    logger.info("test_convert_sequence_to_actions PASSED in {} seconds".format(end - start))


def test_dsl_gscan(dataset):
    start = time.time()

    original_start_pos = TEST_SITUATION_1.agent_pos
    original_start_dir = TEST_SITUATION_1.agent_direction

    start_positions = [dsl.Position(row=5, column=3), dsl.Position(row=0, column=3),
                       dsl.Position(row=7, column=7), dsl.Position(row=13, column=7),
                       dsl.Position(row=14, column=13), dsl.Position(row=0, column=0),
                       dsl.Position(row=5, column=2), dsl.Position(row=14, column=8),
                       dsl.Position(row=9, column=0), dsl.Position(row=0, column=1)]
    start_directions = [dsl.WEST, dsl.WEST, dsl.NORTH, dsl.SOUTH, dsl.EAST, dsl.WEST,
                        dsl.EAST, dsl.EAST, dsl.NORTH, dsl.SOUTH]
    end_positions = [dsl.Position(row=0, column=0), dsl.Position(row=3, column=0),
                     dsl.Position(row=13, column=7), dsl.Position(row=13, column=2),
                     dsl.Position(row=0, column=2), dsl.Position(row=0, column=0),
                     dsl.Position(row=7, column=2), dsl.Position(row=11, column=10),
                     dsl.Position(row=0, column=3), dsl.Position(row=13, column=12)]
    start_positions.extend(end_positions)
    extra_positions = end_positions.copy()
    extra_directions = start_directions.copy()
    random.shuffle(extra_directions)
    random.shuffle(extra_positions)
    start_positions.extend(extra_positions)
    end_positions.extend(start_positions)
    random.shuffle(extra_positions)
    end_positions.extend(extra_positions)
    start_directions.extend(reversed(start_directions))
    start_directions.extend(extra_directions)

    gscan = dsl.Gscan()
    spinning_adverbs = [gscan.spin, gscan.cautiously, gscan.hesitantly]
    spinning_adverb_texts = ["spin", "cautious", "hesitantly"]

    for i, (start_pos, agent_start_dir, end_pos) in enumerate(zip(start_positions,
                                                                  start_directions,
                                                                  end_positions)):
        TEST_SITUATION_1.agent_pos = start_pos
        TEST_SITUATION_1.agent_direction = agent_start_dir
        planned_sequence = dsl.simulate_planner(start_pos, end_pos)
        col_distance = abs(start_pos.column - end_pos.column)
        sample_recursion = random.randint(0, col_distance*2)
        if (i + 1) % 4 == 0:
            gscan.zigzag(planned_sequence, sample_recursion)
        target_action_sequence = dsl.convert_sequence_to_actions(planned_sequence, agent_start_dir)
        if i % 4 != 0:
            idx = i % 3
            spinning_adverbs[idx](target_action_sequence, recursion_depth=1)
        command = "Test"
        target_actions = target_action_sequence.get_gscan_actions()
        _, _, end_column, end_row = dataset.demonstrate_target_commands(command,
                                                                        TEST_SITUATION_1,
                                                                        target_actions)
        assert end_column == end_pos.column and end_row == end_pos.row, \
            "test_dsl_gscan FAILED for expected end pos ({},{}), actual ({},{})".format(
                end_pos.row, end_pos.column, end_row, end_column
            )

    end = time.time()
    logger.info("test_dsl_gscan PASSED in {} seconds".format(end - start))

    TEST_SITUATION_1.agent_pos = original_start_pos
    TEST_SITUATION_1.agent_direction = original_start_dir


def test_get_num_push_pull_actions(dataset):
    start = time.time()
    current_situation = dataset._world.get_current_situation()
    current_mission = dataset._world.mission

    dataset._world.clear_situation()
    dataset.initialize_world(TEST_SITUATION_1)

    # First test
    last_direction = dsl.EAST
    position = dsl.Position(column=0, row=0)
    expected_free_push = 14
    expected_free_pull = 0
    actual_free_push = dsl.get_num_push_actions(last_direction, position,
                                                dataset._world.grid)
    actual_free_pull = dsl.get_num_pull_actions(last_direction, position,
                                                dataset._world.grid)
    assert actual_free_pull == expected_free_pull, "Wrong number of pull actions, expected {}, actual {}".format(
        expected_free_pull, actual_free_pull)
    assert actual_free_push == expected_free_push, "Wrong number of push actions, expected {}, actual {}".format(
        expected_free_push, actual_free_push)

    # Second test
    last_direction = dsl.EAST
    position = dsl.Position(column=12, row=10)
    expected_free_push = 2
    expected_free_pull = 7
    actual_free_push = dsl.get_num_push_actions(last_direction, position,
                                                dataset._world.grid)
    actual_free_pull = dsl.get_num_pull_actions(last_direction, position,
                                                dataset._world.grid)
    assert actual_free_pull == expected_free_pull, "Wrong number of pull actions, expected {}, actual {}".format(
        expected_free_pull, actual_free_pull)
    assert actual_free_push == expected_free_push, "Wrong number of push actions, expected {}, actual {}".format(
        expected_free_push, actual_free_push)

    # Third test
    last_direction = dsl.SOUTH
    position = dsl.Position(column=12, row=10)
    expected_free_push = 4
    expected_free_pull = 6
    actual_free_push = dsl.get_num_push_actions(last_direction, position,
                                                dataset._world.grid)
    actual_free_pull = dsl.get_num_pull_actions(last_direction, position,
                                                dataset._world.grid)
    assert actual_free_pull == expected_free_pull, "Wrong number of pull actions, expected {}, actual {}".format(
        expected_free_pull, actual_free_pull)
    assert actual_free_push == expected_free_push, "Wrong number of push actions, expected {}, actual {}".format(
        expected_free_push, actual_free_push)

    # Fourth test
    last_direction = dsl.NORTH
    position = dsl.Position(column=12, row=10)
    expected_free_push = 6
    expected_free_pull = 4
    actual_free_push = dsl.get_num_push_actions(last_direction, position,
                                                dataset._world.grid)
    actual_free_pull = dsl.get_num_pull_actions(last_direction, position,
                                                dataset._world.grid)
    assert actual_free_pull == expected_free_pull, "Wrong number of pull actions, expected {}, actual {}".format(
        expected_free_pull, actual_free_pull)
    assert actual_free_push == expected_free_push, "Wrong number of push actions, expected {}, actual {}".format(
        expected_free_push, actual_free_push)

    # Fifth test
    last_direction = dsl.WEST
    position = dsl.Position(column=6, row=9)
    expected_free_push = 6
    expected_free_pull = 8
    actual_free_push = dsl.get_num_push_actions(last_direction, position,
                                                dataset._world.grid)
    actual_free_pull = dsl.get_num_pull_actions(last_direction, position,
                                                dataset._world.grid)
    assert actual_free_pull == expected_free_pull, "Wrong number of pull actions, expected {}, actual {}".format(
        expected_free_pull, actual_free_pull)
    assert actual_free_push == expected_free_push, "Wrong number of push actions, expected {}, actual {}".format(
        expected_free_push, actual_free_push)

    # Sixth test
    last_direction = dsl.NORTH
    position = dsl.Position(column=12, row=4)
    expected_free_push = 0
    expected_free_pull = 10
    actual_free_push = dsl.get_num_push_actions(last_direction, position,
                                                dataset._world.grid)
    actual_free_pull = dsl.get_num_pull_actions(last_direction, position,
                                                dataset._world.grid)
    assert actual_free_pull == expected_free_pull, "Wrong number of pull actions, expected {}, actual {}".format(
        expected_free_pull, actual_free_pull)
    assert actual_free_push == expected_free_push, "Wrong number of push actions, expected {}, actual {}".format(
        expected_free_push, actual_free_push)

    dataset.initialize_world(current_situation, mission=current_mission)
    end = time.time()
    logger.info("test_get_num_push_pull_actions PASSED in {} seconds".format(end - start))


def test_gscan_examples(dataset):
    start = time.time()
    current_situation = dataset._world.get_current_situation()
    current_mission = dataset._world.mission
    num_generate_examples = 10000
    num_test_examples = 1000
    # TODO: how to make sure push/pull/walk and all adverbs are tested
    logger.info("test_gscan_examples loading {} and testing {} examples. Might take a while.".format(num_generate_examples,
                                                                                                     num_test_examples))
    dataset.get_data_pairs(max_examples=num_generate_examples)
    indices = [i for i in range(len(dataset._data_pairs["train"]))]
    random.shuffle(indices)
    gscan = dsl.Gscan()
    verb_adverb = {}
    num_examples_tested = 0
    for example_idx in indices[:num_test_examples]:
        example = dataset._data_pairs["train"][example_idx]
        if (num_examples_tested + 1) % 500 == 0:
            logger.info("Tested {} of {} examples succesfully".format(
                num_examples_tested, num_test_examples))
        verb_in_command = example["verb_in_command"]
        manner = example["manner"]
        if verb_in_command not in verb_adverb.keys():
            verb_adverb[verb_in_command] = {}
        if not manner:
            manner_str = "no_manner"
        else:
            manner_str = manner
        if manner_str not in verb_adverb[verb_in_command].keys():
            verb_adverb[verb_in_command][manner_str] = 0
        verb_adverb[verb_in_command][manner_str] += 1
        size = int(example["situation"]["target_object"]["object"]["size"])
        heavy = False
        if size in dataset._world._object_vocabulary._heavy_weights.keys():
            heavy = True
        target_position = example["situation"]["target_object"]["position"]
        target_position = dsl.Position(column=int(target_position["column"]),
                                       row=int(target_position["row"]))
        start_position = example["situation"]["agent_position"]
        start_position = dsl.Position(column=int(start_position["column"]),
                                      row=int(start_position["row"]))
        situation = Situation.from_representation(example["situation"])
        dataset.initialize_world(situation, example["command"])
        grid = dataset._world.grid
        start_direction = example["situation"]["agent_direction"]
        expected_target_commands = example["target_commands"]
        actual_target_commands = gscan.parse_gscan_example(verb_in_command,
                                                           start_position,
                                                           start_direction,
                                                           target_position,
                                                           manner,
                                                           grid,
                                                           heavy)
        actual_target_commands = ",".join(actual_target_commands)
        assert expected_target_commands == actual_target_commands, \
            "test_gscan_examples FAILED with verb {}, manner {}," \
            "expected commands: {}, actual commands: {}".format(verb_in_command, manner,
                                                                expected_target_commands,
                                                                actual_target_commands)
        num_examples_tested += 1

    dataset.initialize_world(current_situation, mission=current_mission)
    end = time.time()
    logger.info("test_gscan_examples PASSED in {} seconds".format(end - start))


def test_adverb_sampling(dataset):
    start = time.time()
    current_situation = dataset._world.get_current_situation()
    current_mission = dataset._world.mission

    world = dsl.AdverbWorld(seed=1)
    movement_rewrite_adverbs = world._all_adverbs["movement_rewrite"]
    movement_adverbs = world._all_adverbs["movement"]
    nonmovement_direction_adverbs = world._all_adverbs["nonmovement_direction"]
    nonmovement_first_person_adverbs = world._all_adverbs[
        "nonmovement_first_person"]
    print("movement_rewrite {}".format(len(movement_rewrite_adverbs)))
    print("movement_adverbs {}".format(len(movement_adverbs)))
    print("nonmovement_direction_adverbs {}".format(len(nonmovement_direction_adverbs)))
    print("nonmovement_first_person_adverbs {}".format(len(nonmovement_first_person_adverbs)))
    all_adverbs = {"nonmovement_direction": nonmovement_direction_adverbs,
                   "movement_adverbs": movement_adverbs,
                   "movement_rewrite": movement_rewrite_adverbs,
                   "nonmovement_first_person": nonmovement_first_person_adverbs}
    recursions_on_sequence = [lambda cols, rows: 1,
                              lambda cols, rows: cols - 1,
                              lambda cols, rows: rows - 1,
                              lambda cols, rows: cols + rows - 1]
    other_recursions = [lambda cols, rows: 1,
                        lambda cols, rows: 2,
                        lambda cols, rows: 3,
                        lambda cols, rows: 4]
    zero_recursions = [lambda cols, rows: 1,
                       lambda cols, rows: 1,
                       lambda cols, rows: 1,
                       lambda cols, rows: 1]
    all_recursions_system = {"nonmovement_direction": other_recursions,
                             "movement_adverbs": other_recursions,
                             "movement_rewrite": zero_recursions,
                             "nonmovement_first_person": other_recursions}
    all_recursions_sequence = {"nonmovement_direction": zero_recursions,
                               "movement_adverbs": zero_recursions,
                               "movement_rewrite": recursions_on_sequence,
                               "nonmovement_first_person": zero_recursions}
    adverb_type = ["nonmovement_first_person",
                   "nonmovement_direction",
                   "movement_adverbs",
                   "movement_rewrite"]
    for type in adverb_type:
        adverb_list = all_adverbs[type]
        recursions_system = all_recursions_system[type]
        recursions_sequence = all_recursions_sequence[type]
        adverbs_to_test = random.sample(adverb_list, 10)
        # adverbs_to_test = adverb_list[:5]
        to_visualize = random.sample(range(5), 5)
        for i, adverb in enumerate(adverbs_to_test):
            if i in to_visualize:
                parent_save_dir = "{}_{}".format(type, i)
                save_dir = os.path.join(TEST_DIRECTORY, parent_save_dir)
                if not os.path.exists(save_dir):
                    os.mkdir(save_dir)
                infile = open(os.path.join(save_dir, "l_system.txt"), "w")
                infile.write(str(adverb))
            for recursion_lambda_system, recursion_lambda_sequence in zip(recursions_system, recursions_sequence):
                for _ in range(1):
                    start_direction = random.randint(0, 3)
                    start_row = random.randint(0, dataset._world.grid_size - 1)
                    start_col = random.randint(0, dataset._world.grid_size - 1)
                    end_row = random.randint(0, dataset._world.grid_size - 1)
                    end_col = random.randint(0, dataset._world.grid_size - 1)
                    num_rows = abs(end_row - start_row)
                    num_cols = abs(end_col - start_col)
                    recursion_system = recursion_lambda_system(num_cols, num_rows)
                    recursion_sequence = recursion_lambda_sequence(num_cols, num_rows)
                    if i in to_visualize:
                        infile.write("num_cols %d num_rows %d recursion system %d\n" % (num_cols,
                                                                                 num_rows,
                                                                                 recursion_system))
                        infile.write("num_cols %d num_rows %d recursion sequence %d\n" % (num_cols,
                                                                                          num_rows,
                                                                                          recursion_sequence))
                    start_position = dsl.Position(row=start_row, column=start_col)
                    end_position = dsl.Position(row=end_row, column=end_col)
                    current_situation = Situation(grid_size=dataset._world.grid_size,
                                                  agent_position=start_position,
                                                  agent_direction=INT_TO_DIR[start_direction],
                                                  target_object=PositionedObject(
                                                      object=Object(size=2, color='red', shape='circle'),
                                                      position=end_position,
                                                      vector=np.array([1, 0, 1])),
                                                  placed_objects=[PositionedObject(
                                                      object=Object(size=2, color='red', shape='circle'),
                                                      position=end_position,
                                                      vector=np.array([1, 0, 1]))], carrying=None)
                    # dataset.initialize_world(current_situation, mission=current_mission)
                    sequence, is_rejected = world.generate_example(start_position,
                                                      start_direction,
                                                      end_position,
                                                      adverb,
                                                      recursion_depth_system=recursion_system,
                                                      recursion_depth_sequence=recursion_sequence,
                                                      grid_size=dataset._world.grid_size,
                                                      type_adverb=type)
                    if is_rejected:
                        logger.info("Rejected L-system.")
                        logger.info(str(adverb))
                        if i in to_visualize:
                            infile.write("Rejected L-system.\n")
                        continue
                    (commands, demonstration,
                     actual_end_col, actual_end_row) = dataset.demonstrate_target_commands(
                        "", current_situation, target_commands=sequence)
                    if i in to_visualize:
                        mission = ' '.join(["\n      Target:"] + commands)

                        save_dir_prediction = dataset.visualize_command(
                            current_situation, "command", demonstration, mission=mission,
                            parent_save_dir=parent_save_dir)
                    assert end_col == actual_end_col, "Wrong end col for adverb_type {}".format(type)
                    assert end_row == actual_end_row, "Wrong end row for adverb_type {}".format(type)
            if i in to_visualize:
                infile.close()
    dataset.initialize_world(current_situation, mission=current_mission)
    end = time.time()
    logger.info("test_adverb_sampling PASSED in {} seconds".format(end - start))


def test_remove_out_of_grid(dataset):
    start = time.time()
    init_current_situation = dataset._world.get_current_situation()
    init_current_mission = dataset._world.mission

    meta_grammar = dsl.MetaGrammar()
    world = dsl.AdverbWorld(seed=1)
    l_system = dsl.LSystem()
    l_system.add_rule(meta_grammar.get_rule(lhs_str="West",
                                            rhs_str="North West South"))
    l_system.add_rule(meta_grammar.get_rule(lhs_str="East",
                                            rhs_str="North East South"))
    l_system.add_rule(meta_grammar.get_rule(lhs_str="North",
                                            rhs_str="East North West"))
    l_system.add_rule(meta_grammar.get_rule(lhs_str="South",
                                            rhs_str="East South West"))
    l_system.finish_l_system()

    start_direction = 0
    start_position = dsl.Position(row=0, column=0)
    expected_end_row = 5
    expected_end_col = 5
    end_position = dsl.Position(row=expected_end_row, column=expected_end_col)
    sequence, rejected = world.generate_example(start_position,
                                                start_direction,
                                                end_position,
                                                l_system,
                                                recursion_depth_system=0,
                                                recursion_depth_sequence=1,
                                                grid_size=6,
                                                type_adverb="movement")
    sequence_two, _ = world.generate_example(start_position,
                                             start_direction,
                                             end_position,
                                             l_system,
                                             recursion_depth_system=1,
                                             recursion_depth_sequence=1,
                                             grid_size=6,
                                             type_adverb="movement")
    sequence_three, _ = world.generate_example(start_position,
                                               start_direction,
                                               end_position,
                                               l_system,
                                               recursion_depth_system=1,
                                               recursion_depth_sequence=2,
                                               grid_size=6,
                                               type_adverb="movement")
    current_situation = Situation(grid_size=6,
                                  agent_position=start_position,
                                  agent_direction=INT_TO_DIR[start_direction],
                                  target_object=PositionedObject(
                                      object=Object(size=2, color='red', shape='circle'),
                                      position=end_position,
                                      vector=np.array([1, 0, 1])),
                                  placed_objects=[PositionedObject(
                                      object=Object(size=2, color='red', shape='circle'),
                                      position=end_position,
                                      vector=np.array([1, 0, 1])),
                                      PositionedObject(
                                          object=Object(size=4, color='green', shape='circle'),
                                          position=Position(row=3, column=3),
                                          vector=np.array([0, 1, 0]))], carrying=None)

    (commands, demonstration,
     end_col, end_row) = dataset.demonstrate_target_commands(
        "", current_situation, target_commands=sequence)
    assert expected_end_col == end_col, "test_remove_out_of_grid FAILED. Wrong end col"
    assert expected_end_row == end_row, "test_remove_out_of_grid FAILED. Wrong end row"
    (commands, demonstration,
     end_col, end_row) = dataset.demonstrate_target_commands(
        "", current_situation, target_commands=sequence_two)
    assert expected_end_col == end_col, "test_remove_out_of_grid FAILED. Wrong end col"
    assert expected_end_row == end_row, "test_remove_out_of_grid FAILED. Wrong end row"
    (commands, demonstration,
     end_col, end_row) = dataset.demonstrate_target_commands(
        "", current_situation, target_commands=sequence_three)
    assert expected_end_col == end_col, "test_remove_out_of_grid FAILED. Wrong end col"
    assert expected_end_row == end_row, "test_remove_out_of_grid FAILED. Wrong end row"

    dataset.initialize_world(init_current_situation, mission=init_current_mission)
    end = time.time()
    logger.info("test_remove_out_of_grid PASSED in {} seconds".format(end - start))


def test_generate_adverb_challenge():
    start = time.time()

    adverb_world = dsl.AdverbWorld(seed=1)
    data = adverb_world.generate_adverb_challenge(num_training_adverbs=20,
                                                  num_train_examples_per_train_adverb=50,
                                                  num_testing_adverbs=5,
                                                  num_train_examples_per_test_adverb=50,
                                                  grid_size=6,
                                                  save_directory=TEST_DIRECTORY)

    end = time.time()
    logger.info("test_generate_adverb_challenge PASSED in {} seconds".format(end - start))


def run_all_tests():
    # test_save_and_load_dataset(TEST_DATASET)
    # test_save_and_load_dataset(TEST_DATASET_NONCE)
    # test_save_and_load_dataset_nonce()
    # test_derivation_from_rules(TEST_DATASET)
    # test_derivation_from_rules(TEST_DATASET_NONCE)
    # test_derivation_from_string(TEST_DATASET)
    # test_derivation_from_string(TEST_DATASET_NONCE)
    # test_demonstrate_target_commands_one(TEST_DATASET)
    # test_demonstrate_target_commands_one(TEST_DATASET_NONCE)
    # test_demonstrate_target_commands_two(TEST_DATASET)
    # test_demonstrate_target_commands_two(TEST_DATASET_NONCE)
    # test_demonstrate_target_commands_three(TEST_DATASET)
    # test_demonstrate_target_commands_three(TEST_DATASET_NONCE)
    # test_demonstrate_command_one(TEST_DATASET)
    # test_demonstrate_command_one(TEST_DATASET_NONCE)
    # test_demonstrate_command_two(TEST_DATASET)
    # test_demonstrate_command_two(TEST_DATASET_NONCE)
    # test_demonstrate_command_three(TEST_DATASET)
    # test_demonstrate_command_three(TEST_DATASET_NONCE)
    # test_demonstrate_command_four(TEST_DATASET)
    # test_demonstrate_command_four(TEST_DATASET_NONCE)
    # test_demonstrate_command_five(TEST_DATASET)
    # test_demonstrate_command_five(TEST_DATASET_NONCE)
    # test_demonstrate_command_six(TEST_DATASET)
    # test_demonstrate_command_six(TEST_DATASET_NONCE)
    # test_find_referred_target_one(TEST_DATASET)
    # test_find_referred_target_one(TEST_DATASET_NONCE)
    # test_find_referred_target_two(TEST_DATASET)
    # test_find_referred_target_two(TEST_DATASET_NONCE)
    # test_generate_possible_targets_one(TEST_DATASET)
    # test_generate_possible_targets_one(TEST_DATASET_NONCE)
    # test_generate_possible_targets_two(TEST_DATASET)
    # test_generate_possible_targets_two(TEST_DATASET_NONCE)
    # test_generate_situations_one(TEST_DATASET)
    # test_generate_situations_one(TEST_DATASET_NONCE)
    # test_generate_situations_two(TEST_DATASET)
    # test_generate_situations_two(TEST_DATASET_NONCE)
    # test_generate_situations_three(TEST_DATASET)
    # test_generate_situations_three(TEST_DATASET_NONCE)
    # test_situation_representation_eq()
    # test_example_representation_eq(TEST_DATASET)
    # test_example_representation_eq(TEST_DATASET_NONCE)
    # test_example_representation(TEST_DATASET)
    # test_example_representation(TEST_DATASET_NONCE)
    # test_initialize_world(TEST_DATASET)
    # test_initialize_world(TEST_DATASET_NONCE)
    # test_image_representation_situations(TEST_DATASET)
    # test_image_representation_situations(TEST_DATASET_NONCE)
    # test_encode_situation(TEST_DATASET)
    # test_encode_situation(TEST_DATASET_NONCE)
    # test_k_shot_generalization(TEST_DATASET)
    # test_k_shot_generalization(TEST_DATASET_NONCE)
    # test_generate_manners()
    # test_apply_while_spinning()
    # test_apply_cautiously()
    # test_apply_hesitantly()
    # test_apply_while_zigzagging()
    # test_convert_sequence_to_actions()
    # test_dsl_gscan(TEST_DATASET)
    # test_get_num_push_pull_actions(TEST_DATASET)
    # test_gscan_examples(TEST_DATASET_2)
    # test_adverb_sampling(TEST_DATASET_2)
    # test_remove_out_of_grid(TEST_DATASET_2)
    test_generate_adverb_challenge()
    # shutil.rmtree(TEST_DIRECTORY)
