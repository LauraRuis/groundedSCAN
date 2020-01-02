# TODO: splits
# TODO: implement generate_all_situations for conjuncations (i.e. with multiple targets)
# TODO: make target_commands an enum like Actions in minigrid
# TODO: what to do about pushing something that's on the border (currently just not pushed, doesn't make sense)
# TODO: change plots and error analysis in presentation
# TODO: train model that works on different size representation in grid
from GroundedScan.dataset import GroundedScan
from GroundedScan.dataset_test import run_all_tests

import argparse
import os
import logging

FORMAT = '%(asctime)-15s %(message)s'
logging.getLogger("PyQt5").disabled = True
logging.getLogger('matplotlib.font_manager').disabled = True
logging.basicConfig(format=FORMAT, level=logging.DEBUG, datefmt='%Y-%m-%d %H:%M')
logger = logging.getLogger("GroundedScan")
logging.getLogger("PyQt5").setLevel(logging.WARNING)


def main():
    parser = argparse.ArgumentParser(description="Grounded SCAN")

    # General arguments.
    parser.add_argument('--mode', type=str, default='execute_commands',
                        help='Generate (mode=generate) data, run tests (mode=test) or execute commands from a file'
                             '(mode=execute_commands).')
    parser.add_argument('--load_dataset_from', type=str, default='', help='Path to file with dataset.')
    parser.add_argument('--output_directory', type=str, default='output', help='Path to a folder in which '
                                                                               'all outputs should be '
                                                                               'stored.')
    parser.add_argument('--predicted_commands_file', type=str, default='predict.json',
                        help='Path to a file with predictions.')
    parser.add_argument('--save_dataset_as', type=str, default='dataset.txt', help='Filename to save dataset in.')
    parser.add_argument("--count_equivalent_examples", dest="count_equivalent_examples", default=False,
                        action="store_true")
    parser.add_argument("--only_save_errors", dest="only_save_errors", default=False,
                        action="store_true")

    # Dataset arguments.
    parser.add_argument('--max_examples', type=int, default=None, help="Max. num. examples to generate.")
    parser.add_argument('--split', type=str, default='uniform', choices=['uniform', 'generalization'])
    parser.add_argument('--k_shot_generalization', type=int, default=0)
    parser.add_argument('--num_resampling', type=int, default=10, help='Number of time to resample a semantically '
                                                                       'equivalent situation (which will likely result'
                                                                       ' in different situations in terms of object '
                                                                       'locations).')
    parser.add_argument('--visualize_per_template', type=int, default=0, help='How many visualization to generate per '
                                                                              'command template.')
    parser.add_argument('--visualize_per_split', type=int, default=0, help='How many visualization to generate per '
                                                                           'test split.')
    parser.add_argument('--train_percentage', type=float, default=.8,
                        help='Percentage of examples to put in the training set (rest is test set).')

    # World arguments.
    parser.add_argument('--grid_size', type=int, default=6, help='Number of rows (and columns) in the grid world.')
    parser.add_argument('--min_other_objects', type=int, default=0, help='Minimum amount of objects to put in the grid '
                                                                         'world.')
    parser.add_argument('--max_objects', type=int, default=2, help='Maximum amount of objects to put in the grid '
                                                                   'world.')
    parser.add_argument('--sample_vocabulary', type=str, default='default', choices=['default', 'sample'])
    parser.add_argument('--min_object_size', type=int, default=1, help='Smallest object size.')
    parser.add_argument('--max_object_size', type=int, default=4, help='Biggest object size.')
    parser.add_argument('--other_objects_sample_percentage', type=float, default=.5,
                        help='Percentage of possible objects distinct from the target to place in the world.')

    # Grammar and Vocabulary arguments
    parser.add_argument('--type_grammar', type=str, default='normal', choices=['simple_intrans', 'simple_trans',
                                                                               'normal', 'adverb', 'full'])
    parser.add_argument('--intransitive_verbs', type=str, default='walk', help='Comma-separated list of '
                                                                               'intransitive verbs.')
    parser.add_argument('--transitive_verbs', type=str, default='push', help='Comma-separated list of '
                                                                             'transitive verbs.')
    parser.add_argument('--adverbs', type=str,
                        default='hesitantly,slowly,while spinning,quickly,while zigzagging,cautiously',
                        help='Comma-separated list of adverbs.')
    parser.add_argument('--nouns', type=str, default='circle,square,cylinder', help='Comma-separated list of nouns.')
    parser.add_argument('--color_adjectives', type=str, default='green,red,blue', help='Comma-separated list of '
                                                                                       'colors.')
    parser.add_argument('--size_adjectives', type=str, default='small,big', help='Comma-separated list of sizes.')
    parser.add_argument('--num_intransitive_verbs', type=int, default=1, help='number of intransitive verbs to sample.')
    parser.add_argument('--num_transitive_verbs', type=int, default=1, help='number of transitive verbs to sample.')
    parser.add_argument('--num_adverbs', type=int, default=6, help='number of adverbs to sample.')
    parser.add_argument('--num_nouns', type=int, default=3, help='number of nouns to sample.')
    parser.add_argument('--num_color_adjectives', type=int, default=2, help='number of color adjectives to sample.')
    parser.add_argument('--num_size_adjectives', type=int, default=2, help='number of size adjectives to sample.')

    flags = vars(parser.parse_args())

    if flags['mode'] == 'execute_commands' or flags['mode'] == 'error_analysis':
        assert os.path.exists(flags['load_dataset_from']), \
            "if mode={}, please specify data location in --load_dataset_from".format(flags['mode'])

    # Create directory for visualizations if it doesn't exist.
    if flags['output_directory']:
        visualization_path = os.path.join(os.getcwd(), flags['output_directory'])
        if not os.path.exists(visualization_path):
            os.mkdir(visualization_path)

    if flags['mode'] == 'generate':
        intransitive_verbs = flags["intransitive_verbs"].split(',') \
            if flags["sample_vocabulary"] != 'sample' else flags["num_intransitive_verbs"]
        transitive_verbs = flags["transitive_verbs"].split(',') \
            if flags["sample_vocabulary"] != 'sample' else flags["num_transitive_verbs"]
        adverbs = flags["adverbs"].split(',') if flags["sample_vocabulary"] != 'sample'else flags["num_adverbs"]
        nouns = flags["nouns"].split(',') if flags["sample_vocabulary"] != 'sample' else flags["num_nouns"]
        if flags["sample_vocabulary"] != 'sample':
            color_adjectives = flags["color_adjectives"].split(',') if flags["color_adjectives"] else []
            size_adjectives = flags["size_adjectives"].split(',') if flags["size_adjectives"] else []
        else:
            color_adjectives = flags["num_color_adjectives"]
            size_adjectives = flags["num_size_adjectives"]

        # Sample a vocabulary and a grammar with rules of form NT -> T and T -> {words from vocab}.
        grounded_scan = GroundedScan(
            intransitive_verbs=intransitive_verbs, transitive_verbs=transitive_verbs, adverbs=adverbs, nouns=nouns,
            color_adjectives=color_adjectives, size_adjectives=size_adjectives,
            min_object_size=flags["min_object_size"], max_object_size=flags["max_object_size"],
            sample_vocabulary=flags["sample_vocabulary"], save_directory=flags["output_directory"],
            grid_size=flags["grid_size"], type_grammar=flags["type_grammar"])

        # Generate all possible commands from the grammar
        grounded_scan.get_data_pairs(max_examples=flags["max_examples"],
                                     num_resampling=flags['num_resampling'],
                                     other_objects_sample_percentage=flags['other_objects_sample_percentage'],
                                     visualize_per_template=flags['visualize_per_template'],
                                     visualize_per_split=flags['visualize_per_split'],
                                     split_type=flags["split"],
                                     train_percentage=flags['train_percentage'],
                                     min_other_objects=flags['min_other_objects'],
                                     k_shot_generalization=flags['k_shot_generalization'])
        logger.info("Gathering dataset statistics...")
        grounded_scan.save_dataset_statistics(split="train")
        if flags["split"] == "uniform":
            grounded_scan.save_dataset_statistics(split="test")
        elif flags["split"] == "generalization":
            for split in ["test", "visual", "situational_1", "situational_2", "contextual"]:
                grounded_scan.save_dataset_statistics(split=split)
        dataset_path = grounded_scan.save_dataset(flags['save_dataset_as'])
        grounded_scan.visualize_data_examples()
        logger.info("Saved dataset to {}".format(dataset_path))
        if flags['count_equivalent_examples']:
            if flags["split"] == "uniform":
                splits_to_count = ["test"]
            elif flags["split"] == "generalization":
                splits_to_count = ["visual", "situational_1", "situational_2", "contextual"]
            else:
                raise ValueError("Unknown option for flag --split: {}".format(flags["split"]))
            for split in splits_to_count:
                logger.info("Equivalent examples in train and testset: {}".format(
                    grounded_scan.count_equivalent_examples("train", split)))
    elif flags['mode'] == 'execute_commands':
        assert os.path.exists(flags["predicted_commands_file"]), "Trying to execute commands from non-existing file: "\
                                                                 "{}".format(flags["predicted_commands_file"])
        grounded_scan = GroundedScan.load_dataset_from_file(flags["load_dataset_from"],
                                                            flags["output_directory"])
        grounded_scan.visualize_prediction(flags["predicted_commands_file"], only_save_errors=flags["only_save_errors"])
    elif flags['mode'] == 'test':
        logger.info("Running all tests..")
        run_all_tests()
    elif flags['mode'] == 'error_analysis':
        logger.info("Performing error analysis on file with predictions: {}".format(flags["predicted_commands_file"]))
        grounded_scan = GroundedScan.load_dataset_from_file(flags["load_dataset_from"],
                                                            flags["output_directory"])
        grounded_scan.error_analysis(predictions_file=flags["predicted_commands_file"],
                                     output_file=os.path.join(flags["output_directory"], "error_analysis.txt"))
        logger.info("Wrote data to path: {}.".format(os.path.join(flags["output_directory"], "error_analysis.txt")))
        logger.info("Saved plots in directory: {}.".format(flags["output_directory"]))
    else:
        raise ValueError("Unknown value for command-line argument 'mode'={}.".format(flags['mode']))


if __name__ == "__main__":
    main()
