# Grounded SCAN

This repository contains the code for generating the grounded SCAN benchmark.
Grounded SCAN poses a simple task, where an agent must execute action sequences based on a synthetic language instruction.


The agent is presented with a simple grid world containing a collection of
objects, each of which is associated with a vector of features. The agent is
evaluated on its ability to follow one or more instructions in this environment.
Some instructions require interaction with particular kinds of objects.

## Data for Paper

All data used in the paper can be found in compressed folders in `data`.
To see how to read this data for a computational model and train models on it please refer t <TODO INSERT REPO>.

## Extra Documentation

Find glossary in `documentation/glossary.md`.

## Getting Started

Make a virtualenvironment that uses Python 3.7 or higher:

```>> virtualenv --python=/usr/bin/python3.7 <path/to/virtualenv>```

Activate the environment and install the requirements with a package manager:

```>> { source <path/to/virtualenv>/bin/activate; python3.7 -m pip install -r requirements; }```

Make sure all tests run correctly:

```(virtualenv) >> python3.7 -m GroundedScan --mode=test```

## Using the Repository


### Generating Data
To generate data for the compositional generalization splits:

    (virtualenv) >> python3.7 -m GroundedScan --mode=generate --output_directory=compositional_splits --type_grammar=adverb --split=generalization

in a directory containing the folder `GroundedScan`. This will generate a dataset in the folder `compositional_splits`.
The dataset will be in `compositional_splits/dataset.txt` and for each split there will be an associated `compositional_splits/<split>_stats.txt`
and plots visualizing the statistics of the data in that particular split (e.g. the number of examples totally, broken down per grid location, per referred target, etc.).
For an example dataset see the folder `data/compositional_splits.zip`, containing the data used for the paper.

#### Important Parameters

* `grid_size` specifies the number of columns and rows in the world. If set higher, more examples will be generated and the maximum target length will be higher.
* `num_resampling` this determines how often you want to resample an example with the same specifications in terms of input instruction,
target referrent, and relative position of the agent versus the target. If set to higher than 1, the code samples new random agent and target locations that still satisfy
the constraints. Apart from setting a higher grid size, this parameter can be used to generate more data.
* `split` determines for which splits we are generating data. Can be set to uniform (i.e. no systematic difference between training and test), 
generalization (i.e. the compositional generalization splits), or target_lengths (i.e. generalizing to larger target lengths). For this
latter split make sure to also set `cut_off_target_length`
* `visualize_per_template` is an integer determining how many gifs to make per template example (see glossary.md for explanation of template).
* `type_grammar` if set to `adverb` just gives the fully implemented grammar from the paper, but you can also set it to `normal` to exclude
adverbs, or to `simple_intrans` and `simple_trans` to use only intransitive or transitive verbs respectively.

### Error Analysis
To do an error analysis over a predictions file generated by a model trained on the data, run:

    (virtualenv) >> python3.7 -m GroundedScan --mode=error_analysis --output_directory=error_analysis --load_dataset_from=compositional_splits/dataset.txt  --predicted_commands_file=predict.json

The predicted commands file must be a file generated by the predict function in the provided model code (TODO: refer to model code). For an example file with predictions see 
the file at `experiment_logs/adverb_k_1_run_3/dev_predict_adverb_k_1_run_3.json`. The error analysis will generate plots in
the output directory specified, as well as a .txt and .csv file summarizing all the results for own anaylsis.
The .csv outputs were used in the paper.

###  Execute Commands
To make gifs of predictions use the mode `execute_commands`:

    (virtualenv) >> python3.7 -m GroundedScan --mode=execute_commands --load_dataset_from=compositional_splits/dataset.txt --output_directory=visualized_examples
    --predicted_commands_file=predict.json
 
This will visualize the predictions in `predict.json` and visualize the execution in a gif in the specified output directory.

## Implementation Details

We
construct a `Grammar` that produces transitive and intransitive sentences (with
a depth-limited recursive operation for introducing adjectives and adverbs). Sentences are of the form

    ((Adverb* Verb (Adjective* Noun)?)+

It is possible to sample from a `Grammar` to obtain paired _sentences_
(sequences of tokens) and _meanings_ (hacky neo-Davidsonian logical forms). For
example, the sentence

    push a big square

gets associated with the logical form

    lambda $v1. exists $n. push($v1) and patient($v1, $n) and big($n) and
    square($n)
    
represented in the code by

    (push x0:verb) ^ (big x2:noun) ^ (square x2:noun) ^ (patient x0:verb x2:noun)

In order to pair instructions with demonstrations, we need to map from meanings
to sequences of low-level environment actions. 

## Semantics

The example logical form depicted above specifies one action: a square-pushing
action. It must be performed in a grounded context. In
this library, computations associated with grounding live in `world.py`. The
grounded context for a specific instruction is represented by a `Situation`.
`Situations` are generated from `World`s. A situation is what is used for one example to specify all the important information
to generate a visualization as well as input grid world for a computational model. The function to visualize a current situation in
an RGB image is `World.get_current_situation_image()` in `world.py` and to get the simplified input to a model (see paper Section 3 (World Model) for explanation)
run the function `World.get_current_situation_grid_repr()`. This is all done automatically when generating the data.

Important semantics:

* `walk`: means walking somewhere with action command `walk`, and for navigation `turn left` and `turn right`.
* `push`: means pushing an object to the front of the agent until it hits a wall or another object, to move it 1 grid cell takes 1 `push` action if the object is of size 1 or 2, and 2 push actions `push push` if it is of size 3 or 4.
* `pull`: means pulling an object to the back of the agent until it hits a wall or another object, with `pull`-actions, again needing 1 action for light objects (size 1 and 2), and 2 for heavy ones (size 3 and 4).
* `while spinning`: generate a sequence of `turn left turn left turn left turn left` (i.e. one spin) every time before executing the actions to move a grid cell.
* `cautiously`: generate a sequence of `turn right turn left turn left turn right` (i.e. look to the right and left) each time before you move over a grid line (note the distinction with when to execute the sequence from while spinning, see examples at bottom of this page for clarification).
* `hesitantly`: generate a action command to `stay` every time after moving a grid cell.
* `while zigzagging`: instead of walking all the way horizontally first and then vertically, alternate between horizontal and vertical until in line with the target, from then go straight.


## Code-Flow Diagram
For a schematic depiction of how exactly the code is setup see the following image:
![Code flow](https://raw.githubusercontent.com/LauraRuis/groundedSCAN/master/documentation/Code-flow%20diagram.png)


## Generalization

After instructions have been generated, we construct various held-out sets
designed to test different notions of compositional generalization. This is done in the class method `GroundedScan.assign_splits()`
in `GroundedScan/dataset.py`

**object properties**
* visual: all examples where red squares are the target object.
* visual_easier: all examples where yellow squares are referred to with a color and a shape at least.

**situational**:
* situational_1: all examples where the target object is to the South-West of the agent.
* situational_2: all examples where the target object is a circle of size 2 and is being referred to with the small modifier.

**contextual**
* contextual: all examples where the agent needs to push a square of size 3.

**adverb**:
* adverb_1: all examples with the adverb 'cautiously', of which we randomly select k to go in the training set.
* adverb_2: all examples with the adverb 'while spinning' and the verb 'pull'.

**target_lengths**
* target_lengths: all examples with a target length above `--cut_off_target_length`

## Example gridworld command and demonstration

### Compositional Splits Examples
#### Random Example
![Random Example](https://raw.githubusercontent.com/LauraRuis/groundedSCAN/master/documentation/movie.gif)

#### Visual (Red Squares)
![Red Squares](https://raw.githubusercontent.com/LauraRuis/groundedSCAN/master/documentation/examples_per_split/Red%20Squares/movie.gif)

#### Visual easier (Yellow Squares)
![Yellow Squares](https://raw.githubusercontent.com/LauraRuis/groundedSCAN/master/documentation/examples_per_split/Yellow%20Squares/movie.gif)

#### Situational 1 (Novel Direction)
![Novel Direction](https://raw.githubusercontent.com/LauraRuis/groundedSCAN/master/documentation/examples_per_split/Novel%20Direction/movie.gif)

#### Situational 2 (Relativity)
![Relativity Size](https://raw.githubusercontent.com/LauraRuis/groundedSCAN/master/documentation/examples_per_split/Relativity%20Size/movie.gif)

#### Contextual (Class Inference)
![Latent Class Inference](https://raw.githubusercontent.com/LauraRuis/groundedSCAN/master/documentation/examples_per_split/Latent%20Class%20Inference/movie.gif)

#### Adverb 1 (Adverb cautiously k-shot)
![Adverb 1](https://raw.githubusercontent.com/LauraRuis/groundedSCAN/master/documentation/examples_per_split/Adverb%20k-shot/movie.gif)

#### Adverb 2 (Adverb while spinning to verb pull)
![Adverb 2](https://raw.githubusercontent.com/LauraRuis/groundedSCAN/master/documentation/examples_per_split/Adverb%20to%20Verb/movie.gif)

### Adverb Examples

#### While Spinning
![while spinning](https://raw.githubusercontent.com/LauraRuis/groundedSCAN/master/documentation/examples_per_adverb/While%20spinning/movie.gif)

#### Cautiously
![Cautiously](https://raw.githubusercontent.com/LauraRuis/groundedSCAN/master/documentation/examples_per_adverb/Cautiously/movie.gif)

#### Hesitantly
![Hesitantly](https://raw.githubusercontent.com/LauraRuis/groundedSCAN/master/documentation/examples_per_adverb/Hesitantly/movie.gif)

#### While Zigzagging
![while zigzagging](https://raw.githubusercontent.com/LauraRuis/groundedSCAN/master/documentation/examples_per_adverb/While%20zigzagging/movie.gif)
