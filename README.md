# Grounded SCAN

This repository implements a simple navigation task (in the spirit of the
original SCAN dataset) where agents must follow instructions in a _grounded_
context.

The agent is presented with a simple grid world containing a collection of
objects, each of which is associated with a vector of features. The agent is
evaluated on its ability to follow one or more instructions in this environment.
Some instructions require interaction with particular kinds of objects.

## Quickstart

Run

    ./main.py

in this directory. This will sample a random instruction grammar, and then
sample a sequence of (instruction, demonstration) pairs from this grammar.
Instructions will be bucketed according to various notions of compositional
generalization, and the number of instructions in each bucket printed.

Instruction grammars ("syntax"), instruction interpretation ("semantics"), and
generalization criteria are each discussed in more detail below.

**TODO:** currently, none of the generated data actually gets written to a file.
Implement some kind of serialization mechanism.

## Syntax

This library has been designed to support experiments in which structural
properties of the language presented to the learner (e.g. vocabulary size, word
frequency distribution, etc.) are varied. Thus, rather than defining a single
vocabulary and grammar over instructions, we randomly generate a new vocabulary
and associated grammar each time a dataset is created.

As seen in the demo code in `main.py`, most of the action happens in
`Vocab.sample()` under `grammar.py`. We sample a random number of intransitive
verbs, transitive verbs, adverbs, nouns, and adjectives, and generate a
corresponding set of random nonce words for each lexical role. Because
adjectives and adverbs also pick out objects in the world, each is assigned a
random weight vector (whose use is described in "Semantics" below). These word
lists and weight vectors collectively form a `Vocab`. Given a `Vocab`, we
construct a `Grammar` that produces transitive and intransitive sentences (with
a depth-limited recursive operation for introducing adjectives and adverbs). We
additionally allow top-level coordination of multiple commands (again via
depth-limited recursion). To summarize, sentences are of the form

    ((Adverb* Verb (Adjective* Noun)?)+

It is possible to sample from a `Grammar` to obtain paired _sentences_
(sequences of tokens) and _meanings_ (hacky neo-Davidsonian logical forms). For
example, the sentence

    quickly push a blue ball and jump

gets associated with the logical form

    lambda $v1, $v2. exists $n. push($v1) and patient($v1, $n) and blue($n) and
    ball($n) and jump($v2) and before($v1, $v2)

In order to pair instructions with demonstrations, we need to map from meanings
to sequences of low-level environment actions.

## Semantics

The example logical form depicted above specifies two actions: a ball-pushing
action and a jumping action. These must be performed in a grounded context. In
this library, computations associated with grounding live in `world.py`. The
grounded context for a specific instruction is represented by a `Situation`.
`Situations` are generated from `World`s (again, see the demo in `main.py`).
`Situations` are basically environment states analogous to the ones used in
reinforcement learning packages like OpenAI's Gym---they expose a set of
features to the agent, and accept low-level actions that cause them to
transition into new states.

For grounded SCAN, low-level actions are represented by the `Command` class. At
present, a `Command` consists of a direction (one of north, east, south, west or
"stay") and an action specification (a logical form describing a single event,
e.g.

    lambda $v2. jump($v2)

from above).

**TODO:** this is ugly and we should find a representation better suited to
prediction in a machine learning context.

A `Situation` tracks the agent's current location, and the locations of a number
of randomly-generated objects, each of which is represented by a feature vector. 
Where a command constraints its argument (like `ball`), we determine
whether an object matches the constraints on the argument by taking the dot
product between the argument features and the weights associated with the
constraint word in the `Vocab`. The object is determined to satisfy the
predicate if the dot product is greater than zero. For compound predicates
(`blue ball`) the object must satisfy each predicate individually.

Helper code is provided that will take a complete logical form and a `Situation`
and generate a sequence of `Commands` consistent with the actions required by
the logical form. For intransitive verbs, the agent will simply execute the
command in place; for transitive verbs, it will first navigate to an object
satisfying constraints on the argument, then perform the action.

## Generalization

After instructions have been generated, we construct various held-out sets
designed to test different notions of compositional generalization. These are:

**adverb**:
Contains an adverb not seen in the main training set.

**adjective**:
Contains an adjective not seen in the main training set (in contrast to above,
tests generalization in perception rather than action)

**object**:
Contains situations in which the agent must interact with an object that
combines features never seen together in training (analogous to the CLEVR
"CoGenT" split).

**application**
The agent must recognize an object as satisfying a predicate, even though it's
never seen that predicate--object combo used (even if it has seen both the
predicate and object in other contexts).

**composition**:
The agent must recognize an object as satisfying two predicates that have never
previously been conjoined.

**recursion**
Held-out commands contain a predicate introduced by a rule at depth _k_, where
the training set only ever uses it at depth ever seen it at _j_ < _k_ (analogous
to SHAPES "depth" split).

## Documentation
![Code flow](https://raw.githubusercontent.com/jacobandreas/grounded-scan/dev/documentation/Code-flow%20diagram.png)

## Example gridworld command and demonstration
![Grid World](https://raw.githubusercontent.com/jacobandreas/grounded-scan/dev/documentation/movie.gif)
