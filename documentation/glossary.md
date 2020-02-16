# Term glossary
| Term | Description |
| ----------- | ----------- |
| Command | A command refers to the synthetic language instruction the agent needs to follow. E.g. 'push the small blue circle.' |
| Demonstration | A demonstration refers to the sequence of action commands that correctly caries out an command/instruction w.r.t. a world state/situation. |
| Derivation | A `Derivation` is the constituency tree for an input command / instruction. |
| Instruction | The same as a command, this refers to the synthetic language instruction the agent needs to follow. E.g. 'walk to the red square.' |
| Logical Form | This is the form the input instruction is parsed into to generate a demonstration. It finds the action verb, the 'manner' (i.e. the adverb) and the target referrent. |
| Semantically Equivalent | Two examples are considered 'equivalent' if of a pair of examples the input instruction and target sequence are the same and the target object is located on the same grid cell. NB: the location of the distractor objects is irrelevant in this definition. |
| Situation | A situation is an particular instantiation of the `World` class, and represents one world state in a data example. |
| Split | A split refers to one test set or experiment with a particular type of data example, e.g. all examples with the adverb `cautiously` are part of the split `adverb_1`. |
| Template | A template instruction refers to an instruction in non-terminal symbols before it is filled with terminal symbols. E.g. the template `VV_transitive 'a' JJ JJ NN` could be instatiated with `Push a big red square`. |
| World | The world class is built on the minigrid environment and implements the grid world. |
| World State | The same as a situation, one instatiation of the world. |
| `visual` (split)| This refers to the 'Red Squares'-split in the paper. |
| `visual_easier` (split)| This refers to the 'Yellow Squares'-split in the paper.|
|`situational_1` (split)| This refers to the 'Novel Direction'-split in the paper.|
|`situational_2` (split)| This refers to the 'Relativity of Size'-split in the paper.|
|`contextual` (split)| This refers to the 'Class Inference'-split in the paper.|
|`adverb_1` (split)| This refers to the 'Adverb k-shot'-split in the paper.|
|`adverb_2` (split)| This refers to the 'Adverb to verb'-split in the paper.|