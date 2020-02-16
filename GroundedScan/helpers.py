import numpy as np
from typing import Tuple
from typing import List
from typing import Any
import matplotlib.pyplot as plt
import cv2

from GroundedScan.gym_minigrid.minigrid import DIR_TO_VEC


# TODO faster
def topo_sort(items, constraints):
    if not constraints:
        return items
    items = list(items)
    constraints = list(constraints)
    out = []
    while len(items) > 0:
        roots = [
            i for i in items
            if not any(c[1] == i for c in constraints)
        ]
        assert len(roots) > 0, (items, constraints)
        to_pop = roots[0]
        items.remove(to_pop)
        constraints = [c for c in constraints if c[0] != to_pop]
        out.append(to_pop)
    return out


def random_weights(size: int) -> np.ndarray:
    return 2 * (np.random.random(size) - 0.5)


def accept_weights(size: int) -> np.ndarray:
    return np.ones(size)


def plan_step(position: Tuple[int, int], move_direction: int):
    """

    :param position: current position of form (x-axis, y-axis) (i.e. column, row)
    :param move_direction: East is 0, south is 1, west is 2, north is 3.
    :return: next position of form (x-axis, y-axis) (i.e. column, row)
    """
    assert 0 <= move_direction < 4
    dir_vec = DIR_TO_VEC[move_direction]
    return position + dir_vec


def one_hot(size: int, idx: int) -> np.ndarray:
    one_hot_vector = np.zeros(size, dtype=int)
    one_hot_vector[idx] = 1
    return one_hot_vector


def generate_possible_object_names(color: str, shape: str) -> List[str]:
    # TODO: does this still make sense when size is not small or large
    names = [shape, ' '.join([color, shape])]
    return names


def save_counter(description, counter, file):
    file.write(description + ": \n")
    for key, occurrence_count in counter.items():
        file.write("   {}: {}\n".format(key, occurrence_count))


def bar_plot(values: dict, title: str, save_path: str, errors={}, y_axis_label="Occurrence"):
    sorted_values = list(values.items())
    sorted_values = [(y, x) for x, y in sorted_values]
    sorted_values.sort()
    values_per_label = [value[0] for value in sorted_values]
    if len(errors) > 0:
        sorted_errors = [errors[value[1]] for value in sorted_values]
    else:
        sorted_errors = None
    labels = [value[1] for value in sorted_values]
    assert len(labels) == len(values_per_label)
    y_pos = np.arange(len(labels))

    plt.bar(y_pos, values_per_label, yerr=sorted_errors, align='center', alpha=0.5)
    plt.gcf().subplots_adjust(bottom=0.2, )
    plt.xticks(y_pos, labels, rotation=90, fontsize="xx-small")
    plt.ylabel(y_axis_label)
    plt.title(title)

    plt.savefig(save_path)
    plt.close()


def grouped_bar_plot(values: dict, group_one_key: Any, group_two_key: Any, title: str, save_path: str,
                     errors_group_one={}, errors_group_two={}, y_axis_label="Occurence", sort_on_key=True):
    sorted_values = list(values.items())
    if sort_on_key:
        sorted_values.sort()
    values_group_one = [value[1][group_one_key] for value in sorted_values]
    values_group_two = [value[1][group_two_key] for value in sorted_values]
    if len(errors_group_one) > 0:
        sorted_errors_group_one = [errors_group_one[value[0]] for value in sorted_values]
        sorted_errors_group_two = [errors_group_two[value[0]] for value in sorted_values]
    else:
        sorted_errors_group_one = None
        sorted_errors_group_two = None
    labels = [value[0] for value in sorted_values]
    assert len(labels) == len(values_group_one)
    assert len(labels) == len(values_group_two)
    y_pos = np.arange(len(labels))

    fig, ax = plt.subplots()
    width = 0.35
    p1 = ax.bar(y_pos, values_group_one, width, yerr=sorted_errors_group_one, align='center', alpha=0.5)
    p2 = ax.bar(y_pos + width, values_group_two, width, yerr=sorted_errors_group_two, align='center', alpha=0.5)
    plt.gcf().subplots_adjust(bottom=0.2, )
    plt.xticks(y_pos, labels, rotation=90, fontsize="xx-small")
    plt.ylabel(y_axis_label)
    plt.title(title)
    ax.legend((p1[0], p2[0]), (group_one_key, group_two_key))

    plt.savefig(save_path)
    plt.close()


def numpy_array_to_image(numpy_array, image_name):
    plt.imsave(image_name, numpy_array)


def image_to_numpy_array(image_path):
    im = cv2.imread(image_path)
    return np.flip(im, 2)  # cv2 returns image in BGR order
