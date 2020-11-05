import math
import gym
from enum import IntEnum
import numpy as np
from gym import spaces
from gym.utils import seeding

# Size in pixels of a cell in the full-scale human view
CELL_PIXELS = 60

# Map of color names to RGB values
COLORS = {
    'red': np.array([128, 0, 0]),
    'green': np.array([46, 139, 87]),
    'blue': np.array([25, 25, 112]),
    'purple': np.array([112, 39, 195]),
    'yellow': np.array([255, 191, 0]),
    'grey': np.array([100, 100, 100]),
    'pink': np.array([255, 192, 203])
}

COLOR_NAMES = sorted(list(COLORS.keys()))

# Used to map colors to integers
COLOR_TO_IDX = {
    'red': 0,
    'green': 1,
    'blue': 2,
    'purple': 3,
    'yellow': 4,
    'grey': 5,
    'pink': 6
}

IDX_TO_COLOR = dict(zip(COLOR_TO_IDX.values(), COLOR_TO_IDX.keys()))

# Map of object type to integers
OBJECT_TO_IDX = {
    'unseen': 0,
    'empty': 1,
    'circle': 2,
    'cylinder': 3,
    'square': 4,
    'agent': 5,
}

IDX_TO_OBJECT = dict(zip(OBJECT_TO_IDX.values(), OBJECT_TO_IDX.keys()))

# Map of agent direction indices to vectors
DIR_TO_VEC = [
    # Pointing right (positive X)
    np.array((1, 0)),
    # Down (positive Y)
    np.array((0, 1)),
    # Pointing left (negative X)
    np.array((-1, 0)),
    # Up (negative Y)
    np.array((0, -1)),
]

# TODO: change
WEIGHT_TO_MOMENTUM = {
    "light": 1,
    "heavy": 2
}


class WorldObj:
    """
    Base class for grid world objects
    """

    def __init__(self, type, color, size=1, vector_representation=None, object_representation=None, target=False,
                 weight="light"):
        assert type in OBJECT_TO_IDX, type
        assert color in COLOR_TO_IDX, color
        assert 1 <= size <= 4, "Sizes outside of range [1,4] not supported."
        self.type = type
        self.color = color
        self.border_color = color
        self.contains = None
        self.size = size

        # Initial position of the object
        self.init_pos = None

        # Current position of the object
        self.cur_pos = None

        # Representations
        self.vector_representation = vector_representation
        self.object_representation = object_representation

        # Boolean whether an object is a target
        self.target = target

        # Determining whether a heavy object can be moved in the next step or not
        self.momentum = 0
        self.weight = weight
        self.momentum_threshold = WEIGHT_TO_MOMENTUM[self.weight]
        self.pushed = False
        self.pulled = False

    def can_overlap(self):
        """Can the agent overlap with this?"""
        return True

    def can_pickup(self):
        """Can the agent pick this up?"""
        return False

    def can_push(self):
        """Can the agent push this?"""
        return False

    def render(self, r):
        """Draw this object with the given renderer"""
        raise NotImplementedError

    def _set_color(self, r):
        """Set the color of this object as the active drawing color"""
        c = COLORS[self.color]
        border_color = COLORS[self.border_color]
        r.setLineColor(border_color[0], border_color[1], border_color[2])
        r.setColor(c[0], c[1], c[2])


class Square(WorldObj):
    def __init__(self, color='grey', size=1, vector_representation=None, object_representation=None, target=False,
                 weight="light"):
        super().__init__('square', color, size, vector_representation=vector_representation,
                         object_representation=object_representation, target=target, weight=weight)

    def render(self, r):
        self._set_color(r)

        # TODO: max_size is 4 here hardcoded
        r.drawPolygon([
            (0, CELL_PIXELS * (self.size / 4)),
            (CELL_PIXELS * (self.size / 4), CELL_PIXELS * (self.size / 4)),
            (CELL_PIXELS * (self.size / 4), 0),
            (0, 0)
        ])

    def can_pickup(self):
        return True

    def can_push(self):
        return True

    def push(self):
        self.momentum += 1
        if self.momentum >= self.momentum_threshold:
            self.momentum = 0
            return True
        else:
            return False


class Cylinder(WorldObj):
    def __init__(self, color='blue', size=1, vector_representation=None, object_representation=None, weight="light"):
        super(Cylinder, self).__init__('cylinder', color, size, vector_representation,
                                       object_representation=object_representation, weight=weight)
        # TODO: generalize sizes

    def can_pickup(self):
        return True

    def render(self, r):
        self._set_color(r)

        # Vertical quad
        parallelogram_width = (CELL_PIXELS / 2) * (self.size / 4)
        parallelogram_height = CELL_PIXELS * (self.size / 4)
        r.drawPolygon([
            (CELL_PIXELS / 2, 0),
            (CELL_PIXELS / 2 + parallelogram_width, 0),
            (CELL_PIXELS / 2, parallelogram_height),
            (CELL_PIXELS / 2 - parallelogram_width, parallelogram_height)
        ])

    def can_push(self):
        return True

    def push(self):
        self.momentum += 1
        if self.momentum >= self.momentum_threshold:
            self.momentum = 0
            return True
        else:
            return False


class Circle(WorldObj):
    def __init__(self, color='blue', size=1, vector_representation=None, object_representation=None, target=False,
                 weight="light"):
        super(Circle, self).__init__('circle', color, size, vector_representation,
                                     object_representation=object_representation, target=target, weight=weight)

    def can_pickup(self):
        return True

    def can_push(self):
        return True

    def render(self, r):
        self._set_color(r)
        r.drawCircle(CELL_PIXELS * 0.5, CELL_PIXELS * 0.5, CELL_PIXELS // 10 * self.size)

    def push(self):
        self.momentum += 1
        if self.momentum >= self.momentum_threshold:
            self.momentum = 0
            return True
        else:
            return False


class Grid:
    """
    Represent a grid and operations on it
    """

    def __init__(self, width, height, depth):
        assert width >= 3
        assert height >= 3

        self.width = width
        self.height = height
        self._num_attributes_object = depth
        self.grid = [None] * width * height

    def __contains__(self, key):
        if isinstance(key, WorldObj):
            for e in self.grid:
                if e is key:
                    return True
        elif isinstance(key, tuple):
            for e in self.grid:
                if e is None:
                    continue
                if (e.color, e.type) == key:
                    return True
                if key[0] is None and key[1] == e.type:
                    return True
        return False

    def __eq__(self, other):
        grid1 = self.encode()
        grid2 = other.encode()
        return np.array_equal(grid2, grid1)

    def __ne__(self, other):
        return not self == other

    def copy(self):
        from copy import deepcopy
        return deepcopy(self)

    def set(self, i, j, v):
        assert i >= 0 and i < self.width
        assert j >= 0 and j < self.height
        self.grid[j * self.width + i] = v

    def get(self, i, j):
        assert i >= 0 and i < self.width
        assert j >= 0 and j < self.height
        return self.grid[j * self.width + i]

    def rotate_left(self):
        """
        Rotate the grid to the left (counter-clockwise)
        """

        grid = Grid(self.height, self.width)

        for i in range(self.width):
            for j in range(self.height):
                v = self.get(i, j)
                grid.set(j, grid.height - 1 - i, v)

        return grid

    def slice(self, topX, topY, width, height):
        """
        Get a subset of the grid
        """

        grid = Grid(width, height)

        for j in range(0, height):
            for i in range(0, width):
                x = topX + i
                y = topY + j

                if x >= 0 and x < self.width and \
                   y >= 0 and y < self.height:
                    v = self.get(x, y)
                else:
                    v = Square()

                grid.set(i, j, v)

        return grid

    def render(self, r, tile_size, attention_weights=[]):
        """
        Render this grid at a given scale
        :param r: target renderer object
        :param tile_size: tile size in pixels
        """

        assert r.width == self.width * tile_size
        assert r.height == self.height * tile_size

        # Total grid size at native scale
        widthPx = self.width * CELL_PIXELS
        heightPx = self.height * CELL_PIXELS

        r.push()

        # Internally, we draw at the "large" full-grid resolution, but we
        # use the renderer to scale back to the desired size
        r.scale(tile_size / CELL_PIXELS, tile_size / CELL_PIXELS)

        if len(attention_weights) > 0:
            if len(attention_weights) == self.width * self.height:
                pixel_attention = False
                attention_weights = attention_weights.reshape(self.width, self.height)
            elif len(attention_weights) == self.width * CELL_PIXELS * self.height * CELL_PIXELS:
                pixel_attention = True
                attention_weights = attention_weights.reshape(self.width * CELL_PIXELS, self.height * CELL_PIXELS)
            start_range = 0
            end_range = 150
        else:
            pixel_attention = False
        # Draw the background of the in-world cells black
        if not pixel_attention:
            r.fillRect(
                0,
                0,
                widthPx,
                heightPx,
                255, 255, 255
            )
        else:
            for j in range(0, heightPx):
                for i in range(0, widthPx):
                    current_weight = attention_weights[j, i]
                    color = int((end_range - start_range) * (1 - current_weight))
                    r.push()
                    r.fillRect(i, j, 1, 1, r=color, g=color, b=color)
                    r.pop()

        # Draw grid lines
        r.setLineColor(100, 100, 100)
        for rowIdx in range(0, self.height):
            y = CELL_PIXELS * rowIdx
            r.drawLine(0, y, widthPx, y)
        for colIdx in range(0, self.width):
            x = CELL_PIXELS * colIdx
            r.drawLine(x, 0, x, heightPx)

        # Render the grid
        for j in range(0, self.height):
            for i in range(0, self.width):
                cell = self.get(i, j)
                if len(attention_weights) > 0 and not pixel_attention:
                    current_weight = attention_weights[j, i]
                    color = int((end_range - start_range) * (1 - current_weight))
                    r.push()
                    r.fillRect(i * CELL_PIXELS, j * CELL_PIXELS, CELL_PIXELS, CELL_PIXELS, r=color, g=color, b=color)
                if cell == None:
                    continue
                r.push()
                r.translate(i * CELL_PIXELS, j * CELL_PIXELS)
                cell.render(r)
                r.pop()

        r.pop()

    def encode(self, agent_row: int, agent_column: int, agent_direction: int):
        """
        Produce a compact numpy encoding of the grid.
        """
        array = np.zeros((self.width, self.height, self._num_attributes_object + 1 + 4), dtype='uint8')
        for col in range(self.width):
            for row in range(self.height):
                grid_cell = self.get(col, row)
                empty_representation = np.zeros(self._num_attributes_object + 1 + 4)
                if grid_cell:
                    empty_representation[:-5] = grid_cell.vector_representation

                # Set agent feature to 1 for the grid cell with the agent and add it's direction in one-hot form.
                if col == agent_column and row == agent_row:
                    empty_representation[-5] = 1
                    one_hot_direction = np.zeros(4)
                    one_hot_direction[agent_direction] = 1
                    empty_representation[-4:] = one_hot_direction
                array[row, col, :] = empty_representation
        return array


class MiniGridEnv(gym.Env):
    """
    2D grid world game environment.
    """

    metadata = {
        'render.modes': ['human', 'rgb_array', 'pixmap'],
        'video.frames_per_second': 10
    }

    # Enumeration of possible actions
    class Actions(IntEnum):
        # Turn left, turn right, move forward
        left = 0
        right = 1
        forward = 2

        # Pick up an object
        pickup = 3
        # Drop an object
        drop = 4

        # Done completing task
        done = 6

    def __init__(self, grid_size=None, width=None, height=None, max_steps=100, seed=1337):
        # Can't set both grid_size and width/height
        if grid_size:
            assert width == None and height == None
            width = grid_size
            height = grid_size

        # Action enumeration for this environment
        self.actions = MiniGridEnv.Actions

        # Actions are discrete integer values
        self.action_space = spaces.Discrete(len(self.actions))

        # Range of possible rewards
        self.reward_range = (0, 1)

        # Renderer object used to render the whole grid (full-scale)
        self.grid_render = None

        # Renderer used to render observations (small-scale agent view)
        self.obs_render = None

        # Environment configuration
        self.width = width
        self.height = height
        self.max_steps = max_steps

        # Current position and direction of the agent
        self.agent_pos = None
        self.agent_dir = None

        # Initialize the RNG
        self.seed(seed=seed)

        # Initialize the state
        self.reset()

    def reset(self):
        # Current position and direction of the agent
        self.agent_pos = None
        self.agent_dir = None

        # Generate a new random grid at the start of each episode
        # To keep the same grid for each episode, call env.seed() with
        # the same seed before calling env.reset()
        self._gen_grid(self.width, self.height)

        # These fields should be defined by _gen_grid
        assert self.agent_pos is not None
        assert self.agent_dir is not None

        # Check that the agent doesn't overlap with an object
        start_cell = self.grid.get(*self.agent_pos)
        assert start_cell is None or start_cell.can_overlap()

        # Item picked up, being carried, initially nothing
        self.carrying = None

        # Step count since episode start
        self.step_count = 0

        return

    def seed(self, seed=1337):
        # Seed the random number generator
        self.np_random, _ = seeding.np_random(seed)
        return [seed]

    @property
    def steps_remaining(self):
        return self.max_steps - self.step_count

    def __str__(self):
        """
        Produce a pretty string of the environment's grid along with the agent.
        A grid cell is represented by 2-character string, the first one for
        the object and the second one for the color.
        """

        # Map of object types to short string
        OBJECT_TO_STR = {
            'circle': 'A',
            'square': 'B',
            'cylinder': 'C',
        }

        # Map agent's direction to short string
        AGENT_DIR_TO_STR = {
            0: '>',
            1: 'V',
            2: '<',
            3: '^'
        }

        str = ''
        for j in range(self.grid.height):
            for i in range(self.grid.width):
                if i == self.agent_pos[0] and j == self.agent_pos[1]:
                    str += 2 * AGENT_DIR_TO_STR[self.agent_dir]
                    continue
                c = self.grid.get(i, j)
                if not c:
                    str += '  '
                    continue
                str += OBJECT_TO_STR[c.type] + c.color[0].upper()
            if j < self.grid.height - 1:
                str += '\n'
        return str

    def _gen_grid(self, width, height):
        assert False, "_gen_grid needs to be implemented by each environment"

    def _rand_int(self, low, high):
        """
        Generate random integer in [low,high[
        """

        return self.np_random.randint(low, high)

    def place_obj(self, obj, top=None, size=None, reject_fn=None, max_tries=math.inf):
        """
        Place an object at an empty position in the grid

        :param obj:
        :param top: top-left position of the rectangle where to place
        :param size: size of the rectangle where to place
        :param reject_fn: function to filter out potential positions
        :param max_tries:
        """

        if top is None:
            top = (0, 0)
        else:
            top = (max(top[0], 0), max(top[1], 0))

        if size is None:
            size = (self.grid.width, self.grid.height)

        num_tries = 0

        while True:
            # This is to handle with rare cases where rejection sampling
            # gets stuck in an infinite loop
            if num_tries > max_tries:
                raise RecursionError('rejection sampling failed in place_obj')

            num_tries += 1

            pos = np.array((
                self._rand_int(top[0], min(top[0] + size[0], self.grid.width)),
                self._rand_int(top[1], min(top[1] + size[1], self.grid.height))
            ))

            # Don't place the object on top of another object
            if self.grid.get(*pos) != None:
                continue

            # Check if there is a filtering criterion
            if reject_fn and reject_fn(self, pos):
                continue

            break

        self.grid.set(*pos, obj)

        if obj is not None:
            obj.init_pos = pos
            obj.cur_pos = pos

        return pos

    def place_agent(
        self,
        top=None,
        size=None,
        rand_dir=True,
        max_tries=math.inf
    ):
        """
        Set the agent's starting point at an empty position in the grid
        """

        self.agent_pos = None
        pos = self.place_obj(None, top, size, max_tries=max_tries)
        self.agent_pos = pos

        if rand_dir:
            self.agent_dir = self._rand_int(0, 4)

        return pos

    @property
    def dir_vec(self):
        """
        Get the direction vector for the agent, pointing in the direction
        of forward movement.
        """

        assert self.agent_dir >= 0 and self.agent_dir < 4
        return DIR_TO_VEC[self.agent_dir]

    @property
    def right_vec(self):
        """
        Get the vector pointing to the right of the agent.
        """

        dx, dy = self.dir_vec
        return np.array((-dy, dx))

    @property
    def front_pos(self):
        """
        Get the position of the cell that is right in front of the agent
        """

        return self.agent_pos + self.dir_vec

    def step(self, action):
        self.step_count += 1

        reward = 0
        done = False

        current_cell = self.grid.get(*self.agent_pos)

        # Rotate left
        if action == self.actions.left:
            self.agent_dir -= 1
            if self.agent_dir < 0:
                self.agent_dir += 4

        # Rotate right
        elif action == self.actions.right:
            self.agent_dir = (self.agent_dir + 1) % 4

        # Move forward
        elif action == self.actions.forward:
            # Get the position in front of the agent
            fwd_pos = self.front_pos

            # Get the contents of the cell in front of the agent
            fwd_cell = self.grid.get(*fwd_pos)
            if fwd_cell == None or fwd_cell.can_overlap():
                self.agent_pos = fwd_pos
            if fwd_cell != None and fwd_cell.type == 'goal':
                done = True
                reward = self._reward()
            if fwd_cell != None and fwd_cell.type == 'lava':
                done = True

        # Pick up an object
        elif action == self.actions.pickup:
            if current_cell.can_pickup():
                if self.carrying is None:
                    self.carrying = current_cell
                    self.carrying.cur_pos = np.array([-1, -1])
                    self.grid.set(*self.agent_pos, None)

        # Drop an object
        elif action == self.actions.drop:
            if not current_cell and self.carrying:
                self.grid.set(*self.agent_pos, self.carrying)
                self.carrying.cur_pos = self.agent_pos
                self.carrying = None

        # Done action (not used by default)
        elif action == self.actions.done:
            pass

        else:
            assert False, "unknown action"

        if self.step_count >= self.max_steps:
            done = True

        return reward, done, {}

    def render(self, mode='', close=False, highlight=True, tile_size=CELL_PIXELS, attention_weights=[]):
        """
        Render the whole-grid human view
        """

        if close:
            if self.grid_render:
                self.grid_render.close()
            return

        if self.grid_render is None or self.grid_render.window is None or (self.grid_render.width != self.width * tile_size):
            from GroundedScan.gym_minigrid.rendering import Renderer
            self.grid_render = Renderer(
                self.width * tile_size,
                self.height * tile_size,
                True if mode == 'human' else False
            )

        r = self.grid_render

        if r.window:
            r.window.setText(self.mission)

        r.beginFrame()

        # Render the whole grid
        if len(attention_weights) > 0:
            flat_attention_weights = attention_weights[0]
        else:
            flat_attention_weights = attention_weights
        self.grid.render(r, tile_size, attention_weights=flat_attention_weights)

        # Draw the agent
        ratio = tile_size / CELL_PIXELS
        r.push()
        r.scale(ratio, ratio)
        r.translate(
            CELL_PIXELS * (self.agent_pos[0] + 0.5),
            CELL_PIXELS * (self.agent_pos[1] + 0.5)
        )
        r.rotate(self.agent_dir * 90)
        r.setLineColor(255, 192, 203)
        r.setColor(255, 192, 203)
        r.drawPolygon([
            (-12, 10),
            (12, 0),
            (-12, -10)
        ])
        r.pop()
        r.endFrame()

        if mode == 'rgb_array':
            return r.getArray()
        elif mode == 'pixmap':
            return r.getPixmap()
        return r
