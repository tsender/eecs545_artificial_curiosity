---
menu: main
title: API Documentation
---

<a name="artificial_curiosity_types"></a>
# artificial\_curiosity\_types

<a name="artificial_curiosity_types.addType"></a>
#### addType

```python
addType(**args)
```

Makes it easier to add types that reference other custom types
Parameters
__________
**args : typeing.Type
    Can add any type by passing <name> = <type>, where <type> is a type from the typing library

Returns
_______
None

<a name="memory"></a>
# memory

<a name="memory.BaseMemory"></a>
## BaseMemory Objects

```python
class BaseMemory(, metaclass=abc.ABCMeta)
```

Base memory class for the agent's brain.

Methods

`__init__(maxLength: int = 64)`  
    Initializes the memory unit with a default capacity of 64 Experiences
`push(data: Experience)`  
    Adds an Experience to the memory unit.
`as_list() -> List[Experience]`  
    Returns a list of Experience instances

<a name="memory.BaseMemory.__init__"></a>
#### \_\_init\_\_

```python
 | __init__(max_length: int = 64)
```

**Arguments**:

  maxLength : int
  The maximum number of experiences(Experience) that the memory unit can contain

<a name="memory.BaseMemory.push"></a>
#### push

```python
 | @abc.abstractmethod
 | push(data: Experience)
```

Add an experience to memory

Args
    data : Experience  
        An experience to add

<a name="memory.BaseMemory.get_name"></a>
#### get\_name

```python
 | @abc.abstractmethod
 | get_name()
```

Returns the name of the memory object.

Returns
    The name of the memory object as a string

<a name="memory.BaseMemory.as_list"></a>
#### as\_list

```python
 | as_list() -> List[Experience]
```

Returns a copy of the current memory

Returns
    A list of Experience objects

<a name="memory.PriorityBasedMemory"></a>
## PriorityBasedMemory Objects

```python
class PriorityBasedMemory(BaseMemory)
```

Memory class that uses a fixed-length priority queue to store experiences based on their novelty.
Low novelty corresponds to higher priority (also makes it easier to remove the experience).

<a name="memory.PriorityBasedMemory.push"></a>
#### push

```python
 | push(data: Experience)
```

Add an experience to memory

Args
    data : Experience  
        An experience to add. If full, experiences that are less novel are removed (forgotten).

<a name="memory.PriorityBasedMemory.get_name"></a>
#### get\_name

```python
 | get_name()
```

Returns the full descriptive name of the memory object.

Returns
    The name of the memory object as a string

<a name="memory.ListBasedMemory"></a>
## ListBasedMemory Objects

```python
class ListBasedMemory(BaseMemory)
```

Memory class that uses a simple fixed-length list to store the latest experiences.

<a name="memory.ListBasedMemory.push"></a>
#### push

```python
 | push(data: Experience)
```

Add an experience to memory

Args
    data : Experience  
        An experience to add. If full, remove oldest experience and add new experience.

<a name="memory.ListBasedMemory.get_name"></a>
#### get\_name

```python
 | get_name()
```

Returns the full descriptive name of the memory object.

Returns
    The name of the memory object as a string

<a name="__init__"></a>
# \_\_init\_\_

<a name="experience"></a>
# experience

<a name="experience.Experience"></a>
## Experience Objects

```python
class Experience()
```

A type for a measure of novelty and an associated image. This is meant to be used by the Memory and the Brain

Attributes
----------
self.novelty : float
    A float that represents the movelty of the Experience
self.grain : Image.Image
    An image that will show us what the machine remembers


Methods
-------
__init__(nov: float, grn: Image.Image)
    Initializes the Experience with the given novelty, feature vector, and image
__lt__(other)
    Compares against the novelty. Works for scalars and other instances of Experience
__le__(other)
    Compares against the novelty. Works for scalars and other instances of Experience
__gt__(other)
    Compares against the novelty. Works for scalars and other instances of Experience
__ge__(other)
    Compares against the novelty. Works for scalars and other instances of Experience
__eq__(other)
    Compares against the novelty. Works for scalars and other instances of Experience
__ne__(other)
    Compares against the novelty. Works for scalars and other instances of Experience
__str__()
    Returns a string representation of Experience

<a name="experience.Experience.__init__"></a>
#### \_\_init\_\_

```python
 | __init__(nov: float, grn: Image.Image)
```

Parameters
__________
nov : float
    The measure of novelty, expressed as a float
grn: Image.Image
    A grain (image) to be remembered. This exists so we can reference it later

Returns
_______
Experience

<a name="experience.Experience.__lt__"></a>
#### \_\_lt\_\_

```python
 | __lt__(other)
```

Parameters
__________
other : any

Returns
_______
bool

<a name="experience.Experience.__le__"></a>
#### \_\_le\_\_

```python
 | __le__(other)
```

Parameters
__________
other : any

Returns
_______
bool

<a name="experience.Experience.__gt__"></a>
#### \_\_gt\_\_

```python
 | __gt__(other)
```

Parameters
__________
other : any

Returns
_______
bool

<a name="experience.Experience.__ge__"></a>
#### \_\_ge\_\_

```python
 | __ge__(other)
```

Parameters
__________
other : any

Returns
_______
bool

<a name="experience.Experience.__eq__"></a>
#### \_\_eq\_\_

```python
 | __eq__(other)
```

Parameters
__________
other : any

Returns
_______
bool

<a name="experience.Experience.__ne__"></a>
#### \_\_ne\_\_

```python
 | __ne__(other)
```

Parameters
__________
other : any

Returns
_______
bool

<a name="experience.Experience.__str__"></a>
#### \_\_str\_\_

```python
 | __str__()
```

Parameters
__________
None

Returns
_______
string

<a name="agent"></a>
# agent

<a name="agent.Motivation"></a>
## Motivation Objects

```python
class Motivation(, metaclass=abc.ABCMeta)
```

This is an abstract class that represents the decision making process (or motivation) behind an agent

<a name="agent.Motivation.get_from_position"></a>
#### get\_from\_position

```python
 | @abc.abstractmethod
 | get_from_position(position: Tuple[int])
```

This will get the agents next position based on the current position passed to it

<a name="agent.Curiosity"></a>
## Curiosity Objects

```python
class Curiosity(Motivation)
```

This class extends Motivation and creates a curious motivation for an agent

<a name="agent.Curiosity.get_from_position"></a>
#### get\_from\_position

```python
 | get_from_position(position: Tuple[int])
```

Implements the abstract method from Motivation. Gets the next position from the current position

<a name="agent.Curiosity.get_reconstruction_snapshot"></a>
#### get\_reconstruction\_snapshot

```python
 | get_reconstruction_snapshot(position: Tuple[int])
```

Saves the full-view and its reconstruction from the current position

<a name="agent.Random"></a>
## Random Objects

```python
class Random(Motivation)
```

This class extends Motivation, and randomly selects a position based on what is available

<a name="agent.Random.get_from_position"></a>
#### get\_from\_position

```python
 | get_from_position(position: Tuple[int])
```

Implements the abstract method from Motivation. Gets the next position from the current position

<a name="agent.Linear"></a>
## Linear Objects

```python
class Linear(Motivation)
```

This class extends Motivation, and is designed to move on a linear path

<a name="agent.Linear.get_from_position"></a>
#### get\_from\_position

```python
 | get_from_position(position: Tuple[int])
```

Implements the abstract method from Motivation. Gets the next position from the current position

<a name="agent.Agent"></a>
## Agent Objects

```python
class Agent()
```

This abtracts the motivation away so that we can iterate over them later

<a name="agent.Agent.set_data_dir"></a>
#### set\_data\_dir

```python
 | set_data_dir(data_dir: str)
```

Set a new data directory

<a name="agent.Agent.step"></a>
#### step

```python
 | step()
```

This performs a simple step for the agent, moving it from one position to the next

<a name="agent.Agent.save_reconstruction_snapshot"></a>
#### save\_reconstruction\_snapshot

```python
 | save_reconstruction_snapshot()
```

Saves the full-view and its reconstruction from the current position

<a name="agent.Agent.get_path_novelty"></a>
#### get\_path\_novelty

```python
 | get_path_novelty()
```

Return average path variance. Note, images are converted to the range [-1,1]

<a name="agent.Agent.save_data"></a>
#### save\_data

```python
 | save_data()
```

Save agent's data to a folder

<a name="testing"></a>
# testing

<a name="engine"></a>
# engine

<a name="engine.plot_paths"></a>
#### plot\_paths

```python
plot_paths(map: Map, agent_list: List[Agent], show: bool, save: bool, dirname: str)
```

Plots out the paths of the agents on the map

Params
------
map: Map
    A map that will be used to get the bounds and background for plotting

agent_list: List[Agent]
    A list of agents whose paths need to be plotted

show: bool
    Whether the plots should be displayed or not

save: bool=False
    Whether the graphs should be saved

dirname: str
    The directory where the images will be stored

Returns
-------
None

<a name="engine.run_agents"></a>
#### run\_agents

```python
run_agents(agent_list: List[Agent], path_length: int)
```

Runs an experiment on the provided agents.

Params
------
agent_list: List[Agent]
    A list of Agent instances to be ran

path_length: int
    The number of steps that each agent should take.

Returns
-------
None

<a name="engine.run_experiments"></a>
#### run\_experiments

```python
run_experiments(map: Map, num_starting_positions)
```

Run a series of experiments. Generate Random, Linear, and Curiosity agents for each starting position.
Test a series of brain configurations for the Curiosity agent so we can see if there is an optimal configuration.

<a name="engine.save_and_plot"></a>
#### save\_and\_plot

```python
save_and_plot(agent_list: List[Agent], novelty_filename: str, dirname: str, show: bool = False, save_plots: bool = True)
```

Save the path record of each agent as a csv file, save the novelty to a .txt file, and plot and save graphs.
Note: When plotting, it will assume all agents use the same map.

Params:
------
agent_list: List[Agent]
A list of agent whose path coordinates to be saved

novelty_filename: str
The filename for storing the agents' novelty

dirname: str
The directory name where the csv file will be saved

show: bool
Whether the graphs should be displayed

save_plots: bool
Whether the plots should be saved to the disk or not

**Returns**:

  ------
  None

<a name="engine.load_agent_data"></a>
#### load\_agent\_data

```python
load_agent_data(path: str)
```

Loads information from a given file. This will not be used as part of the engine.
However, I thought it would be useful to include here so that if we make changes to
the serialization, we have the data loding close by and can edit it easily

Params
------

path: str
    The path to the file

Returns
-------

List[Tuple[int]]
    Returns a list of x and y coordinates

<a name="engine.get_time_str"></a>
#### get\_time\_str

```python
get_time_str(time_in)
```

Convert time_in to a human-readible string, e.g. '1 days 01h:45m:32s'

**Arguments**:

- `time_in` - float
  Time in seconds
  

**Returns**:

  A string

<a name="evaluate"></a>
# evaluate

<a name="evaluate.load_from_map"></a>
#### load\_from\_map

```python
load_from_map(map: Map, positions: List[Tuple[int]])
```

Grabs images at a list of coordinates, converts them to a numpy array, and returns them

Params
------

map: Map
    The map that you want to get the images from

positions: List[Tuple[int]]
    A list of all of the points that will be sampled and turned into an numpy array

Returns
-------

A numpy array of dimension (<number_of_images>, <height_of_images>, <width_of_images>

<a name="evaluate.avg_pixelwise_var"></a>
#### avg\_pixelwise\_var

```python
avg_pixelwise_var(images_seen: np.int16)
```

Computes the variance for every pixel p across all images, resulting in a matrix holding
the variance for eack pixel p, then calculates the average of that variance across all
pixels. This allows us to compensate for different fov sizes.
Note: images are normalized to [-1,1] before calculations

Params
------

images_seen
A numpy matrix holding numpy versions of all of our images

Returns
-------

The aaverage pixelwise variation across all images, as a float

<a name="map"></a>
# map

<a name="map.down"></a>
#### down

```python
down(y_pos: int, distance: int)
```

Find the pixel a given distance from the virtual agent location

Params
------

y_pos: int
    An index along the y axis that will be used as the starting point

distance: int
    The distance we want to trayel along the y axis from that point.

Returns
-------

An integer for the new location

<a name="map.up"></a>
#### up

```python
up(y_pos: int, distance: int)
```

Find the pixel a given distance from the virtual agent location

Params
------

y_pos: int
    An index along the y axis that will be used as the starting point

distance: int
    The distance we want to trayel along the y axis from that point.

Returns
-------

An integer for the new location

<a name="map.left"></a>
#### left

```python
left(x_pos: int, distance: int)
```

Find the pixel a given distance from the virtual agent location

Params
------

x_pos: int
    An index along the x axis that will be used as the starting point

distance: int
    The distance we want to trayel along the x axis from that point.

Returns
-------

An integer for the new location

<a name="map.right"></a>
#### right

```python
right(x_pos: int, distance: int)
```

Find the pixel a given distance from the virtual agent location

Params
------

x_pos: int
    An index along the x axis that will be used as the starting point

distance: int
    The distance we want to trayel along the x axis from that point.

Returns
-------

An integer for the new location

<a name="map.Map"></a>
## Map Objects

```python
class Map()
```

This class will hold some basic information about the environment our agent is training in,
which allows us to train multiple agents in the same environment. It's main purpose is to
act as a wrapper for the image we will be using as our domain

It is important to note that the map uses an inverted y axis due to the graphics backend

<a name="map.Map.__init__"></a>
#### \_\_init\_\_

```python
 | __init__(filepath: str, fov: int, sqrtGrains: int, greyscale: bool = True)
```

Params
------
filepath: str
    The path to the image that we want to use as our domain. It will be loaded automatically

fov: int
    The radius of our agent's field of view (FOV). This is how far they can see, as well as
    how far they'll stay away from the walls (because we don't want them to be able to go
    off the map).

sqrtGrains: int
    The square root of the number of grains, or sub-images, that we want the agent to use.
    This number can also be thought of as the number of grains that are along one side of
    the square that makes ip the FOV (hence thew square root).

    The number of grains is effectively the number of sub-patches that we create for the
    model to use for navigation. Right now, we have informally agreed to limit them
    to two for simplicity. This will not be the case in the future

greyscale: bool
    Whether the image should be converted to greyscale

<a name="map.Map.full_view"></a>
#### full\_view

```python
 | full_view(position: Tuple[int])
```

This fuction will get the image that covers the entirety of
the agent's FOV, This is so that we can see everything 
that the agent has seen, as well as use this as an intermediate
function later when we're getting individual grains

<a name="map.Map.get_fov"></a>
#### get\_fov

```python
 | get_fov(position: Tuple[int])
```

Gets the grains for the rover at the given position and returns them

Params
------

position: Tuple[int]
    The position that the rover is sitting at. This will be adgusted so that the
    rover is virtually in between four points.

Returns
-------

List[List[Image.Image]]
    Returns the agent's field of view as a 2d array of images in the format
    <visual_top_left>,    <visual_top_right>
    <visual_bottom_left>, <visual_buttom_right>

    Visual refers to the fact that <visual_top_left> looks like it's in the
    top left, but according to PIL it's actually in the bottom left

<a name="map.Map.clean_directions"></a>
#### clean\_directions

```python
 | clean_directions(c: List[List[Tuple[int]]])
```

This function makes sure that the agent wouldn't be able to see out
of the map if it moved to these positions. This is to prevent the
agent from running off the map.

For right now we assume that there are 4 grains being passed. Because
we're only checking 4 positions, it's easier and more efficient to
just unroll the lop and write them all out individually

Params
------
c: List[List[int]]
    A 2d list of positions that will be checked against the map.

Returns
-------
List[List[bool]]
    A 2d list of booleans that tell whether a given direction is valid

<a name="networks"></a>
# networks

<a name="networks.create_network"></a>
#### create\_network

```python
create_network(image_size: Tuple)
```

Create the CNN-AE based on the input image size. Only square grey scale images allowed.
The input and output sizes for the network are the same.

**Arguments**:

- `image_size` - Tuple
  Image size as Tuple of (H,W,C)
  

**Returns**:

  A tensorflow model for the network.

<a name="networks.create_network32"></a>
#### create\_network32

```python
create_network32()
```

Create the network for input size of (32, 32, 1)

<a name="networks.create_network64"></a>
#### create\_network64

```python
create_network64()
```

Create the network for input size of (64, 64, 1)

<a name="networks.create_network128"></a>
#### create\_network128

```python
create_network128()
```

Create the network for input size of (128, 128, 1)

<a name="brain"></a>
# brain

<a name="brain.Brain"></a>
## Brain Objects

```python
class Brain()
```

<a name="brain.Brain.__init__"></a>
#### \_\_init\_\_

```python
 | __init__(memory: BaseMemory, img_size: Tuple, nov_thresh: float = 0.25, novelty_loss_type: str = 'MSE', train_epochs_per_iter: int = 1, learning_rate: float = 0.001)
```

Initializes the Brain by creating CNN and AE

**Arguments**:

- `memory` - BaseMemory
  A memory object that implements BaseMemory  (such as PriorityBasedMemory)
- `img_size` - Tuple
  The image size of each grain from the agent's field of view
  nov_thresh : float
  (Currently deprecated). The novelty cutoff used in training
- `novelty_loss_type` - str
  A string indicating which novelty function to use (MSE or MAE)
- `train_epochs_per_iter` - int
  Number of epochs to train for in a single training session
- `learning_rate` - float
  Learning rate for neural network optimizer

<a name="brain.Brain.get_name"></a>
#### get\_name

```python
 | get_name()
```

Returns the full descriptive name of the brain object.

Returns
    The name of the brain object as a string

<a name="brain.Brain.add_grains"></a>
#### add\_grains

```python
 | add_grains(grains: List[List[Image.Image]])
```

Add new grains to memory

Params:
grains: List[List[Image.Image]]
2D List of new grains

**Returns**:

  2D List of novelty for new grains

<a name="brain.Brain.evaluate_grains"></a>
#### evaluate\_grains

```python
 | evaluate_grains(grains: List[List[Image.Image]])
```

Evaluate a list of grains

Params:
grains: List[List[Image.Image]]
2D List of new grains

**Returns**:

  2D List of novelty for new grains, and 2D list for reconstructed grains

<a name="brain.Brain.learn_grains"></a>
#### learn\_grains

```python
 | learn_grains()
```

Train the network to learn new features from memory

**Returns**:

  The current average loss from the last training epoch

