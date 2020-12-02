---
menu: main
title: API Documentation
---

<a name="experience"></a>
# experience

<a name="experience.Experience"></a>
## Experience Objects

```python
class Experience()
```

A type for a measure of novelty, a feature vector, and an associated image. This is meant to be used by the Memory and the Brain

Attributes
----------
self.novelty : float
    A float that represents the movelty of the Experience
self.featureVector : np.float32
    A numpy.float32 1D array that holds the different features that represent this memory
self.grain : Image.Image
    An image that will show us what the machine remembers


Methods
-------
__init__(nov: float, fVect: np.float32, grn: Image.Image)
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
 | __init__(nov: float, fVect: np.float32, grn: Image.Image)
```

Parameters
__________
nov : float
    The measure of novelty, expressed as a float
fVect : np.float32
    A numpy.float32 1D array that holds the different features that represent this grain
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

<a name="memory"></a>
# memory

<a name="memory.Memory"></a>
## Memory Objects

```python
class Memory()
```

This class abstracts away the specific implementation of an autonomous agent's memory unit

Attributes

None


Methods

`__init__(maxLength: int = 32)`  
    Initializes the memory unit with a default capacity of 32 Experiences
`push(data: Experience)`  
    Adds an Experience to the memory unit. If the memory is full, it forgets the Experience that had the smallest act.Novelty
`memList() -> List[Experience]`  
    Returns a list of Experience instances

<a name="memory.Memory.__init__"></a>
#### \_\_init\_\_

```python
 | __init__(maxLength: int = 32)
```

Parameters
---------
maxLength : int  
    The maximum number of experiences(Experience) that the memory unit can contain

Returns
-------
Memory

<a name="memory.Memory.push"></a>
#### push

```python
 | push(data: Experience)
```

### Parameters

> data : Experience  
>    > Adds an experience (Experience) to memory. Once full, experiences that are less novel (lower values of act.Novelty) will be forgotten as new experiences are added

Returns

None

<a name="memory.Memory.memList"></a>
#### memList

```python
 | memList() -> List[Experience]
```

### Parameters

> None

### Returns

> List[Experience]
>    > A list of Experience objects

<a name="engine"></a>
# engine

<a name="engine.plot_paths"></a>
#### plot\_paths

```python
plot_paths(map: Map, agent_lst: List[Agent], show: bool, save: bool, dirname: str)
```

Plots out the paths of the agents on the map

Params
------
map: Map
    A map that will be used to get the bounds and background for plotting

agent_lst: List[Agent]
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

<a name="engine.run_experiment"></a>
#### run\_experiment

```python
run_experiment(motivation_lst: List[Motivation], position_lst: List[Tuple[int]], map: Map, iterations: int, show: bool = True, saveGraph: bool = False, saveLocation: bool = True, dirname: str = None)
```

Runs an experiment on the motication given, then handles plotting and saving data. Agents are updated in a round-robin configuration, so each gets an ewual number of executions, and they all rpogress together.

Params
------
motivation_lst: List[Motivation]
    A list of Motivation class instances to be used as the drivers for agents. This must be the same length as position_lst

position_lst: List[Tuple[int]]
    A list of poitions for the agents to start out at (matching indices with elements from motivation_lst). It must be the same length as potivation_lst.

map: Map
    An instance of the Map class that will be used by the agent to handle directions, and for the plotting

iterations: int
    The number of steps that each agent should take.

show: bool=True
    Whether the graphs should be displayed

saveGraph: bool
    Whether the plots should be saved to the disk or not

saveLoction: bool
    Whether the agent's position should be save to the disk

dirname: str=None
    The directory in which the graphs will be stored


Returns
-------
None

<a name="engine.save_agent_data"></a>
#### save\_agent\_data

```python
save_agent_data(agent_lst: List[Agent], save: bool, dirname: str)
```

Save the path record of each agent as a csv file

Params:
------
agent_lst: List[Agent]
A list of agent whose path coordinates to be saved

save: bool
Save the agent's path data or not

dirname: str
THe directory name where the csv file will be saved

**Returns**:

  ------
  None

<a name="map_helpers"></a>
# map\_helpers

File that has helper functions for the Map class

1. is_grey_scale(img) : Checks whether the given image is greyscale or color
2. is_valid_position(img, fov, position): Checks whether the given position is valid or not
3. find_sitting_pixels(position, width, height): Finds the pixel that the rover is "sitting" on
4. find_coordinates(rover_position, fov, width, height): Finds the cropping coordinates that create grains

<a name="map_helpers.is_grey_scale"></a>
#### is\_grey\_scale

```python
is_grey_scale(img)
```

Parameter:
img: the PIL image object

**Returns**:

  A boolean indicating whether img is greyscale or not

<a name="map_helpers.is_valid_position"></a>
#### is\_valid\_position

```python
is_valid_position(img, fov, position)
```

**Arguments**:

  The image, fov and position passed to get_fov in map.py
  

**Returns**:

  A boolean indicating whether it is a valid position or not, i.e. atleast [fov] pixels away from image edge

<a name="map_helpers.find_sitting_pixels"></a>
#### find\_sitting\_pixels

```python
find_sitting_pixels(position, width, height)
```

**Arguments**:

- `position` - the position of the rover
- `width` - width (number of columns) of the image
- `height` - height (number of rows) of the image
  

**Returns**:

  A dictionary that represents the 4 pixels that the rover is "sitting" on

<a name="map_helpers.find_coordinates"></a>
#### find\_coordinates

```python
find_coordinates(rover_position, fov, width, height)
```

**Arguments**:

- `rover_position` - A dictionary of the pixels that the rover is "sitting" on, i.e. the one returned by find_sitting_pixels
- `fov` - the fov
- `width` - the width of the image
- `height` - the height of the image
  

**Returns**:

  A list of the cropping coordinates for each of the (max) four grains

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

<a name="__init__"></a>
# \_\_init\_\_

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

<a name="agent.Agent.step"></a>
#### step

```python
 | step()
```

This performs a simple step for the agent, moving it from one position to the next

<a name="testing"></a>
# testing

<a name="novelty"></a>
# novelty

The novelty module creates functions to evaluate the novelty (loss) which
are used in brain. The nevelty functions operate on the results of autoencoder
novelty func. 1: l1_norm loss
novelty func. 2: l2_norm loss
novelty func. 3:

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
 | __init__(nov_thresh: float, novelty_loss_type: str, max_train_epochs: int = 100)
```

Initializes the Brain by creating CNN and AE

Params
------
nov_thresh : float
    The novelty cutoff used in training
novelty_function: Callable
    The callback that will be used to determine the novelty for any given feature-vector/reconstructed-vector pairs
max_train_epochs: int
    Maximum number of training epochs (in case avg loss is still not at novelty thresh)

<a name="brain.Brain.add_grains"></a>
#### add\_grains

```python
 | add_grains(grains: List[Image.Image])
```

Add new grains to memory

Params:
grains: List[Image.Image]
List of new grains

**Returns**:

  List of novelty for new grains

<a name="brain.Brain.evaluate_novelty"></a>
#### evaluate\_novelty

```python
 | evaluate_novelty(grains: List[Image.Image])
```

Evaluate novelty of a list of grains

Params:
grains: List[Image.Image]
List of new grains

**Returns**:

  List of novelty for new grains

<a name="brain.Brain.learn_grains"></a>
#### learn\_grains

```python
 | learn_grains()
```

Train the AE to learn new features from memory

<a name="map"></a>
# map

<a name="map.Map"></a>
## Map Objects

```python
class Map()
```

Map class that creates instances of the terrain map that the model will work on

Methods

`__init__(filepath: str, fov: int, sqrtGrains: int` 
	initialize an instance of the given map and store fov and sqrtGrains

`get_fov(position: tuple)` 
	returns a list of grains (sub-images) with radius fov given the position of the model on the map

`clean_directions(coordinates: list)` 
	return a boolean list that corresponds to whether the model can move to the coordinates specified by the argument

<a name="map.Map.__init__"></a>
#### \_\_init\_\_

```python
 | __init__(filepath: str, fov: int, sqrtGrains: int)
```

**Arguments**:

  
- `filepath` - the stringpath containing the input terrain map -- can be a jpg or png
- `fov` - an int radius of the field-of-view
- `sqrtGrains` - The square root of the number of grains (sub-squares) in the fov -- an int
  
  

**Returns**:

  
  A Map object with:
  
  The image from filepath (in greyscale),
  fov,
  sqrtGrains

<a name="map.Map.get_fov"></a>
#### get\_fov

```python
 | get_fov(position: tuple)
```

Parameter:
position: Position of the rover on the map -- a tuple expected in (column, row)

**Returns**:

  
  A list of the grains that have been split from the img with radius fov, i.e.
  the legal squares that the rover can go to in fov steps

<a name="map.Map.clean_directions"></a>
#### clean\_directions

```python
 | clean_directions(coordinates: list)
```

Parameter:
coordinates: A list of tuples that represent coordinates

**Returns**:

  A boolean list corresponding to each coordinate that indicates whether the model can move to that coordinate

