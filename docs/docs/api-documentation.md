---
menu: main
title: API Documentation
---

<a name="Memory"></a>
# Memory

<a name="Memory.Memory"></a>
## Memory Objects

```python
class Memory()
```

This class abstracts away the specific implementation of an autonomous agent's memory unit

Attributes

None


Methods

`__init__(maxLength: int = 30)`  
    Initializes the memory unit with a default capacity of 30 Experience
`push(data: Experience)`  
    Adds an Experience to the memory unit. If the memory is full, it forgets the Experience that had the gratest act.Novelty
`memIter() -> Generator`  
    Creates an iterator that can be used to iterate over Experience instances

<a name="Memory.Memory.__init__"></a>
#### \_\_init\_\_

```python
 | __init__(maxLength: int = 30)
```

### Parameters

> maxLength : int  
>    > The maximum number of experiences(Experience) that the memory unit can contain

### Returns

> Memory

<a name="Memory.Memory.push"></a>
#### push

```python
 | push(data: Experience)
```

### Parameters

> data : Experience  
>    > Adds an experience (Experience) to memory. Once full, experiences that are less novel (lower values of act.Novelty) will be forgotten as new experiences are added

Returns

None

<a name="Memory.Memory.memIter"></a>
#### memIter

```python
 | memIter() -> Generator
```

### Parameters

> None

### Returns

> Generator
>    > An iterator that operates over all experiences (Experience) in memory

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

**Arguments**:

- `img` - the PIL image object
  
- `Returns` - A boolean indicating whether img is greyscale or not

<a name="map_helpers.is_valid_position"></a>
#### is\_valid\_position

```python
is_valid_position(img, fov, position)
```

Parameters: The image, fov and position passed to get_fov in map.py

Returns: A boolean indicating whether it is a valid position or not, i.e. atleast [fov] pixels away from image edge

<a name="map_helpers.find_sitting_pixels"></a>
#### find\_sitting\_pixels

```python
find_sitting_pixels(position, width, height)
```

Parameters: the position of the rover and width and height of the image

Returns: A dictionary that represents the 4 pixels that the rover is "sitting" on
of the form {"position":(x, y), "above": (e, f), "right": (g, h), "top_right": (a, s)}

<a name="map_helpers.find_coordinates"></a>
#### find\_coordinates

```python
find_coordinates(rover_position, fov, width, height)
```

Parameters: A dictionary of the pixels that the rover is "sitting" on, i.e. the one returned by find_sitting_pixels,
The fov, width, height of image

Returns: A list of the cropping coordinates for each of the (max) four grains

<a name="Experience"></a>
# Experience

<a name="Experience.Experience"></a>
## Experience Objects

```python
class Experience()
```

A type for a measure of novelty, a feature vector, and an associated image. This is meant to be used by the Memory and the Brain



Attributes
----------
self.novelty : act.Novelty
    A float that represents the movelty of the Experience
self.featureVector : List[float]
    A vector that holds the different features that represent this memory
self.grain : act.Image
    An image that will show us what the machine remembers


Methods
-------
__init__(nov: act.Novelty, fVect: List[float], grn: act.Image)
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

<a name="Experience.Experience.__init__"></a>
#### \_\_init\_\_

```python
 | __init__(nov: act.Novelty, fVect: List[float], grn: act.Image)
```

Parameters
__________
nov : act.Novelty
    The measure of novelty, expressed as a float
fVect : List[float]
    A feature vector expressing the grain (image)
grn: act.Image
    A grain (image) to be remembered. This exists so we can reference it later

Returns
_______
Experience

<a name="Experience.Experience.__lt__"></a>
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

<a name="Experience.Experience.__le__"></a>
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

<a name="Experience.Experience.__gt__"></a>
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

<a name="Experience.Experience.__ge__"></a>
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

<a name="Experience.Experience.__eq__"></a>
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

<a name="Experience.Experience.__ne__"></a>
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

<a name="Experience.Experience.__str__"></a>
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

<a name="testing"></a>
# testing

<a name="testing.MapTest"></a>
## MapTest Objects

```python
class MapTest(unittest.TestCase)
```

<a name="testing.MapTest.test_directions"></a>
#### test\_directions

```python
 | test_directions()
```

Aravind, please finish this test case

<a name="ArtificialCuriosityTypes"></a>
# ArtificialCuriosityTypes

<a name="ArtificialCuriosityTypes.addType"></a>
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

<a name="map"></a>
# map

<a name="map.Map"></a>
## Map Objects

```python
class Map()
```

Map class that creates instances of the terrain map that
the model will work on

__init__ : 
	initialize an instance of the given map and store fov and sqrtGrains

get_fov(position): 
	returns a list of grains (sub-images) with radius fov given the position of the model on the map

clean_directions(coordinates): 
	return a boolean list that corresponds to whether the model can move to the coordinates specified by the argument

<a name="map.Map.__init__"></a>
#### \_\_init\_\_

```python
 | __init__(filepath: str, fov: int, sqrtGrains: int)
```

**Arguments**:

  
- `filepath` - the input terrain map -- can be a jpg or png
- `fov` - radius of the field-of-view
- `sqrtGrains` - The square root of the number of grains (sub-squares) in the fov
  
  

**Returns**:

  
  Initializes an object with:
  
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

Parameters: A list of tuples that represent coordinates

Returns: A boolean array corresponding to each coordinate that indicates whether the model can move to that coordinate

