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

__init__(maxLength: int = 30)
    Initializes the memory unit with a default capacity of 30 Experience
push(data: Experience)
    Adds an Experience to the memory unit. If the memory is full, it forgets the Experience that had the gratest act.Novelty
memIter() -> Generator
    Creates an iterator that can be used to iterate over Experience instances

<a name="Memory.Memory.__init__"></a>
#### \_\_init\_\_

```python
 | __init__(maxLength: int = 30)
```

### Parameters

> maxLength : int  
    > The maximum number of experiences(Experience) that the memory unit can contain

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

