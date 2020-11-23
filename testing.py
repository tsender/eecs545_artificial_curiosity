from Experience import Experience
from Memory import Memory
from ArtificialCuriosityTypes import ArtificialCuriosityTypes as ac

def test_experience():
    # Testing positive comparisons
    assert Experience(1, 0, 0) > Experience(0, 0, 0)
    assert Experience(1, 0, 0) >= Experience(0, 0, 0)
    assert Experience(1, 0, 0) >= Experience(1, 0, 0)
    assert Experience(1, 0, 0) == Experience(1, 0, 0)
    assert Experience(1, 0, 0) <= Experience(1, 0, 0)
    assert Experience(0, 0, 0) <= Experience(1, 0, 0)
    assert Experience(0, 0, 0) < Experience(1, 0, 0)
    assert Experience(0, 0, 0) != Experience(1, 0, 0)

    # Testing negative comparisons
    assert not Experience(0, 0, 0) > Experience(1, 0, 0)
    assert not Experience(0, 0, 0) >= Experience(1, 0, 0)
    assert not Experience(0, 0, 0) == Experience(1, 0, 0)
    assert not Experience(1, 0, 0) <= Experience(0, 0, 0)
    assert not Experience(1, 0, 0) < Experience(0, 0, 0)
    assert not Experience(0, 0, 0) != Experience(0, 0, 0)


def test_memory():
    m = Memory(5)

    assert m.maxLength == 5

    for i in range(6):
        m.push(Experience(i, None, None))

    for i in m.memIter():
        assert not i.novelty == 0

if __name__=="__main__":
    test_experience()
    test_memory()
    
   
