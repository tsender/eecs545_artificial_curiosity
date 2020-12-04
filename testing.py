from PIL import Image
import unittest

from experience import Experience
from memory import PriorityBasedMemory, ListBasedMemory
from artificial_curiosity_types import Artificial_Curiosity_Types as act
from map import Map

class ExperienceTest(unittest.TestCase):
    # Testing positive comparisons
    def test_positive(self):
        self.assertTrue(Experience(1, 0) > Experience(0, 0))
        self.assertTrue(Experience(1, 0) > Experience(0, 0))
        self.assertTrue(Experience(1, 0) >= Experience(0, 0))
        self.assertTrue(Experience(1, 0) >= Experience(1, 0))
        self.assertTrue(Experience(1, 0) == Experience(1, 0))
        self.assertTrue(Experience(1, 0) <= Experience(1, 0))
        self.assertTrue(Experience(0, 0) <= Experience(1, 0))
        self.assertTrue(Experience(0, 0) < Experience(1, 0))
        self.assertTrue(Experience(0, 0) != Experience(1, 0))

    # Testing negative comparisons
    def test_negative(self):
        self.assertFalse(Experience(0, 0) > Experience(1, 0))
        self.assertFalse(Experience(0, 0) >= Experience(1, 0))
        self.assertFalse(Experience(0, 0) == Experience(1, 0))
        self.assertFalse(Experience(1, 0) <= Experience(0, 0))
        self.assertFalse(Experience(1, 0) < Experience(0, 0))
        self.assertFalse(Experience(0, 0) != Experience(0, 0))


class MemoryTest(unittest.TestCase):
    def test_priority_based_memory(self):
        m = PriorityBasedMemory(5)
        for i in range(6):
            m.push(Experience(i, None))

        for i in m.as_list():
            self.assertNotEqual(i.novelty, 0)

    def test_list_based_memory(self):
        m = ListBasedMemory(5)
        for i in range(6):
            m.push(Experience(i, None))

        for i in m.as_list():
            self.assertNotEqual(i.novelty, 0)

class MapTest(unittest.TestCase):
    def test_init(self):
        m = Map("data/x.jpg", 30, 2)

        #  WARN: This might need to be addressed in future versions
        self.assertEqual(m.sqrtGrains, 4 ** (1/2))

        self.assertEqual(m.fov, 30)
        self.assertIsInstance(m.img, act.Grain)


    # WARN: Updated the map and now need to redo these test cases
    def test_map_exceptions(self):
        m = Map("data/x.jpg", 30, 4)
        width, height = m.img.size

        # Removed the Raise Exception implementation because it should not
        # be necessary if you call clean_directions first 
        # with self.assertRaises(Exception):
        #     m.get_fov((0, 0))

        # with self.assertRaises(Exception):
        #     m.get_fov((width-30, 0))

        # with self.assertRaises(Exception):
        #     m.get_fov((0, height-30))


        # with self.assertRaises(Exception):
        #     m.get_fov((29, 29))

        # with self.assertRaises(Exception):
        #     m.get_fov((width-1, height-1))

        m.get_fov((width/2, height/2))
        m.get_fov((30, 30))

    # WARN: Need to redo these test cases because map was redone
    def test_directions(self):

        m = Map("data/x.jpg", 30, 4)

        # self.assertEqual(m.clean_directions([
        #     [(0, 0), (28, 300), (30, 29)]
        # ]), [False, False, False])

        # self.assertEqual(m.clean_directions([(800, 534), (770, 300), (50, 505)]), [False, False, False])

        # self.assertEqual(m.clean_directions([(30, 30), (600, 504), (500, 500)]), [True, True, True])
        


if __name__=="__main__":
    unittest.main()
    
   
