from Experience import Experience
from Memory import Memory
from ArtificialCuriosityTypes import ArtificialCuriosityTypes as ac
from Map import Map
from PIL import Image
import unittest

class ExperienceTest(unittest.TestCase):
    # Testing positive comparisons
    def test_positive(self):
        self.assertTrue(Experience(1, 0, 0) > Experience(0, 0, 0))
        self.assertTrue(Experience(1, 0, 0) > Experience(0, 0, 0))
        self.assertTrue(Experience(1, 0, 0) >= Experience(0, 0, 0))
        self.assertTrue(Experience(1, 0, 0) >= Experience(1, 0, 0))
        self.assertTrue(Experience(1, 0, 0) == Experience(1, 0, 0))
        self.assertTrue(Experience(1, 0, 0) <= Experience(1, 0, 0))
        self.assertTrue(Experience(0, 0, 0) <= Experience(1, 0, 0))
        self.assertTrue(Experience(0, 0, 0) < Experience(1, 0, 0))
        self.assertTrue(Experience(0, 0, 0) != Experience(1, 0, 0))

    # Testing negative comparisons
    def test_negative(self):
        self.assertFalse(Experience(0, 0, 0) > Experience(1, 0, 0))
        self.assertFalse(Experience(0, 0, 0) >= Experience(1, 0, 0))
        self.assertFalse(Experience(0, 0, 0) == Experience(1, 0, 0))
        self.assertFalse(Experience(1, 0, 0) <= Experience(0, 0, 0))
        self.assertFalse(Experience(1, 0, 0) < Experience(0, 0, 0))
        self.assertFalse(Experience(0, 0, 0) != Experience(0, 0, 0))


class MemoryTest(unittest.TestCase):
    def test_init(self):
        m = Memory(5)
        self.assertEqual(m.maxLength, 5)

    def test_push(self):
        m = Memory(5)
        for i in range(6):
            m.push(Experience(i, None, None))

        for i in m.memIter():
            self.assertNotEqual(i.novelty, 0)

class MapTest(unittest.TestCase):
    def test_init(self):
        m = Map("x.jpg", 30, 4)

        #  WARN: This might need to be addressed in future versions
        # self.assertEqual(m.sqrtGrains, 4 ** (1/2))

        self.assertEqual(m.fov, 30)
        self.assertIsInstance(m.img, Image.Image)

    def test_map_exceptions(self):
        m = Map("x.jpg", 30, 4)
        width, height = m.img.size

        with self.assertRaises(Exception):
            m.get_fov((0, 0))

        with self.assertRaises(Exception):
            m.get_fov((width-1, 0))

        with self.assertRaises(Exception):
            m.get_fov((0, height-1))


        with self.assertRaises(Exception):
            m.get_fov((29, 29))

        with self.assertRaises(Exception):
            m.get_fov((width-1, height-1))

        m.get_fov((width/2, height/2))
        m.get_fov((30, 30))

    def test_directions(self):
        # Aravind, please finish this test case
        


if __name__=="__main__":
    unittest.main()
    
   
