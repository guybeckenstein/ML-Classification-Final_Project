import unittest
from csv_classification import get_csv_file, load_dataset

class TestCSV(unittest.TestCase):
    
    @classmethod
    def SetUpClass(cls):
        pass
    
    @classmethod
    def TearDownClass(cls):
        pass
    
    def SetUp(self):
        self.csv_file = get_csv_file()
        
    def TearDown(self):
        pass
    
    def test_csv_exists(self):
        self.assertEqual(self.csv_file, 'time_off_data_train.csv')
    
    def test_load_csv(self):
        with self.assertRaises(FileNotFoundError):
            load_dataset('')
        with self.assertRaises(FileNotFoundError):
            load_dataset('file.csv')

if __name__ == '__main__':
    unittest.main()
