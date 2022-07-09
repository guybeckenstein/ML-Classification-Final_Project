import unittest
from csv_classification import get_csv_file

class TestCSV(unittest.TestCase):
    def test_csv_exists(self):
        self.assertEqual(get_csv_file(), 'time_off_data_train.csv')

if __name__ == '__main__':
    unittest.main()