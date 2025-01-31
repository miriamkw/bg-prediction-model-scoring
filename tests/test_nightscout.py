import unittest
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import pandas as pd
import sys

# Add the parent directory to Python path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from glupredkit.parsers.nightscout import Parser

class TestNightscoutParser(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        """Set up test data directory and load files."""
        # Test files are in the same directory as this test script
        cls.test_dir = Path(__file__).parent
        
        # Define test period
        cls.start_date = datetime(2024, 10, 26, 0, 0, tzinfo=timezone.utc)
        cls.end_date = datetime(2024, 10, 28, 0, 0, tzinfo=timezone.utc)

    def test_parser(self):
        """Test the parser with our test data files."""
        # Initialize parser
        parser = Parser()
        
        print(f"Test directory: {self.test_dir}")
        print(f"Looking for files:")
        print(f"- {self.test_dir / 'nightscout_profiles.json'}")
        print(f"- {self.test_dir / 'nightscout_treatments.json'}")
        print(f"- {self.test_dir / 'nightscout_entries.json'}")
        
        # Verify test files exist
        assert (self.test_dir / "nightscout_profiles.json").exists(), "Profiles file not found"
        assert (self.test_dir / "nightscout_treatments.json").exists(), "Treatments file not found"
        assert (self.test_dir / "nightscout_entries.json").exists(), "Entries file not found"
        
        # Run parser
        df = parser(
            start_date=self.start_date,
            end_date=self.end_date,
            username="dummy_url",
            password="dummy_password",
            test_mode=True,
            test_data_dir=str(self.test_dir)  # Convert Path to string
        )
        
        # Save output to a snapshot file if it doesn't exist
        snapshot_file = self.test_dir / "nightscout_expected_output.csv"
        if not snapshot_file.exists():
            df.to_csv(snapshot_file)
            print(f"\nCreated initial snapshot at {snapshot_file}")
            print("Please verify the output manually before proceeding with tests")
            return
        
        # Load expected output
        expected_df = pd.read_csv(snapshot_file, parse_dates=['date'], index_col='date')
        
        # Compare output with expected
        pd.testing.assert_frame_equal(df, expected_df)

if __name__ == '__main__':
    unittest.main()