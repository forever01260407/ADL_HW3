import unittest
from datetime import datetime, timezone
from time_budget import deadline_reached


class DeadlineTests(unittest.TestCase):
    def test_no_deadline_does_not_impose_a_daily_curfew(self):
        self.assertFalse(deadline_reached(None, datetime(2026, 11, 17, 23, 59, tzinfo=timezone.utc)))

    def test_offset_and_exact_deadline(self):
        deadline = "2026-11-09T22:50:00+08:00"
        self.assertFalse(deadline_reached(deadline, datetime(2026, 11, 9, 14, 49, tzinfo=timezone.utc)))
        self.assertTrue(deadline_reached(deadline, datetime(2026, 11, 9, 14, 50, tzinfo=timezone.utc)))
        self.assertTrue(deadline_reached(deadline, datetime(2026, 11, 17, tzinfo=timezone.utc)))

    def test_ambiguous_timezone_is_rejected(self):
        with self.assertRaises(ValueError):
            deadline_reached("2026-11-09T22:50:00")


if __name__ == "__main__":
    unittest.main()
