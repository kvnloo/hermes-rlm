"""Stats paths are filesystem names, not unescaped SQLite URI components."""
import json
import os
from contextlib import closing
from pathlib import Path
import sqlite3
import tempfile
import unittest
from unittest.mock import patch

import paper_matrix as matrix


class StatsUriTests(unittest.TestCase):
    def test_existing_stats_db_in_valid_special_character_directories(self):
        with tempfile.TemporaryDirectory() as directory:
            for name in ('ordinary', 'run#one', 'run?one', 'run%23one', 'café space'):
                if '?' in name and os.name == 'nt':
                    continue  # Windows does not permit '?' in filesystem names.
                with self.subTest(name=name):
                    home = Path(directory) / name
                    home.mkdir()
                    path = home / 'state.db'
                    with closing(sqlite3.connect(path)) as db, db:
                        db.execute('CREATE TABLE sessions (id, input_tokens, output_tokens, api_call_count, tool_call_count)')
                        db.execute('INSERT INTO sessions VALUES (?, ?, ?, ?, ?)', ('s', 7, 3, 2, 1))
                        db.execute('CREATE TABLE messages (tool_calls)')
                        db.execute('INSERT INTO messages VALUES (?)', (json.dumps([{'function': {'name': 'tool_search'}}]),))
                    original = path.read_bytes()
                    connect = sqlite3.connect

                    def checked_connection(*args, **kwargs):
                        db = connect(*args, **kwargs)
                        with self.assertRaisesRegex(sqlite3.OperationalError, 'readonly'):
                            db.execute('CREATE TABLE unexpected_write (x)')
                        return db

                    with patch.object(matrix.sqlite3, 'connect', side_effect=checked_connection):
                        result = matrix.hermes_stats(home)
                    self.assertEqual(result, dict(session_ids=['s'], discovery_traces=['tool_search'],
                                                  host_root_input_tokens=7, host_root_output_tokens=3,
                                                  host_api_calls=2, tool_calls=1))
                    self.assertEqual(path.read_bytes(), original)

    def test_missing_stats_db_is_not_created(self):
        with tempfile.TemporaryDirectory() as directory:
            home = Path(directory) / 'missing#home'
            home.mkdir()
            self.assertEqual(matrix.hermes_stats(home), dict(session_ids=[], discovery_traces=[]))
            self.assertFalse((home / 'state.db').exists())
