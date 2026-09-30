import unittest
import os
import sys
import pandas as pd

# Add project root to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from utils.sql_validator import validate_sql_safety
from utils.etl_tools import validate_python_ast, ETLTools
from Models.schema import AgentSchema, RouterSchema, DataAgentSchema


class TestDataAgentImprovements(unittest.TestCase):

    def test_sql_validator_safe_queries(self):
        safe_queries = [
            "SELECT * FROM users LIMIT 10",
            "SELECT user_id, AVG(rating) FROM ratings GROUP BY user_id ORDER BY 2 DESC",
            "WITH high_rides AS (SELECT * FROM rides WHERE fare > 50) SELECT * FROM high_rides;",
            "SELECT u.first_name, p.amount FROM users u JOIN payments p ON u.user_id = p.user_id;"
        ]
        for q in safe_queries:
            is_safe, reason = validate_sql_safety(q)
            self.assertTrue(is_safe, f"Expected safe query: {q}, got: {reason}")

    def test_sql_validator_blocks_dangerous_queries(self):
        dangerous_queries = [
            "DROP TABLE users;",
            "DELETE FROM payments WHERE amount < 10;",
            "UPDATE rides SET status = 'completed';",
            "INSERT INTO ratings (driver_id) VALUES (1);",
            "ALTER TABLE vehicles DROP COLUMN color;",
            "TRUNCATE TABLE rides;",
            "SELECT * FROM users; DROP TABLE users;",
            "SELECT pg_sleep(10);"
        ]
        for q in dangerous_queries:
            is_safe, reason = validate_sql_safety(q)
            self.assertFalse(is_safe, f"Expected dangerous query to be blocked: {q}")

    def test_python_ast_sandbox_safety(self):
        safe_code = """
import pandas as pd
import numpy as np

df = pd.DataFrame({'a': [1, 2, 3], 'b': [10, 20, 30]})
df['c'] = df['a'] * df['b']
"""
        is_safe, reason = validate_python_ast(safe_code)
        self.assertTrue(is_safe, f"Expected safe code, got: {reason}")

    def test_python_ast_sandbox_blocks_os_and_eval(self):
        unsafe_snippets = [
            "import os\nos.system('dir')",
            "import subprocess\nsubprocess.run(['ls'])",
            "import sys\nsys.exit()",
            "eval('2 + 2')",
            "open('secret.txt', 'w').write('hacked')"
        ]
        for code in unsafe_snippets:
            is_safe, reason = validate_python_ast(code)
            self.assertFalse(is_safe, f"Expected unsafe code to be blocked: {code}")

    def test_etl_execution_sandbox(self):
        etl = ETLTools()
        # Test safe execution
        result_safe = etl.execute_code("df = pd.DataFrame({'x': [1, 2, 3]})")
        self.assertIn("successfully", result_safe)

        # Test blocked execution
        result_blocked = etl.execute_code("import os\nos.system('whoami')")
        self.assertIn("blocked", result_blocked.lower())


if __name__ == '__main__':
    unittest.main()
