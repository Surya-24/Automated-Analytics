import unittest
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from langchain_core.messages import HumanMessage
from Models.schema import DataAgentSchema, RouterSchema, AgentSchema
from utils.database import DatabaseUtil
from agents.data_agent import router_node, route_edge


class TestAgentIntegration(unittest.TestCase):

    def test_router_heuristic_and_state(self):
        state = DataAgentSchema(
            messages=[HumanMessage(content="Extract pokemon data from API and save to csv")],
            route_response=""
        )
        self.assertEqual(len(state.messages), 1)
        self.assertEqual(state.route_response, "")
        
        # Test routing decision function
        state.route_response = "etl"
        self.assertEqual(route_edge(state), "etl_node")

        state.route_response = "sql"
        self.assertEqual(route_edge(state), "sql_node")

    def test_database_util_graceful_error_handling(self):
        db = DatabaseUtil({
            "host": "invalid_host_xyz",
            "port": 5432,
            "user": "fake_user",
            "password": "wrong_password",
            "dbname": "fake_db"
        })
        # Verifies database util does not crash the app on connection failure
        rows, err = db.execute_sql("SELECT * FROM users;")
        self.assertIsNone(rows)
        self.assertIsNotNone(err)

    def test_agent_schema_defaults(self):
        schema = AgentSchema(user_question="Show users")
        self.assertEqual(schema.retry_count, 0)
        self.assertEqual(schema.error_trace, "")
        self.assertIsNone(schema.structured_data)


if __name__ == '__main__':
    unittest.main()
