import unittest
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from utils.observability import get_active_callbacks, get_langfuse_callback, check_langsmith_status
from Models.schema import AgentSchema, DataAgentSchema


class TestObservabilitySetup(unittest.TestCase):

    def test_observability_graceful_fallbacks(self):
        # When no keys are set, functions should return gracefully without crashing
        callbacks = get_active_callbacks()
        self.assertIsInstance(callbacks, list)

        langsmith_active = check_langsmith_status()
        self.assertIsInstance(langsmith_active, bool)

    def test_pydantic_schema_instrumentation(self):
        # Ensure Pydantic schemas work properly with Logfire instrumentation
        state = DataAgentSchema(route_response="sql")
        self.assertEqual(state.route_response, "sql")
        self.assertEqual(state.messages, [])


if __name__ == '__main__':
    unittest.main()
