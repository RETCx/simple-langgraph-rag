import unittest
from contextlib import redirect_stdout
from io import StringIO
from unittest.mock import patch

from src.main import create_initial_state, run_graph


class FakeGraph:
    def __init__(self):
        self.state = {}

    def stream(self, state, stream_mode):
        assert stream_mode == "updates"
        self.state = dict(state)
        update = {"results": [(object(), 0.9)]}
        self.state.update(update)
        yield {"retrieve": update}
        update = {"has_context": True}
        self.state.update(update)
        yield {"check_context": update}
        update = {"answer": "test answer"}
        self.state.update(update)
        yield {"answer": update}

class TestCLIProgress(unittest.TestCase):
    def test_reports_real_graph_nodes_and_returns_merged_state(self):
        output = StringIO()
        state = create_initial_state()
        state["question"] = "test question"

        with patch("src.main.get_graph", return_value=FakeGraph()):
            with redirect_stdout(output):
                result = run_graph(state)

        text = output.getvalue()
        self.assertIn("Retrieval", text)
        self.assertIn("Context check", text)
        self.assertIn("Answer", text)
        self.assertEqual(result["answer"], "test answer")

    def test_quiet_mode_hides_progress(self):
        output = StringIO()
        with patch("src.main.get_graph", return_value=FakeGraph()):
            with redirect_stdout(output):
                run_graph(create_initial_state(), show_progress=False)

        self.assertEqual(output.getvalue(), "")


if __name__ == "__main__":
    unittest.main()
