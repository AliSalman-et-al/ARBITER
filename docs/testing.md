# Testing

## Agent notes

Use a command timeout that matches the expected test scope. The full unit suite can exceed short default tool limits, so run `uv run pytest tests/unit -q` with at least a 300-second timeout before treating a timeout as a test failure.
