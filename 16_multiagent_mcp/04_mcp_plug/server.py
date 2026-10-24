"""A tiny MCP plug. Read README.md in this folder, then read this file top to bottom.

    1. Name the plug.
    2. Publish a tool. The type hints are the arguments a host app can see.
    3. Publish a second tool on the same plug.
    4. mcp.run() waits for a host to call. This is not a website.

A real plug would publish search_the_packet and run_this_select.
The tools would change. These four steps would stay.
"""

from mcp.server.fastmcp import FastMCP

# The name a host shows when it connects.
mcp = FastMCP("demo")


@mcp.tool()
def add(a: int, b: int) -> int:
    """Add two numbers. The host can see that both arguments are integers."""
    return a + b


@mcp.tool()
def word_count(text: str) -> int:
    """Count the words in a piece of text."""
    return len(text.split())


if __name__ == "__main__":
    # Speak over standard input and output. The host is the other end of that pipe.
    mcp.run()
