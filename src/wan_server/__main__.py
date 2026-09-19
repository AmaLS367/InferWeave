"""CLI entrypoint for wan_server: python3 -m wan_server."""

import sys

from inferweave.workers.wan import main

if __name__ == "__main__":
    main(sys.argv[1:])
