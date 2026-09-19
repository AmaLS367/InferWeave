"""Standalone executable script for WAN video generation server."""

import sys

from inferweave.workers.wan import main

if __name__ == "__main__":
    main(sys.argv[1:])
