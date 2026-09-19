"""CLI entrypoint for WAN video worker: python3 -m inferweave_worker.wan."""

import sys

from inferweave.workers.wan import main

if __name__ == "__main__":
    main(sys.argv[1:])
