"""CLI entrypoint for FLUX worker: python3 -m inferweave_worker.flux."""

import sys

from inferweave.workers.flux import main

if __name__ == "__main__":
    main(sys.argv[1:])
