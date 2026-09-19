"""CLI dispatcher for inferweave_worker: python3 -m inferweave_worker <subcommand>."""

import sys


def main() -> None:
    if len(sys.argv) < 2 or sys.argv[1] in ("-h", "--help"):
        print("Usage: python3 -m inferweave_worker [flux|wan] [options...]")
        sys.exit(0)

    subcommand = sys.argv[1]
    remaining = sys.argv[2:]

    if subcommand == "flux":
        from inferweave.workers.flux import main as flux_main

        flux_main(remaining)
    elif subcommand in ("wan", "wan-video"):
        from inferweave.workers.wan import main as wan_main

        wan_main(remaining)
    else:
        print(f"Unknown subcommand: {subcommand}. Available: flux, wan")
        sys.exit(1)


if __name__ == "__main__":
    main()
