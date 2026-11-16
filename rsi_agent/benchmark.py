"""Launch the pinned upstream runner with its official benchmark environment."""

import argparse
import subprocess
import sys
from pathlib import Path

UPSTREAM = "https://github.com/ZiwenWang-code/RSIAgent.git"
REVISION = "a9e56263f6deaa493496ad6b155fe24bf131bc12"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkout", type=Path)
    parser.add_argument("action", choices=["setup", "osworld", "ale"])
    args, forwarded = parser.parse_known_args()
    checkout = args.checkout.resolve()

    if args.action == "setup":
        if forwarded:
            parser.error("setup does not take runner arguments")
        subprocess.run(["git", "clone", "--no-checkout", UPSTREAM, str(checkout)], check=True)
        subprocess.run(["git", "checkout", "--detach", REVISION], cwd=checkout, check=True)
        return

    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=checkout, text=True).strip()
    if commit != REVISION:
        parser.error(f"expected upstream revision {REVISION}, got {commit}")
    if forwarded[:1] == ["--"]:
        forwarded = forwarded[1:]
    command = [sys.executable, f"run_{args.action}.py", *forwarded]
    raise SystemExit(subprocess.run(command, cwd=checkout).returncode)


if __name__ == "__main__":
    main()
