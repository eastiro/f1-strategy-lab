from __future__ import annotations

import argparse

from .config import DEFAULT_COUNTRY, DEFAULT_SESSION, DEFAULT_YEAR
from .openf1_client import fetch_session_bundle


def main() -> None:
    parser = argparse.ArgumentParser(description="Download OpenF1 race data.")
    parser.add_argument("--year", type=int, default=DEFAULT_YEAR)
    parser.add_argument("--country", default=DEFAULT_COUNTRY)
    parser.add_argument("--session", default=DEFAULT_SESSION)
    args = parser.parse_args()

    saved = fetch_session_bundle(args.year, args.country, args.session)
    for name, path in saved.items():
        print(f"{name}: {path}")


if __name__ == "__main__":
    main()

