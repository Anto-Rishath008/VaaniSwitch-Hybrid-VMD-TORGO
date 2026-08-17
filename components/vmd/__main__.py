"""Allow ``python -m components.vmd`` execution."""

from components.vmd.cli import main

if __name__ == "__main__":
    raise SystemExit(main())
