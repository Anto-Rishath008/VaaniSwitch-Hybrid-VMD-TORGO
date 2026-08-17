"""Allow ``python -m vaaniswitch.m1_vmd`` execution."""

from vaaniswitch.m1_vmd.cli import main

if __name__ == "__main__":
    raise SystemExit(main())
