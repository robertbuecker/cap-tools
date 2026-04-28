"""Compatibility launcher for the removed cell clustering tool.

Unit-cell clustering and merge/finalize automation have moved out of cap-tools.
The retained part of the old workflow is the finalization comparison viewer.
"""

from cap_tools.finalization_viewer import main_cli


if __name__ == "__main__":
    main_cli()
