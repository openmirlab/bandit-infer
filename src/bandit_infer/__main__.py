"""Run the package CLI with ``python -m bandit_infer``.

This is the module counterpart of the installed ``bandit-infer`` command.
Reads: cli.
"""

from .cli import main

raise SystemExit(main())
