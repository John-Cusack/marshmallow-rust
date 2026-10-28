"""Mirror test guard: the suite must exercise the compiled extension.

`python -m pytest` and `python -c` put cwd on `sys.path`, so running from
the source checkout shadows the installed wheel with a package whose
`_core` is missing (silent pure-Python fallback). Fail loudly instead.
"""

import os

import marshmallow_rust as mr


def _is_source_tree(mod):
    here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    return os.path.abspath(mod.__file__ or "").startswith(here + os.sep)


if mr._core is None or _is_source_tree(mr):
    raise ImportError(
        "mirror tests must import the installed marshmallow_rust wheel "
        "(with the compiled _core extension), but the source checkout "
        f"shadowed it: {mr.__file__}. Re-run from a neutral cwd, e.g. "
        "`cd /tmp && testvenv/bin/python -m pytest <mirror>/tests/`."
    )
