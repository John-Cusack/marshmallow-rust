"""`marshmallow_rust`: Rust-accelerated `Schema` for marshmallow.

`Schema` subclasses `marshmallow.Schema` and overrides `dump` and `_do_load`
(the funnel for `load`, `loads`, and `validate`). On first use the schema
tree is compiled to a Rust-core plan (`compile.py`); every batch then
crosses the FFI boundary once. Schemas the compiler cannot prove (decorated
hooks, overridden entry points) or the no-extension interpreter take the
pure-Python upstream path with identical behavior.

Custom `Field` subclasses and per-value user callables do not force
fallback: they compile to opaque nodes that invoke the same callables
upstream would call.
"""

import copy

import marshmallow
from marshmallow import ValidationError
from marshmallow import fields as mm_fields
from marshmallow import validate as mm_validate
from marshmallow.utils import is_sequence_but_not_string

from . import compile as _compile
from .errors import materialize

try:
    from . import _core
except ImportError:  # pure-Python fallback without the extension
    _core = None

__version__ = "0.1.0"

__all__ = [
    "Schema",
    "fields",
    "validate",
    "ValidationError",
    "EXCLUDE",
    "INCLUDE",
    "RAISE",
    "__version__",
]

fields = mm_fields
validate = mm_validate
EXCLUDE = marshmallow.EXCLUDE
INCLUDE = marshmallow.INCLUDE
RAISE = marshmallow.RAISE


class Schema(marshmallow.Schema):
    """A `marshmallow.Schema` whose dump/load kernel runs in Rust.

    Compilation is lazy (first `dump`/`load`) so error timing for bad
    schemas (unknown `Nested` targets, invalid `only`/`exclude`) matches
    upstream, which only touches those paths at call time.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._rust_compiled = False
        self._rust_handle = None

    def __copy__(self):
        new = type(self).__new__(type(self))
        new.__dict__.update(self.__dict__)
        new._rust_compiled = False
        new._rust_handle = None
        return new

    def __deepcopy__(self, memo):
        new = type(self).__new__(type(self))
        memo[id(self)] = new
        new.__dict__.update(copy.deepcopy(self.__dict__, memo))
        new._rust_compiled = False
        new._rust_handle = None
        return new

    # -- compilation -------------------------------------------------

    def _rust_ensure(self):
        if not self._rust_compiled:
            self._rust_compiled = True
            if _core is not None:
                plan = _compile.compile_schema(self)
                if plan is not None:
                    self._rust_handle = _core.compile(plan)
        return self._rust_handle

    # -- entry points ------------------------------------------------

    def dump(self, obj, *, many=None):
        handle = self._rust_ensure()
        if handle is None:
            return super().dump(obj, many=many)
        many = self.many if many is None else bool(many)
        # Compiled schemas have no dump hooks, so processors are skipped
        # exactly as the all-empty check upstream would.
        return _core.dump(handle, obj, many)

    def _do_load(
        self, data, *, many=None, partial=None, unknown=None, postprocess=True
    ):
        handle = self._rust_ensure()
        if handle is None:
            return super()._do_load(
                data, many=many, partial=partial, unknown=unknown, postprocess=postprocess
            )
        many = self.many if many is None else bool(many)
        unknown = self.unknown if unknown is None else unknown
        if partial is None:
            partial = self.partial
        if many and not is_sequence_but_not_string(data):
            # Cold user-error path: delegate so the error shape is upstream's.
            return super()._do_load(
                data, many=many, partial=partial, unknown=unknown, postprocess=postprocess
            )
        result, errors = _core.load(handle, data, many, partial, unknown)
        if errors is None:
            return result
        exc = ValidationError(materialize(errors), data=data, valid_data=result)
        self.handle_error(exc, data, many=many, partial=partial)
        raise exc
