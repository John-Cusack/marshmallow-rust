"""Materialize Rust-core error trees into upstream-shaped messages.

Wire protocol (private between `_core` and this module):

* `("$map", [(key, node), ...])` -> dict; key is `("$s", str)`,
  `("$i", int)`, or `("$k", obj)` for arbitrary mapping keys.
* `[node, ...]` (plain list) -> list of materialized nodes.
* `("$ready", obj)` -> obj verbatim (already-formatted user messages that
  must not be traversed: `ValidationError.messages` may be any shape).
* `("$deferred", template, args, _input_slot, join_input)` -> formats
  exactly like the upstream `make_error` / `_format_error` site it
  mirrors: same template string, same live objects (`args` already holds
  every kwarg the site passed, including `input` when the site passed it),
  same interpreter `str.format`. `join_input` stringifies a sequence value
  first (ContainsOnly/ContainsNoneOf report the joined elements as input).
* `str` -> verbatim.

Bare dicts never cross from Rust; the defensive branch below recurses
them with raw keys only to fail loudly-shaped rather than crash.
"""

_MAP = "$map"
_READY = "$ready"
_DEFERRED = "$deferred"


def materialize(node):
    if isinstance(node, list):
        return [materialize(item) for item in node]
    if isinstance(node, tuple) and len(node) == 2 and node[0] in (_MAP, _READY):
        tag, payload = node
        if tag == _READY:
            return payload
        return {materialize_key(key): materialize(value) for key, value in payload}
    if isinstance(node, tuple) and len(node) == 5 and node[0] == _DEFERRED:
        _, template, args, _input_slot, join_input = node
        return format_deferred(template, args, join_input)
    if isinstance(node, dict):
        return {key: materialize(value) for key, value in node.items()}
    return node


def materialize_key(key):
    if isinstance(key, tuple) and len(key) == 2 and key[0] in ("$s", "$i", "$k"):
        return key[1]
    return key


def format_deferred(template, args, join_input):
    args = dict(args)
    if join_input:
        args["input"] = ", ".join(str(item) for item in args["input"])
    return template.format(**args)
