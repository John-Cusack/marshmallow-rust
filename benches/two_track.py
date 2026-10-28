#!/usr/bin/env python3
"""Two-track bench: upstream marshmallow vs marshmallow-rust, measured only.

Runs the frozen lane workload (500 nested objects, dump(many=True) +
load(many=True) x4) on both sides and reports wall time + peak RSS.

Usage: testvenv/bin/python benches/two_track.py [--reps N]

Each side runs in a fresh process per rep (honest peak RSS). Outputs a
Markdown table on stdout. No projections. Kill line for lane 3: <1.5x on
this workload is no_gain.
"""

import argparse
import os
import statistics
import subprocess
import sys

MIRROR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

def build_data():
    users = [
        {
            "name": f"user-{i}",
            "email": f"user-{i}@example.com",
            "age": 20 + (i % 60),
            "address": {
                "street": f"{i} Main St",
                "city": "Springfield",
                "zip_code": f"{10000 + (i % 89999):05d}",
            },
            "orders": [
                {
                    "item": f"widget-{j}",
                    "quantity": 1 + ((i + j) % 9),
                    "price": 1.5 + (i % 100),
                }
                for j in range(3)
            ],
        }
        for i in range(500)
    ]
    return users


SCHEMA_SRC = """
from marshmallow import Schema, fields

class AddressSchema(Schema):
    street = fields.Str(required=True)
    city = fields.Str(required=True)
    zip_code = fields.Str(required=True, validate=lambda s: len(s) == 5)
    country = fields.Str(load_default="US")

class OrderSchema(Schema):
    item = fields.Str(required=True)
    quantity = fields.Int(required=True, validate=lambda n: 0 < n < 1000)
    price = fields.Float(required=True)
    gift_wrap = fields.Bool(load_default=False)

class UserSchema(Schema):
    name = fields.Str(required=True)
    email = fields.Email(required=True)
    age = fields.Int(validate=lambda n: 0 <= n <= 150)
    address = fields.Nested(AddressSchema, required=True)
    orders = fields.List(fields.Nested(OrderSchema), required=True)
"""


def worker_source(side):
    if side == "port":
        schema_src = SCHEMA_SRC.replace(
            "from marshmallow import Schema, fields",
            "from marshmallow_rust import Schema, fields",
        )
    else:
        schema_src = SCHEMA_SRC
    engage = "assert schema._rust_handle is not None, 'extension not engaged'\n" if side == "port" else ""
    return (
        "import pickle, resource, sys, time\n"
        f"{schema_src}\n"
        "users = pickle.loads(sys.stdin.buffer.read())\n"
        "schema = UserSchema()\n"
        "dumped = schema.dump(users, many=True)\n"
        + engage
        + "t0 = time.perf_counter()\n"
        + "for _ in range(4):\n"
        + "    d = schema.dump(users, many=True)\n"
        + "    loaded = schema.load(dumped, many=True)\n"
        + "assert len(d) == 500 and len(loaded) == 500\n"
        + "wall = (time.perf_counter() - t0)\n"
        + "print(f'wall_ms={wall * 1000:.2f} peak_kb={resource.getrusage(resource.RUSAGE_SELF).ru_maxrss}')"
    )


def run_worker(python, side, payload, reps=5):
    # Neutral cwd: `python -c` puts cwd on sys.path, so a source checkout
    # would shadow the installed wheel (and silently disable the extension).
    import tempfile

    walls, peaks = [], []
    for _ in range(reps):
        p = subprocess.run(
            [python, "-c", worker_source(side)],
            input=payload,
            capture_output=True,
            timeout=600,
            cwd=tempfile.gettempdir(),
        )
        if p.returncode != 0:
            raise RuntimeError(f"{side} worker failed: {p.stderr.decode()[-2000:]}")
        kv = dict(
            t.split("=") for t in p.stdout.decode().strip().split()
        )
        walls.append(float(kv["wall_ms"]))
        peaks.append(int(kv["peak_kb"]))
    return statistics.median(walls), max(peaks)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", type=int, default=5)
    args = ap.parse_args()
    import pickle

    payload = pickle.dumps(build_data(), protocol=4)
    python = sys.executable
    rows = []
    for side in ("upstream", "port"):
        wall, peak = run_worker(python, side, payload, args.reps)
        rows.append((side, wall, peak))
    up, pt = rows[0], rows[1]
    speedup = up[1] / pt[1] if pt[1] else float("nan")
    print("| side | wall (4x dump+load, 500 objs) | peak RSS |")
    print("| --- | --- | --- |")
    for side, wall, peak in rows:
        print(f"| {side} | {wall:.2f} ms | {peak} KiB |")
    print(f"\nspeedup (upstream/port): {speedup:.2f}x")
    print("kill line: <1.5x on this workload is no_gain (lane 3).")


if __name__ == "__main__":
    main()
