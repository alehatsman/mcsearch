#!/usr/bin/env python3
"""Tiny helper: read questions.yml and emit JSON or specific fields.

Usage:
  _qload.py ids                          -> one id per line
  _qload.py get <id> <field>             -> value of that field for id
"""
import sys, json, pathlib, yaml

QFILE = pathlib.Path(__file__).resolve().parent.parent / "questions" / "questions.yml"

def load():
    with QFILE.open() as f:
        return yaml.safe_load(f)["questions"]

def main():
    if len(sys.argv) < 2:
        sys.exit("usage: _qload.py ids | get <id> <field>")
    cmd = sys.argv[1]
    qs = load()
    if cmd == "ids":
        for q in qs:
            print(q["id"])
    elif cmd == "get":
        qid, field = sys.argv[2], sys.argv[3]
        for q in qs:
            if q["id"] == qid:
                v = q.get(field, "")
                if isinstance(v, (list, dict)):
                    print(json.dumps(v))
                else:
                    print(v, end="")
                return
        sys.exit(f"no such question: {qid}")
    else:
        sys.exit(f"unknown cmd: {cmd}")

if __name__ == "__main__":
    main()
