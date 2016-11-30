#!/usr/bin/env bash
grep ^REWARD $1 | cut -f2 | jq '.reward' | sort | uniq -c | normalise.py  | sort -k3 -nr
