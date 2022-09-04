#!/bin/bash
for f in ./*.csv; do
   wc -l "$f"
done
