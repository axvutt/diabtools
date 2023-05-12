#!/usr/bin/env python3

import diabtools
import argparse

def main():
    parser = argparse.ArgumentParser(
            prog="spm2txt",
            description="Read SymPolyMat binary file and print content to stdout.")
    parser.add_argument('filename')
    args = parser.parse_args()

    with open(args.filename, "rb") as fin:
        W = diabtools.SymPolyMat.read_from_file(fin)
        print(W)

if __name__ == "__main__":
    main()

