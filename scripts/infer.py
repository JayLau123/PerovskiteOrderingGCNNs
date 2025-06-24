#!/usr/bin/env python3
import sys
import subprocess

def main():
    if len(sys.argv) == 1:
        print('No arguments provided. Use --help for usage.')
        sys.exit(1)
    cmd = ['python3', '2_model_inference.py'] + sys.argv[1:]
    subprocess.run(cmd)

if __name__ == '__main__':
    main() 