#!/usr/bin/env python3
import sys
import subprocess

def main():
    if len(sys.argv) == 1:
        print('No arguments provided. Use --help for usage.')
        sys.exit(1)
    cmd = ['python3', '1_model_training.py'] + sys.argv[1:]
    subprocess.run(cmd)

if __name__ == '__main__':
    main() 