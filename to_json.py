import json
import gzip
import sys

def parse(path):
  g = gzip.open(path, 'r')
  for l in g:
    yield json.dumps(eval(l))

def main(_input, output):
    f = open(output, 'w')
    for i, l in enumerate(parse(_input)):
      if (i < 80000):
        f.write(l + '\n')

if __name__ == "__main__":
   main(sys.argv[1], sys.argv[2])
