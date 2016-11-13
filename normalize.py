import sys
import unicodedata as ud

if len(sys.argv) != 2:
    print('Usage: python normalize.py lang.txt')
    sys.exit (1)

lang = sys.argv[1]

file = open('Corpora/' + lang, 'r')
normalized_file = open('Normalized Corpora/' + lang, 'w')

for line in file.readlines():
    normalized_file.write(ud.normalize('NFKC', line))

file.close()
normalized_file.close()