# An RNN That Distinguishes Languages

Can we tell what language a word is in? Sure, humans can.

This repo contains cleaned and filtered corpora in 36 Latin-script languages and Matlab code for computers to do the same thing. It works&hellip;poorly.

## The Corpora

There are 36 languages represented in `Corpora/` and `Normalized Corpora/`. They consist of lists of the 5000 most-used words on Wikipedia in those languages. All one-letter words are removed. `Mini Corpora/` consists of the 500-1000 most-used words in English, German, and Spanish from the normalized corpora. "Normalized," in this case, means NFKC.

## The RNN

See the code for more details.
