# Final Project Proposal

If you give me a word, can I tell you what language it's in? This project determines that. In addition to writing a small implementation of a recurrent neural network to implement this AI, I'll be testing the project idea out with an already-developed RNN library, TensorFlow. Let's hope it works!

## Description

I'll be creating and training an RNN in MATLAB on ~175,000 words in ~60 different languages.

The RNN is to be as simple as possible, so that I can analyze it and its results more easily.

I'll also be munging (pre-filtering) the input data, scraped from Wikipedia, using a custom algorithm called common sense.

After training, I'll make some analyses, dig a little deeper into the "meaning" of the trained weights, and possibly specialize the RNN for this specific task to increase its accuracy.

## Learning Objective

The goal, of course, is to get good accuracy. This is going to depend on a good implementation of the RNN algorithm and enough computing time and power.

The interesting questions, though, are the subtler questions which concern the implementation and the results. Per-language accuracy, for example: it's easier to tell if a word's in Korean than Danish, simply because only Korean uses the Hangul script. Disambiguating Norwegian from Danish is naturally more difficult, because they're more closely related, and use the same letters. Is this the same for neural nets, though?

Second to accuracy, I'd like to find out how _few_ hidden nodes I'll need for, say, 99% accuracy.

And what are they _doing_? Thirdly, I'd like to interpret how the RNN is classifying (as best I can). Is it using frequency analysis? The frequency of pairs of letters? Triples of letters? How words end (plurals)? An especially interesting topic would be comparing models of similar languages.

Lastly, if _all I want_ is see what language a word is in, there are much, much, better tools for this (dictionaries, right?). With an RNN, though, we can ask questions that dictionaries can't answer, such as: What language is a made-up word, such as "freyguis," in?, or, How "English-like" are loanwords from non-Germanic sources (how much have they changed the language)? Etc.

## Details

### Sources

I'll be using two independent recurrent neural networks to implement this. The first will be Tensorflow, so I'll be using the Tensorflow [documentation](https://www.tensorflow.org/versions/r0.11/api_docs/index.html) and [tutorials](https://www.tensorflow.org/versions/r0.11/tutorials/recurrent/index.html) to get that working. I'm also going to write my own implementation, so my primary resources for that are going to be DLB, NNDL, and whatever I can find online.

There are many academic papers on this specific topic, such as [Multilingual Text Classification](https://www.ijert.org/view-pdf/12550/multilingual-text-classification) by Mittal and Dhyani. These will be useful as well, for domain-specific judgment.

### The Data Set

I'm scraping [E-Z-Glot](http://www.ezglot.com/most-frequently-used-words.php) to get the 3,000 most common words in each of 58 languages (as used on Wikipedia in these languages). To do this, I visit each page, e.g. http://www.ezglot.com/most-frequently-used-words.php?l=est, and run a small JavaScript snippet in the developer console,

```
$$('.topwords li span').reduce(function (acc, x, i) {
    return acc + ' ' + x.innerHTML;
}, '');
```

which returns a string. I then remove the single-letter words from the corpora in a text editor.

The next step is going through every one by hand. Unfortunately, because these are scraped from Wikipedia, there are some errors, and I'm removing the most-obvious ones: latin letters in non-latin scripts, double-width numbers and punctuation marks, Korean has a bunch of Russian (Cyrillic) words in it for some reason, right-to-left scripts need finessing, etc.

Finally, I normalize the words to NFKC, so that they're comparable.

## Questions and Feedback

Please let me know your thoughts, especially if you speak more than one language, or you're working on an RNN, too. I'd also love to test anybody else's questions about language classification on this dataset or network; just comment below.

I'll put this up on GitHub, location TBD, when it's a little further along with a comment as well.

