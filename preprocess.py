#!/usr/bin/env python
# -*- coding: utf-8 -*-


import nltk
import os


def tokenize(infile, targetdir):
    """Preprocess file so that every sentence is in one line"""
    token_list = []
    text = infile.read()
    sent_tok = nltk.sent_tokenize(text)
    for sent in sent_tok:
        word_tok = nltk.word_tokenize(sent)
        token_list.append(' '.join(word_tok))
    with open(os.path.join(targetdir, 'donquijote.txt'), 'w') as target_file:
        for sent in token_list:
            target_file.write(sent+'\n')
