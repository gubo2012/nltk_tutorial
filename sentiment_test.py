#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr  6 16:37:48 2018

@author: gubo
"""

import sentiment_mod as s

print(s.sentiment("This movie was awesome! The acting was great, plot was wonderful, and there were pythons...so yea!"))
print(s.sentiment("This movie was utter junk. There were absolutely 0 pythons. I don't see what the point was at all. Horrible movie, 0/10"))

print(s.sentiment('Like most unintended second installments, this one is superfluous - a remix of moments, scenes, and images from its predecessor infused with the need to make everything bigger and louder.'))
print(s.sentiment('Like leftovers in the fridge or reuniting with a lost love, "Pacific Rim: Uprising" is proof things are never as good the second time around.'))
print(s.sentiment('It knows its mission parameters, and it ticks them off with alacrity.'))


