import argparse
import sys

import matplotlib as mpl
mpl.use("pgf")

import matplotlib.pyplot as plt
import numpy as np

preamble = ("\\usepackage[utf8]{inputenc}\n"
        "\\DeclareUnicodeCharacter{2212}{-}"
        "\\usepackage[T1]{fontenc}\n")

pgf_with_latex = {                      # setup matplotlib to use latex for output
    "pgf.texsystem": "pdflatex",        # change this if using xetex or lautex
    "text.usetex": True,                # use LaTeX to write all text
    "font.family": "DejaVu Sans",
    "font.serif": [],                   # blank entries should cause plots 
    "font.sans-serif": [],              # to inherit fonts from the document
    "font.monospace": [],
    "axes.labelsize": 12,
    "font.size": 12,
    "legend.fontsize": 12,               # Make the legend/label fonts 
    "xtick.labelsize": 12,               # a little smaller
    "ytick.labelsize": 12,
    "figure.figsize": [0.9, 0.9],     # default fig size of 0.9 textwidth
# use utf8 input and T1 fonts 
"pgf.preamble" : preamble        # plots will be generated 
                                           # using this preamble
    }
mpl.rcParams.update(pgf_with_latex)

X = []
Y = []
Y2 = []

s = 0
i = 0


parser = argparse.ArgumentParser()
parser.add_argument("-f", type=str)
args = parser.parse_args()

with open(args.f, 'r') as f:
    for line in f.read().split("\n"):
        i += 1
        l = line.split(",")
        if len(l) < 2:
            continue
        x1, x2, x3 =  l[0], l[1], l[2]
        x1 = int(x1)
        x2 = float(x2)
        x3 = float(x3)*100
        X.append(x1)
        Y.append(x2)
        Y2.append(x3)

n = 10
#X = X[:10]
#Y = Y[:10]
#Y2 = Y2[:10]
fig, ax = plt.subplots(figsize=(7,3))
ax.set_title("Training loss and accuracy")
ax.set_xlabel("epochs")
ax.set_ylabel("training loss", color='tab:red')
ax.plot(X, Y, label="training loss", color='tab:red') # 'r', c=(1,0,0),# label="theta")
ax2 = ax.twinx()  # instantiate a second axes that shares the same x-axis
ax2.set_ylabel("testing accuracy(%)", color='tab:blue')
ax2.plot(X, Y2, label="testing accuracy(%)", color='tab:blue') # 'r', c=(1,0,0),# label="theta")
#plt.legend()

plt.savefig(f"{args.f}_accuracy.pdf",
 bbox_inches = 'tight',
    pad_inches = 0)


