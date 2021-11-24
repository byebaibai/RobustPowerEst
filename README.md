# Robust Power Estimation

This is Python implementation of robust power estimation algorithm from paper:  
[Robust power spectral estimation for EEG data](https://pubmed.ncbi.nlm.nih.gov/27102041/).

## Introduction
The original paper proposed a robsut power estimation method using `median` or `quantile` to replace `mean` operation in averaging over trials, achieving promising result in data with many outliners.

The source code in original paper was written in MATLAB, this branch use Python to implement this algorithm.
