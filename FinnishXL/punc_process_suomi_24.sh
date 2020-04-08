#!/bin/bash

for i in *[_cleanedpMorfp].vrt;do
    sbatch punctuation_suomi24.slrm $i
done