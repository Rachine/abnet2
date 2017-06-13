#!/bin/bash
#Script for Automation
#TODO inject directly training w/ parameters
#TODO launch training and output nn params
#TODO insert initial to evaluate embeddings

Feature_file_name=$1
Feature_extension='.feat'
Feature_file=$Feature_file_name$Feature_extension
Feature_ascii=$Feature_file_name'_ascii'

python h5feat2ascii.py $Feature_file $Feature_ascii

echo $Feature_ascii

mkdir 'eval2_'$Feature_file

#TODO insert eval command HERE

