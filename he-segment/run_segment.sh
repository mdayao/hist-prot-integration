#!/bin/bash

#python segment.py --data-dir ./data/healthy/ --file-metadata ./data/healthy/healthy_list.csv --output-dir ./data/healthy/segment/ > ./data/healthy/segment.log
#python segment.py --data-dir ./data/ckd/ --file-metadata ./data/ckd/ckd_list.csv --output-dir ./data/ckd/segment/ --append > ./data/ckd/segment.log
python segment.py --data-dir ./data/aki/ --file-metadata ./data/aki/aki_list.csv --output-dir ./data/aki/segment/ --append > ./data/aki/segment.log
