#!/usr/bin/python

import csv

data = open('scatter.log','r').read()

lines = data.split('\n')[1:-2] #trim first and last

initialValue = int(lines[0].split(' ')[0])
accum = 0

with open('scatter.csv','w') as csvfile:
    writer = csv.writer(csvfile)

    for line in lines:
        #print(line)
        elements = line.split(' ') #we only care about diff
        val = int(elements[3])
        accum = accum + val
        writer.writerow([float(accum)/1000.0])
