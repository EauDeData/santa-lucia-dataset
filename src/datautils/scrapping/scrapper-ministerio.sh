#!/bin/bash

while read i
do
  wget $i -O scrapped/${i: -6}.pdf 
done < $1
