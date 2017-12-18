#!/usr/bin/env bash


while true; do scancel --user=${USER}; squeue | grep ${USER} | wc -l; sleep 1; done