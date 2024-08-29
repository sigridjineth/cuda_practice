#!/bin/bash

srun --exclusive --gres=gpu:1 \
	./main $@