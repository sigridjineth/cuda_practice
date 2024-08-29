#!/bin/bash

srun -N 1 --partition apws --exclusive ./vec_add_opencl
