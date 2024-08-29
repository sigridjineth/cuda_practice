# Usage: python input_fp32.py <num_samples> <path/to/input_fp32.bin>
# Description: Generate random samples from a normal distribution and save it to a binary file

import sys
import torch
import numpy as np

# Set the random seed for reproducibility
torch.manual_seed(123)

if __name__ == '__main__':
  if len(sys.argv) < 3:
    print("Usage: python input_fp32.py <num_samples> <path/to/input_fp32.bin>")
    sys.exit(1)

  # Get the output path
  output_path = sys.argv[2]

  # Set the number of samples and latent dimension
  num_samples = int(sys.argv[1])
  latent_dim = 128

  # Generate random samples from a normal distribution
  z = torch.randn(num_samples, latent_dim).float()

  # Save the tensor to a binary file
  with open(output_path, "wb") as f:
    f.write(z.numpy().tobytes())
