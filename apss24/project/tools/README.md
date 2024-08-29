# Project Tools

Repo of Tools for APSS'24 Project

## Input Binary File Generation
```bash
# FP32
python input_fp32.py <num_samples> <path/to/input_fp32.bin>

# FP16
python input_fp16.py <num_samples> <path/to/input_fp16.bin>
```

## Answer Binary File Generation

Run skeleton code and save 'output.bin' as 'answer.bin'

```bash
# FP32
./run.sh -n <num_samples>

# FP16
./run.sh -n <num_samples>
```

## Output Result Visualization
### Python
```bash
# FP32
python bin2img_fp32.py <path/to/output.bin> <path/to/output.png>

# FP16
python bin2img_fp16.py <path/to/output.bin> <path/to/output.png>
```
### Executable (by PyInstaller)
```bash
# FP32
./bin2img_fp32 <path/to/output.bin> <path/to/output.png>

# FP16
./bin2img_fp16 <path/to/output.bin> <path/to/output.png>
```

 