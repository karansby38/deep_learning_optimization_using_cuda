# Deep Learning Optimization Using CUDA

This repository explores optimizing deep learning models using CUDA to leverage GPU acceleration. We have implemented fundamental neural network models such as Binary Classification and Multi-Layer Perceptron (MLP) using CUDA on an NVIDIA GTX 1650 Ti.

## Project Purpose
The objective of this project is to:

- Utilize CUDA for parallelizing deep learning computations.
- Compare CPU and GPU implementations to analyze speed improvements.
- Optimize neural network training using GPU acceleration.

## Implemented Features
- **Binary Classification Neural Network**: A simple neural network trained to perform binary classification using CUDA.
- **Multi-Layer Perceptron (MLP)**: Implemented with CUDA to leverage parallel processing capabilities of the GPU.
- **Performance Comparison**: Benchmarking execution time differences between CPU and GPU implementations.

## Requirements
Ensure you have the following dependencies installed:

- CUDA Toolkit (Version compatible with GTX 1650 Ti)
- NVIDIA GPU with CUDA support
- GCC Compiler
- Python (for optional testing and visualization)

## How to Run the Project

### 1. Clone the Repository
```sh
git clone https://github.com/yourusername/deep_learning_optimization_using_cuda.git
cd deep_learning_optimization_using_cuda
```

### 2. Compile the CUDA Code
```sh
nvcc -o mlp_cuda mlp_cuda.cu
nvcc -o binary_classification_cuda binary_classification.cu
```

### 3. Run the Executables
```sh
./mlp_cuda
./binary_classification_cuda
```

### 4. (Optional) Run Python Visualization
If applicable, you can visualize results using Python:
```sh
python visualize_results.py
```

## CPU vs GPU Performance Analysis
We have benchmarked the execution times of both CPU and GPU implementations. The CUDA-accelerated versions demonstrate significant speed improvements over traditional CPU-based training.

## Future Enhancements
- Implement more complex deep learning models.
- Optimize memory usage and kernel performance.
- Experiment with different CUDA architectures and Tensor Cores.

