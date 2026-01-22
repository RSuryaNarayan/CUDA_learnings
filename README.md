# CUDA_learnings
dump of everything I tried while learning CUDA. Tested with MSI A40 GPU. 

MSI module loads:
```bash
module purge
module avail cuda
module avail nvhpc

module load cuda
which nvcc
nvcc --version
```
Use a handy alias to get gpu:

```bash
alias get_gpu="srun --partition=interactive-gpu --gres=gpu:1 --cpus-per-task=2 --mem=8G --time=1:00:00 --pty bash"
```
nsys works on MSI

```bash
nsys profile ./transpose_tiled
```
<img width="1248" height="437" alt="image" src="https://github.com/user-attachments/assets/6b9f2c9f-7d06-475c-b472-1935308cba18" />
