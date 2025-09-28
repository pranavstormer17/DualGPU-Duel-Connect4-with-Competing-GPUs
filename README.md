# DualGPU Duel — Connect4 with Competing GPUs

## Summary
DualGPU Duel is a Connect-4 style demo where two GPU players (Player A on device 0, Player B on device 1 if available) choose columns using GPU kernels. The host enforces game rules, logs every move to `replay/replay.txt`, and prints device enumeration and the chosen GPU device index (exactly as required by the assignment).

This repo contains a minimal but working implementation showing:
- device enumeration & required print statements
- device selection & launching kernels on each GPU
- host-mediated replay logging and win/draw detection
- a sample "random vs heuristic" GPU player implementation

## Build & Run
Requirements: CUDA toolkit and nvcc on PATH.

```bash
# build
./build.sh

# run demo (creates replay/replay.txt)
./run_demo.sh
