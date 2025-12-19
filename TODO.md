# LogRegpy

## Parallel Methods
- [x] Implement initial ParallelTree using multiprocessing
  - [x] Implement ParallelBrancher
  - [x] Connect ParallelBranchers to ParallelTree (used multiprocessing Pipe)
- [x] Optimize and take statistics on solve times for queuing model
  - [x] Startup statistics
  - [x] Brancher Utilization Statistics
  - [x] Full solve time statistics
- [ ] Switch to shared memory queue model

## GPU Processing
- [ ] Gradient Descent
  - [x] Implemented
  - [ ] Hyperparameter Tuning
    - [ ] Initial Learning Rate
    - [ ] Learning Rate Decay
- [ ] Batched Gradient Descent
  - [x] Implemented
  - [ ] Hyperparameter Tuning
    - [ ] Batch Size
    - [ ] Initial Learning Rate
    - [ ] Learning Rate Decay
- [ ] Stochastic Gradient Descent
  - [x] Implemented
  - [ ] Hyperparameter Tuning
    - [ ] Initial Learning Rate
    - [ ] Learning Rate Decay
- [ ] Dual Methods

## General
- [ ] Reimplement greedy methods for brancher objects

## Testing
- [ ]