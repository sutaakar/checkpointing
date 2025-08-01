# Checkpointing with PyTorchJob

This directory contains an example of a fault-tolerant `PyTorchJob` that demonstrates checkpointing capabilities for supervised fine-tuning (SFT) of Large Language Models.

## Overview

The `pytorchjob.yaml` file defines a Kubernetes `PyTorchJob` to fine-tune the `meta-llama/Llama-3.2-1B-Instruct` model on the `gsm8k` dataset.

The primary feature of this example is its resilience to preemption. It includes a custom mechanism to gracefully handle `SIGTERM` signals, which are often sent to pods before they are terminated (e.g., due to node shutdown, preemption by a higher-priority job, or manual deletion). This is achieved through a custom `transformers.TrainerCallback` (`SigtermCheckpointCallback`) implemented directly within the job specification.

This allows the training job to be interrupted and resumed without losing significant progress by:
- **Capturing the `SIGTERM` signal.**
- **Coordinating a graceful shutdown across all distributed workers.**
- **Saving a final model checkpoint before exiting.**
- **Resuming automatically from the last checkpoint when the job restarts.**

## How it Works

The fault tolerance is achieved through a custom `transformers.TrainerCallback` called `SigtermCheckpointCallback`. Here is a step-by-step breakdown of the process:

1.  **Job Submission**: The `PyTorchJob` is submitted to a Kubernetes cluster. The `kueue.x-k8s.io/queue-name` label ensures it's managed by the Kueue scheduler, which may preempt the job.

2.  **Callback Initialization**: At the start of the training script, the `SigtermCheckpointCallback` is added to the `SFTTrainer`.

3.  **Signal Handling Setup**: During the `on_train_begin` event, the callback performs two critical setup steps on all distributed ranks:
    *   It registers a Python signal handler for the `SIGTERM` signal.
    *   It initializes a shared `torch.Tensor` on the GPU (or CPU if not available). This tensor is used to communicate the `SIGTERM` event across all processes in the distributed job.

4.  **SIGTERM Reception**: When a pod in the `PyTorchJob` receives a `SIGTERM` signal (e.g., from Kubernetes due to preemption), the registered handler is invoked. The handler sets a local flag and writes a value of `1.0` to the shared tensor.

5.  **Distributed Coordination**: During the `on_step_end` event, after each training step, every rank performs a `dist.all_reduce` operation on the shared tensor. This operation aggregates the tensor values from all ranks (using a `MAX` operation). If any rank has received the signal and written `1.0` to its tensor, all ranks will see this change after the `all_reduce`.

6.  **Graceful Shutdown**:
    *   Upon detecting the signal, the callback sets `control.should_save = True` to instruct the `Trainer` to save a checkpoint at the end of the current step.
    *   It also sets `control.should_training_stop = True` to ensure the training loop terminates cleanly after saving.

7.  **Checkpointing and Resumption**:
    *   The checkpoint is saved to a shared persistent volume.
    *   When the job is rescheduled by Kueue, it starts a new set of pods. The training script checks for a checkpoint in the output directory using `get_last_checkpoint`. If a checkpoint is found, `trainer.train()` resumes from that state, preserving the training progress.

### `SigtermCheckpointCallback` in Detail

The `SigtermCheckpointCallback` is the core component enabling this fault tolerance. Its functionality is tied into the Hugging Face Trainer's event-based lifecycle:

-   `on_train_begin`: Sets up the signal handler and the distributed tensor for cross-process communication. It uses a `dist.barrier()` to ensure all ranks are synchronized before training begins.
-   `_sigterm_handler`: The function that is executed when a `SIGTERM` signal is caught. It's responsible for updating the distributed tensor to notify other ranks.
-   `_check_distributed_sigterm`: A helper method that performs the `dist.all_reduce` to check if any process has received the signal.
-   `on_step_end`: The hook that periodically calls `_check_distributed_sigterm` and modifies the `Trainer`'s control state to trigger a save and stop operation.
-   `on_save` and `on_train_end`: Provide logging to confirm that the checkpoint was saved and that training stopped due to the signal, which is helpful for monitoring and debugging.

## Prerequisites

To run this example, you need:
- A Kubernetes cluster.
- The Kubeflow Training Operator installed to manage `PyTorchJob` resources.
- Kueue installed for job queueing and scheduling.
- A Persistent Volume Claim (PVC) named `shared` available in the same namespace.
- A Kubernetes secret named `hf-token` containing a Hugging Face token with access to the required models.
