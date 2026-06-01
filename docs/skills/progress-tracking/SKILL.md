---
name: progress-tracking
description: Single-file project management and state tracking across sessions using docs/progress.md
---

# Progress Tracking

To ensure context is maintained across multiple sessions and complex tasks, you must act as your own project manager. You will use a single tracking file to maintain state: `docs/progress.md`.

## Core Directives

1. **Read Before Acting:** At the start of every session or new task, always read `docs/progress.md` first to establish context, understand the overall goal, and see what is pending. If the file does not exist, create it immediately.
2. **Plan Before Coding:** Outline your approach in the tracking file *before* you start modifying the codebase. This ensures actions are deliberate and aligned with the objective.
3. **Keep It Updated:** Every time you complete a sub-task, encounter a blocker, or shift your focus, update `docs/progress.md` immediately. Do not wait until the end of a session.

## File Structure Guidelines

When creating or updating `docs/progress.md`, you must maintain the following format for each major task. Separate multiple major tasks with a horizontal rule (`---`).

- `## [Short Task Title]`
- `### Objective`: A 1-2 sentence description of the overall goal.
- `### Task List`: A markdown checklist (`- [ ]`, `- [x]`) of all planned steps. Check off items as they are completed. Append new steps as they are discovered.
- `### Current Focus`: The specific sub-task you are tackling right now. Update this whenever you pivot to a new step.
- `### Notes & Blockers`: Key decisions made, specific commands run, or current errors/stack traces. Keep this concise but informative enough to resume work after a session break.

## Example Usage

```markdown
## Train BCAT Baseline Model

### Objective
Initialize and train the standard BCAT model on the fluid dynamics dataset to establish baseline metrics.

### Task List
- [x] Verify data directory paths and Hydra configs.
- [ ] Run a quick dryrun to validate the training loop, metrics logging, and memory usage.
- [ ] Launch full 4-GPU training script.

### Current Focus
Executing the dryrun command to verify pipeline stability before dedicating GPU time.

### Notes & Blockers
Command to run: `torchrun --standalone --nnodes 1 --nproc_per_node 4 src/main.py exp_id=bcat_baseline batch_size=32 data=fluids_sample compile=1 model.flex_attn=1 dryrun=1`
Results will be saved under the `debug` experiment name. No blockers currently.
```