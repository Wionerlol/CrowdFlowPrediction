## Training review workflow

For changes involving model training, datasets, loss functions, evaluation,
distributed training, checkpoints, or experiment configuration:

1. Complete the implementation and perform inexpensive validation first.
2. Spawn the `training_reviewer` agent for an independent read-only review.
3. Give the reviewer the changed files, relevant configuration, validation
   output, and training-log paths.
4. Wait for the reviewer to finish before presenting the final result.
5. Address confirmed BLOCKER and HIGH findings.
6. If code is changed in response to review findings, ask the reviewer to
   inspect the final diff again.
7. Do not let the reviewer modify files or launch expensive training jobs.
8. Clearly report unresolved findings to the user.

Do not trigger a full training run solely for review unless the user explicitly
requests it.
