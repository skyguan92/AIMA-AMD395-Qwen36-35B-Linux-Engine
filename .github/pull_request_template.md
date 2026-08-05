## What changed

Describe the user-visible behavior and the narrow implementation scope.

## Evidence

- [ ] `make check` passes.
- [ ] Runtime changes include the exact command, engine hash, host/runtime facts,
      prompt/output lengths, cache state, raw artifact path and correctness gate.
- [ ] Performance claims use the repository's replicate protocol and include a
      correctness result from the same binary.
- [ ] No model weights, private prompts, credentials or host-specific paths are
      committed.

## Release impact

State whether this changes the portable archive, API compatibility, memory
requirements, benchmark claims or release notes. Write `none` when it does not.
