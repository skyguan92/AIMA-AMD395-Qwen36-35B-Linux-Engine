# Contributing

Contributions are welcome when they preserve the project's narrow qualified
contract and evidence boundaries.

## Development setup

CPU-safe checks require Python 3.10 or newer and a C++17 compiler:

```bash
make check
```

GPU/runtime changes additionally require the qualified AMD395 host, matching
model checkpoint and runtime environment.

## Change discipline

- Keep model math, performance, startup and API/correctness claims separate.
- Do not silently add generic fallbacks or user-visible variant switches.
- Preserve batch-1 deterministic behavior unless a new release explicitly
  expands the contract.
- Never commit model weights, startup-image lanes, prompts containing private
  data, credentials or host-specific absolute paths.
- Update runtime hashes when changing a pinned production component.
- Add a regression test for every API, stop, usage or cache-state repair.

Performance pull requests should include the exact engine hash, runtime and
hardware versions, prompt/output sizes, cache state, warmup, raw command,
correctness result and repeat measurements. A single faster run is not enough.

By submitting a contribution, you agree that it may be distributed under the
repository's Apache-2.0 license. Third-party code must retain its original
license and attribution.
