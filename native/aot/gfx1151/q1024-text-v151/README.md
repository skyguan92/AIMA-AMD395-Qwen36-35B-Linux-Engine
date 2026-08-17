# Frozen q1024 text closure

This directory preserves the q1024 prefill closure shipped by the frozen
`v1.5.1` text product at source commit
`65c198415709dad6d046c247acab3dc9df2a95a0`.

- `manifest.json` SHA-256:
  `93853b9f9837deba0a9e051bf5be4c516d74d1c5ea1a33e8e7e47ee81e914125`
- `prefill-schedule.json` SHA-256:
  `10565e59b0805ca407ef453caf72f3dfd254752d150903131e188527b910fb97`

The current q1024 closure remains at `../q1024-output1` and is selected only
for VL requests. Text-only requests bind this frozen schedule to its own
schedule-shaped workspace; captured `transient.N` names are not merged across
the two independent view maps. The frozen map physically owns its exact
`669879552`-byte allocation including prompt ids. The current map aliases its
prefix below byte `668730624`, then maps bindings starting at `transient.73`
into a separately owned `5359616`-byte tail allocated after the text-critical
resident topology. No binding crosses the split, and shared bytes are counted
only by the frozen owner. Do not replace either closure with the other: their
FLA and fused-MoE captures, tensor ABI, allocation layout, and qualification
owners are intentionally different.
