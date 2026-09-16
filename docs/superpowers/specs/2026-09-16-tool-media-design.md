# Tool media support

The September 16 partner feedback has two distinct causes. Remote image URLs
already work when their exact hostname is configured with
`--allowed-media-domain`. Native request parsing currently rejects image and
video parts in tool messages. The user approved implementation after the
source review and AMD395 reproduction.

Extend native parsing to accept `image_url` and `video_url` in tool results,
using the same ordered content normalization and media policy as user media.
Preserve preceding assistant call IDs, unique result IDs, a real user query,
role restrictions for system/developer/assistant, aggregate limits, and the
existing Qwen tool-response template. Reuse the existing visual preparation,
embedding and GPU execution paths.

Tool progress must distinguish empty output from explicit failure. Media-only
results are payload; adding media must not conceal an explicit textual/JSON
failure or change the text-only retry policy. Inspect the original text parts
without inserted visual markers and classify empty/progress/failure separately.

The supported native interface is preferable to caller-side role rewriting,
which loses tool provenance. A new proxy is unnecessary because the existing
native pipeline already binds media by rendered placeholder order.

Validation includes a failing parser regression before implementation, tool
history rejection, media ordering/counts, failure classification, Qwen template
parity, existing project checks, and real AMD395 image/video generation with
stream parity, multi-turn media/cache tests, and pure-text controls. Build an
isolated candidate using the existing pinned runtime libraries. Record hashes
and retain the original package for rollback; this work does not publish an
official release or claim its complete performance qualification.
