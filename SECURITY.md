# Security policy

## Supported versions

| Version | Supported |
|---|---|
| Latest minor release | Yes |
| Earlier/private research versions | No |

The transport hardening documented below is included from v1.4.0. The
v1.5.1-native-vl.1 release additionally applies fail-closed local and remote
media admission. The v1.3.0 server has no built-in bearer authentication or
socket timeout and must remain on loopback or behind a trusted authenticated
gateway.

## Reporting a vulnerability

Use GitHub's private vulnerability reporting for this repository. Do not open
a public issue for a suspected vulnerability that could expose hosts, model
artifacts or service users. Include the release version, engine hash, operating
system, reproduction steps and impact.

Never commit credentials, private keys, private host addresses or personal
deployment paths. `make security-scan`, `make check` and `aima-engine verify`
scan tracked and non-ignored candidate public files for high-signal leak
patterns. This check is a
prevention layer, not a history sanitizer. Treat any credential that ever
entered Git history as compromised: rotate it first, then coordinate a separate
history rewrite and release-asset audit with repository administrators.

## Deployment boundary

The supported deployment artifact is the checksummed portable native archive.
It does not install or load Python, PyTorch, vLLM, Triton or Transformers.
Framework compatibility sources remain in the repository for historical
provenance only; there is intentionally no supported framework-runtime
requirements file. Do not infer a production dependency set from historical
source imports or benchmark records.

The built-in server is an inference transport, not an internet-facing gateway:

- it supports a bearer token from a permission-checked `--api-key-file`, but
  does not provide user-level authorization, TLS or rate limiting;
- a non-loopback bind fails closed unless an API key is configured (the
  `--allow-insecure-remote` override is intentionally explicit);
- `POST /shutdown` can be disabled with `--disable-http-shutdown`; the packaged
  systemd unit disables it and uses signals for lifecycle control;
- request and error artifacts may contain user prompts and generated text;
- the complete request header/body read has an absolute deadline and each
  socket write has a configurable timeout,
  but one active SSE client still owns the single batch-1 execution slot until
  it disconnects or generation ends;
- model weights and startup images are memory-mapped/read by a privileged local
  process.

Multimodal media is untrusted input:

- local `file:` URLs are disabled unless the operator supplies one or more
  `--allowed-local-media-path` roots; resolution is descriptor-relative,
  rejects traversal and symlinks, requires a stable regular file and enforces
  byte limits;
- remote HTTP/HTTPS is disabled unless every hop's hostname is present in an
  exact `--allowed-media-domain` list; DNS results are checked and private,
  loopback or link-local addresses require a separate
  `--allowed-private-media-domain` opt-in;
- URL credentials, proxy inheritance, unsafe schemes, TLS downgrade, excess
  redirects, response bytes and request/decode deadlines are rejected;
- image dimensions, decoded pixels, video source/selected frames and aggregate
  media-token use are bounded before execution;
- cache keys bind the media byte digest, kind and effective processor options,
  so changing content behind the same pathname or URL cannot reuse stale
  embeddings;
- public qualification records and service logs must not contain media bytes,
  credentials or private filesystem paths.

The packaged systemd unit allowlists `/srv/aima-media` for local content and
does not enable remote domains. Keep that directory non-writable by the service
account. Add remote domains only through an audited unit override and keep the
engine behind the same authenticated gateway as text requests.

The default bind address is `127.0.0.1`. For remote access, configure the API
key and place the service behind a TLS proxy, restrict network reachability,
protect the output directory and run it as a dedicated non-root user with GPU
access. The built-in token is defense in depth, not a replacement for TLS.

Do not attach secrets to prompts or publish runtime artifact directories
without reviewing their contents.
