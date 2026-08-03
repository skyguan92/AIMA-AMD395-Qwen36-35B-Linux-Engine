# Security policy

## Supported versions

| Version | Supported |
|---|---|
| Latest minor release | Yes |
| Earlier/private research versions | No |

The hardening documented below is included in v1.4.0. The v1.3.0 server has no
built-in bearer authentication or socket timeout and must remain on loopback
or behind a trusted authenticated gateway.

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

The default bind address is `127.0.0.1`. For remote access, configure the API
key and place the service behind a TLS proxy, restrict network reachability,
protect the output directory and run it as a dedicated non-root user with GPU
access. The built-in token is defense in depth, not a replacement for TLS.

Do not attach secrets to prompts or publish runtime artifact directories
without reviewing their contents.
