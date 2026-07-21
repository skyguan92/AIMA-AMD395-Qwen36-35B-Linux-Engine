# Security policy

## Supported versions

| Version | Supported |
|---|---|
| 1.0.x | Yes |
| Earlier/private research versions | No |

## Reporting a vulnerability

Use GitHub's private vulnerability reporting for this repository. Do not open
a public issue for a suspected vulnerability that could expose hosts, model
artifacts or service users. Include the release version, engine hash, operating
system, reproduction steps and impact.

## Deployment boundary

The built-in server is an inference transport, not an internet-facing gateway:

- it has no authentication, authorization, TLS or rate limiting;
- `POST /shutdown` is intentionally available for local lifecycle control;
- request and error artifacts may contain user prompts and generated text;
- model weights and startup images are memory-mapped/read by a privileged local
  process.

The default bind address is `127.0.0.1`. For remote access, place the service
behind an authenticated TLS proxy, restrict network reachability, protect the
output directory and run it as a dedicated non-root user with GPU access.

Do not attach secrets to prompts or publish runtime artifact directories
without reviewing their contents.
