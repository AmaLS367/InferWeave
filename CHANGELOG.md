# Changelog

## 0.2.0

- Add optional Lightning AI container Deployment provider, shared TTS/image inference,
  user-key Bearer auth, typed autoscaling/resource metadata, recovery and verified
  full deletion/failure cleanup without Studios. Add opt-in L4 live testing.
  SDK 2026.10.1 requires a separate environment from SkyPilot because of upstream
  Click bounds; `all` retains its previous composition.

- Recover persisted deployments with `attach()` and unique-match `find()`.
- Synthesize Fish Speech audio/render FLUX images with typed errors, references,
  per-call options, timeouts and bounded cold-start retries.
- Protect Modal endpoints by default with separate Proxy Tokens for probes/inference;
  credentials resolve at runtime rather than being persisted by endpoint auth.
- Separate container scaling from full idle stop, persist inference activity
  and protect local in-flight calls from idle shutdown.
- Add tutorials, guides, references and credential-free runnable example validation.
  Strengthen live recovery and clean-wheel checks.
- Block cross-origin probe credential forwarding; sanitize repr/probe diagnostics
  and reject non-finite inference timeout/backoff settings.
- Close SQLite connections deterministically to prevent leaked file handles,
  particularly during Windows recovery/cleanup.
- Fix Modal image construction: mount local source after build steps, clear the
  inherited server entrypoint, and expose system Python in the Fish S2 image.

Compatibility: existing imports, provider options, old SQLite records and
`autostop_mins` remain supported. The alias defaults to 30 minutes but no longer
determines Modal's container window (independent default: 1800 seconds).
Protected Modal requires Proxy Tokens in addition to SDK credentials. Successful
probes reset local activity only; inference writes throttle to 30 seconds.
`close()` cancels local monitoring without destroying apps. No unified LLM/video client.

## 0.1.0

Initial registry, runtimes, cloud routing, readiness, persistent state, lifecycle and CLI.
