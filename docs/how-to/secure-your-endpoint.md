# Secure your endpoint

InferWeave protects Modal endpoints by default with
`modal.web_server(..., requires_proxy_auth=True)`. Deploy fails early with
`ProviderAuthError` if endpoint credentials cannot resolve. Dry runs need no tokens.

| Purpose | Configuration | Used by |
| --- | --- | --- |
| Manage Modal resources | `modal setup` profile or `MODAL_TOKEN_ID` + `MODAL_TOKEN_SECRET` | Modal SDK |
| Access protected HTTP | `MODAL_PROXY_TOKEN_ID` + `MODAL_PROXY_TOKEN_SECRET` | Probes and inference |

Create Proxy Tokens in [Modal Settings](https://modal.com/docs/guide/webhook-proxy-auth).
The default `CompositeEndpointAuth(ModalProxyAuth())` reads environment variables
at request time, allowing rotation. Explicit values override environment values;
`token_getter` overrides both. A missing half of a pair produces no headers.
`ModalProxyAuth` only sends tokens for provider `modal` on `*.modal.run`/`*.modal.host`.

```python
import os
from inferweave import InferWeave, ModalProxyAuth

weave = InferWeave(endpoint_auth=ModalProxyAuth(
    token_id=os.environ["MODAL_PROXY_TOKEN_ID"],
    token_secret=os.environ["MODAL_PROXY_TOKEN_SECRET"],
))
```

The resolver authenticates readiness, `wait_for_ready()`, `check_health()`, status
probes and inference after attach. SDK-level auth is adopted by supplied routers
whose providers have no explicit resolver. If a provider has its own auth, configure
matching SDK auth for subsequent requests.

For a custom API key use `StaticHeaderAuth`, scoped by provider, or implement
`EndpointAuthPort.headers_for(provider, endpoint_url) -> dict[str, str]`.
Use a host-checking resolver for multiple trust domains; static auth does not
constrain hosts. Composite resolvers merge headers, with later values winning.
`NoEndpointAuth()` produces no headers.

```python
import os
from inferweave import InferWeave, StaticHeaderAuth

weave = InferWeave(endpoint_auth=StaticHeaderAuth(
    {"X-Api-Key": os.environ["ENDPOINT_API_KEY"]}, providers=("runpod",),
))
```

Auth resolvers stay in memory; resolved endpoint credentials never enter records.
Stored sensitive provider fields and runtime environment keys are redacted.
Auth repr hides values; inference diagnostics redact resolved headers and sanitize
URLs. Health/inference transports refuse cross-origin redirects, including
HTTPS downgrades, and bound same-origin redirect chains.
Do not log headers, secret-bearing raw status URLs, or HTTP wire debug data in
your application. Use header auth instead of query/userinfo credentials; do not
put secrets in engine CLI arguments.

## Deliberately public endpoints

```python
from inferweave import InferWeave, NoEndpointAuth

async def deploy_public():
    weave = InferWeave(endpoint_auth=NoEndpointAuth())
    try:
        deployment = await weave.deploy(
            model="fish-s2-pro", provider="modal",
            custom_args={"requires_proxy_auth": False, "cleanup_on_failure": True},
        )
        try:
            return await deployment.synthesize("Public endpoint")
        finally:
            await deployment.stop()
    finally:
        await weave.close()
```

Anyone with the URL can access GPU inference, increasing abuse/billing exposure.
`NoEndpointAuth()` alone does not remove server protection. Workspace policy may
forbid public URLs. Keep protection enabled for services.
