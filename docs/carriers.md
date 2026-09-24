# Carrier automation (`tron.carriers`)

Run automated tasks against **Verizon** and **AT&T** through their *official
developer APIs*, distributed across the TRON worker fabric with retries, a
shared rate budget, and the spine's event log for traceability.

This is authorized-use automation for accounts you control — IoT SIM fleets,
enterprise messaging you're provisioned for, usage/account reporting. It is
**not** a way to automate consumer accounts, do SIM-swap/porting, blast
unsolicited messages, or spread requests across workers to dodge a carrier's
rate limits. Those guardrails are built in (see below) and are not optional.

## What it talks to

| Carrier | API | Operations |
|--------|-----|-----------|
| Verizon | ThingSpace / Connectivity Management (IoT M2M) | list / status / activate / deactivate SIMs, usage, account |
| Verizon | Enterprise messaging (if provisioned) | send message |
| AT&T | IoT / Control Center (Jasper-based REST) | get / list / activate / deactivate devices, usage, account |
| AT&T | Enterprise messaging (if provisioned) | send message |

Base URLs and endpoint paths are **overridable per tenant** — configure them
to the exact host and account the carrier provisioned for you. The defaults are
documented starting points, not authoritative endpoints.

## Credentials

Credentials are read from environment variables (see `.env.example`) on
whichever machine executes the op — locally, or a TRON worker when
`distributed=True`. They never travel in a task payload or over the wire:
distributed ops ship a **secret-free** `AdapterConfig`, and each worker
resolves its own credentials via `credentials_from_env()`.

- **Verizon:** `VERIZON_CLIENT_ID`, `VERIZON_CLIENT_SECRET`, `VERIZON_API_KEY`,
  `VERIZON_ACCOUNT_NAME` (OAuth2 client-credentials + ThingSpace session login).
- **AT&T:** `ATT_USERNAME`, `ATT_API_KEY`, `ATT_ACCOUNT_NAME` (HTTP Basic).

`CarrierCredentials` never reveals secret material in `repr`/`str`.

## Usage

```python
from tron.carriers import CarrierGateway, CarrierOp, OpKind

gw = (CarrierGateway(rate_per_sec=5, burst=10)
      .register_verizon()
      .register_att())

# Query an IoT SIM's status (local)
res = gw.manage_sim("verizon", "status", iccid="89148000...")

# Fan a batch of usage queries across the cluster
ops = [CarrierOp(kind=OpKind.USAGE_QUERY, carrier="att",
                 params={"iccid": i}) for i in iccids]
results = gw.map(ops, distributed=True)   # needs a running TRON server + workers

# Send an opt-in notification — consent is required
gw.send_message("att", to=["+1555..."], text="Device online",
                consent=True, distributed=True).get()
```

`manage_sim(action=...)` takes `list | status | activate | deactivate`;
`query(what=...)` takes `usage | network | account`.

## Built-in guardrails

1. **Authorized use only.** Adapters never invent credentials or endpoints;
   with no credentials they raise `CarrierAuthError` rather than guessing.
2. **Consent-gated messaging.** `send_message` refuses to send without an
   explicit `consent=True` marker (`ConsentError`). Send only to recipients who
   opted in.
3. **Honor limits, never evade.** The `RateLimiter` is one shared token-bucket
   budget applied before dispatch, regardless of worker count — for staying
   under the carrier's published ceiling, not exceeding it.
4. **No secret leakage.** Credentials are kept out of logs, task payloads, and
   `repr`/`str`.

## Extending

Add a carrier by subclassing `CarrierAdapter` (implement `credentials_from_env`,
`default_base_url`, `default_endpoints`, `supported`, and `_dispatch`) and
registering the class in `tron/carriers/tasks.py::ADAPTER_CLASSES`. The gateway,
rate limiting, retries, consent gating, and distribution come for free.

See `examples/carriers_demo.py` for a runnable walkthrough.

## CLI

Run ops from the shell (credentials from the environment):

```bash
python -m tron.carriers.cli carriers                     # list configured carriers
python -m tron.carriers.cli sim   --carrier verizon --action status --iccid 8914...
python -m tron.carriers.cli sim   --carrier tmobile --action activate --iccid 8901... --service-plan PLAN_A
python -m tron.carriers.cli query --carrier att --what usage --iccid 8901...
python -m tron.carriers.cli send  --carrier att --to +15551234567 --text "Device online" --consent
```

Add `--distributed` to run an op across a running TRON server + workers instead
of locally. `send` refuses without `--consent`. Results print as JSON; the exit
code is non-zero on a failed op (and `3` on a consent refusal).

## Spine integration (3D Grid replay)

Pass a spine `EventLog` and every op is recorded to the append-only log, so
carrier work appears in the same replay/time-travel substrate the Grid renders:

```python
from tron.spine import EventLog
from tron.carriers import CarrierGateway

gw = CarrierGateway(event_log=EventLog(), node_id="edge-1").register_att()
gw.manage_sim("att", "status", iccid="8901...")
# -> emits carrier_op_submitted then carrier_op_completed under one task_id
```

Recipient phone numbers and ICCIDs are **redacted to a short content hash**
before they reach the log, and message bodies are recorded only as a length —
the stream stays traceable without storing PII or secrets.

## Carriers

`verizon` (ThingSpace), `att` (Control Center), and `tmobile` (IoT / Control
Center) are built in. Register the ones you use:
`gw.register_verizon()`, `gw.register_att()`, `gw.register_tmobile()`.
