"""TRON carrier integration — distributed automation over official carrier APIs.

Talk to Verizon's and AT&T's **official developer APIs** (ThingSpace /
Connectivity Management, AT&T IoT / Control Center, and configurable enterprise
messaging) and run those calls across the TRON worker fabric.

Quick start::

    from tron.carriers import CarrierGateway

    gw = (CarrierGateway(rate_per_sec=5)
          .register_verizon()          # creds from VERIZON_* env vars
          .register_att())             # creds from ATT_* env vars

    # Query an IoT SIM's status (runs locally)
    res = gw.manage_sim("verizon", "status", iccid="89148000...")

    # Send opt-in notifications across the cluster, honoring the rate budget
    res = gw.send_message("att", to=["+1555..."], text="Your device is online",
                          consent=True, distributed=True).get()

Authorized use only: configure real credentials/endpoints for accounts you
control, honor carrier rate limits, and only message recipients who opted in.
"""

from .base import (
    CarrierAdapter,
    CarrierAPIError,
    CarrierAuthError,
    CarrierConfigError,
    CarrierCredentials,
    CarrierError,
    CarrierOp,
    CarrierResult,
    ConsentError,
    OpKind,
    RateLimiter,
    RateLimitExceeded,
)
from .verizon import VerizonAdapter
from .att import ATTAdapter
from .tmobile import TMobileAdapter
from .tasks import AdapterConfig, run_op, ADAPTER_CLASSES
from .gateway import CarrierGateway

__all__ = [
    "CarrierGateway",
    "CarrierAdapter",
    "VerizonAdapter",
    "ATTAdapter",
    "TMobileAdapter",
    "AdapterConfig",
    "CarrierOp",
    "CarrierResult",
    "OpKind",
    "CarrierCredentials",
    "RateLimiter",
    "run_op",
    "ADAPTER_CLASSES",
    # errors
    "CarrierError",
    "CarrierAuthError",
    "CarrierConfigError",
    "CarrierAPIError",
    "ConsentError",
    "RateLimitExceeded",
]
