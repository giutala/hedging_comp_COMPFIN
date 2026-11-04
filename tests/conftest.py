from __future__ import annotations

import asyncio
import sys

# On Windows the default Proactor event loop lacks add_reader, which pyzmq needs; under
# nbclient the kernel's execute reply is then never received and notebook tests hang.
if sys.platform == "win32":
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())  # type: ignore[attr-defined,unused-ignore]
