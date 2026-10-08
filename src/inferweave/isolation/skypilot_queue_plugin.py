"""SkyPilot 0.13 queue plugin: every account uses its own request-queue port.

Loaded through SkyPilot's public BasePlugin/PluginContext API in the SDK interpreter.
It needs no InferWeave imports. Server/metrics ports alone are insufficient: SkyPilot's
default multiprocessing queue otherwise uses port 50011 for every server.
"""

import logging
import os
from pathlib import Path
from typing import Any

from sky.server.plugins import BasePlugin, PluginContext  # type: ignore
from sky.server.requests.queues.base import MultiprocessingQueueFactory  # type: ignore


class AccountQueuePlugin(BasePlugin):
    def __init__(self, port: int) -> None:
        self.port = port

    def install(self, context: PluginContext) -> None:
        context.register_queue_backend_factory(MultiprocessingQueueFactory(port=self.port))
        # The SDK may log HTTP response bodies containing keys. Protect every logger,
        # including its private on-disk logs; raw stderr/stdout are also discarded.
        secrets = [os.environ.get("RUNPOD_API_KEY", "")]
        path = Path.home() / ".config" / "vastai" / "vast_api_key"
        if path.exists():
            secrets.append(path.read_text().strip())
        factory = logging.getLogRecordFactory()

        def safe_record(*args: Any, **kwargs: Any) -> logging.LogRecord:
            record = factory(*args, **kwargs)
            message = record.getMessage()
            for secret in secrets:
                if secret:
                    message = message.replace(secret, "***")
            record.msg, record.args = message, ()
            record.exc_info = None
            record.exc_text = None
            record.stack_info = None
            return record

        logging.setLogRecordFactory(safe_record)
