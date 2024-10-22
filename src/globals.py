from typing import Dict, Any
from adapter_idefics import IdeficsAdapter

record_caches: Dict[str, Any] = None
adapter: IdeficsAdapter = None
running_stats: Dict[str, Any] = None
xargs: Any = None
reward_decoder_lib: Any = None
policy_lib: Any = None
