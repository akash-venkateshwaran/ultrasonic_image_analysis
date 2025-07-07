import ray
import time
from loguru import logger
# Run Ray in local mode (single process)
context = ray.init(
        ignore_reinit_error=True,
        include_dashboard=True,
        dashboard_port=8265,
        num_cpus=1,
        num_gpus=1
    )
    
    # Get dashboard URL from context
dashboard_url = context.dashboard_url
logger.info(f"Ray Dashboard available at: http://localhost:{8265}") # NOTE this works because I have forwared the docker port to localhost port 8265

@ray.remote
def my_function():
    return 1

@ray.remote
def slow_function():
    time.sleep(10)
    return 1

# This will run sequentially in local mode
obj_refs = []
for _ in range(4):
    obj_ref = slow_function.remote()
    obj_refs.append(obj_ref)

results = ray.get(obj_refs)
print(f"Results: {results}")

ray.shutdown()