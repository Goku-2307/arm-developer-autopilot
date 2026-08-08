import time

from src.logger import SessionLogger

logger = SessionLogger()

logger.start_stage()

time.sleep(1)

logger.success(
    "Analysis",
    "Project Loaded"
)

logger.start_stage()

time.sleep(2)

logger.success(
    "Detection",
    "Found model.onnx"
)

logger.start_stage()

time.sleep(1)

logger.success(
    "Optimization",
    "INT8 Quantization"
)

print()

print(logger.summary())

print()

for event in logger.get_events():

    print(event)