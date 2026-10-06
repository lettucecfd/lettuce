"""
Score the outflow boundary in boundary.py. DO NOT MODIFY.

    uv run --extra cpu python autoresearch/vortex_cbc/evaluate.py > run.log 2>&1
    grep "^score:" run.log
"""
from prepare import evaluate
from boundary import make_outlet

result = evaluate(make_outlet)
print("---")
for key, value in result.items():
    print(f"{key}: {value:.6f}" if isinstance(value, float) else f"{key}: {value}")
