from exp.utils.time import utc_to_cst
from datetime import datetime, timezone
import pyarrow.dataset as ds
import pyarrow as pa

# Test with naive
naive = datetime(2025, 6, 6, 12, 0, 0)
converted_naive = utc_to_cst(naive).replace(tzinfo=None)
print(f"Naive Input -> Converted: {converted_naive}, tzinfo: {converted_naive.tzinfo}")

# Test with aware
aware = datetime(2025, 6, 6, 12, 0, 0, tzinfo=timezone.utc)
converted_aware = utc_to_cst(aware).replace(tzinfo=None)
print(f"Aware Input -> Converted: {converted_aware}, tzinfo: {converted_aware.tzinfo}")

# Test pyarrow expression
expr = ds.field("time") >= converted_naive
print(f"Expression: {expr}")
# Check literal type if possible
# Unfortunately simpler to just run it.
