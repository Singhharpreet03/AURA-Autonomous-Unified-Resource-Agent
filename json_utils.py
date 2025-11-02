import json
import numpy as np
from datetime import datetime, date

def to_json(obj):
    """Convert un-serialisable objects to JSON-native types."""
    if isinstance(obj, (datetime, date)):
        return obj.isoformat()
    if isinstance(obj, (np.float32, np.float64)):
        return float(obj)
    if isinstance(obj, (np.int32, np.int64)):
        return int(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    raise TypeError(f"Type {type(obj)} not serialisable")

def dumps(data, **kw):
    """json.dumps wrapper that handles numpy / datetime."""
    return json.dumps(data, default=to_json, **kw)