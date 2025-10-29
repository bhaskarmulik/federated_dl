import io, torch

def fedavg(updates):
    """
    updates: list of tuples (state_dict, num_samples)
    returns: aggregated state_dict
    """
    total = sum(n for _, n in updates)
    base_sd = {k: torch.zeros_like(v, dtype=v.dtype) for k, v in updates[0][0].items()}

    for sd, n in updates:
        for k in base_sd:
            # Only average floating-point tensors
            if torch.is_floating_point(base_sd[k]):
                base_sd[k] += sd[k] * (n / total)
            else:
                # Just take from first model (e.g., batch norm counters)
                base_sd[k] = sd[k]

    return base_sd


def sd_to_bytes(sd) -> bytes:
    buf = io.BytesIO()
    torch.save(sd, buf)
    return buf.getvalue()

def bytes_to_sd(b: bytes):
    buf = io.BytesIO(b)
    return torch.load(buf, map_location="cpu")
