import io, torch, numpy as np

def state_dict_to_bytes(model) -> bytes:
    buf = io.BytesIO()
    torch.save(model.state_dict(), buf)   # uses pickle + tensor storage
    return buf.getvalue()

def bytes_to_state_dict(model, b: bytes):
    buf = io.BytesIO(b)
    sd = torch.load(buf, map_location="cpu")
    model.load_state_dict(sd)

def numpy_to_bytes(weight_dict: dict[str, np.ndarray]) -> bytes:
    buf = io.BytesIO()
    np.savez_compressed(buf, **weight_dict)
    return buf.getvalue()

def bytes_to_numpy(b: bytes) -> dict:
    buf = io.BytesIO(b)
    with np.load(buf, allow_pickle=False) as data:
        return {k: data[k] for k in data.files}
