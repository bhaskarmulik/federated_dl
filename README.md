**Tiny glossary (so terms make sense)
**
    gRPC: High-performance RPC framework; you call remote functions like normal Python methods.
    
    Proto file: The API contract; gRPC generates code from it.
    
    Stub: The client-side proxy that lets you call server RPCs.
    
    Servicer: The server-side class that implements the RPC methods.
    
    State dict: PyTorch’s dictionary of layer tensors (weights/biases).
    
    FedAvg: Weighted average of clients’ model parameters by their sample counts.

**How the whole system works (simple story)**

Server boots (server/aggregator.py)

Loads a seed global model from disk.

Waits for clients to call.

Client starts (client/site_client.py)

Calls PullGlobal → gets global weights + a training plan (epochs, batch size, lr).

Loads weights into its local model.

Client trains locally

Uses its own private data (never leaves the site).

Produces an update (usually the full state_dict or a delta), counts samples, gathers metrics.

Client sends update

Calls PushUpdate(LocalUpdate).

Server stores it in a pending list.

Server aggregates periodically

Runs FedAvg over the collected updates → new global weights.

Bumps the round_id so the next PullGlobal serves the new model.

Repeat

Over rounds, the global model improves without moving raw data.
