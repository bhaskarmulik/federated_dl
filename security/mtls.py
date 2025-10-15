from __future__ import annotations
from dataclasses import dataclass
import ssl, pathlib

@dataclass
class MTLSConfig:
    ca_cert: pathlib.Path
    cert: pathlib.Path
    key: pathlib.Path

def grpc_ssl_channel_credentials(cfg: MTLSConfig):
    with open(cfg.ca_cert, 'rb') as f: ca = f.read()
    with open(cfg.cert, 'rb') as f: cert = f.read()
    with open(cfg.key, 'rb') as f: key = f.read()
    import grpc
    return grpc.ssl_channel_credentials(root_certificates=ca, private_key=key, certificate_chain=cert)
