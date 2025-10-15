from __future__ import annotations
import argparse, yaml, json, os, sys

def main():
    p = argparse.ArgumentParser()
    p.add_argument('command', choices=['sim:centralized', 'sim:p2p'])
    p.add_argument('--config', default='configs/mvp.yaml')
    p.add_argument('--clients', type=int, default=10)
    p.add_argument('--alpha', type=float, default=0.5)
    p.add_argument('--sync', action='store_true')
    p.add_argument('--async', dest='asyn', action='store_true')
    p.add_argument('--B', type=int, default=6)
    p.add_argument('--T', type=int, default=3000)
    p.add_argument('--Smax', type=int, default=2)
    p.add_argument('--epochs', type=int, default=1)
    p.add_argument('--batch-size', type=int, default=64)
    p.add_argument('--optimizer', default='adam')
    p.add_argument('--lr', type=float, default=1e-3)
    p.add_argument('--secure-agg', action='store_true')
    p.add_argument('--two-phase-commit', action='store_true')
    args = p.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    print('Command:', args.command)
    print('Args:', vars(args))
    print('Loaded config:', json.dumps(cfg, indent=2))

    if args.command == 'sim:centralized':
        print('TODO: launch coordinator + clients (sync={} async={})'.format(bool(args.sync), bool(args.asyn)))
    else:
        print('TODO: launch P2P PushSum simulation')

if __name__ == '__main__':
    main()
