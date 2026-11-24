"""CPU/file rendezvous for 4GPU GRPO rollouts BEFORE entering NCCL.
Avoids the Rank0/Rank1 early-barrier 90min watchdog failure.
"""
from __future__ import annotations
import json, os, time
from pathlib import Path

def before_collective(root, round_id: int, rank: int, world: int, n_groups: int) -> None:
    if world == 1:
        return  # No inter-rank rendezvous in a single-rank run.
    token=os.environ.get("COUNT_RL_FILE_RENDEZVOUS_ID")
    if not token:
        raise RuntimeError("Missing COUNT_RL_FILE_RENDEZVOUS_ID for multi-rank training; set a shared, unique token at startup")
    if not token.replace("_","").replace("-","").isalnum():
        raise RuntimeError(f"Unsafe rendezvous token {token!r}")
    directory=Path(root)/"rollout_ready"/token/f"round{round_id}"
    directory.mkdir(parents=True,exist_ok=True)
    out=directory/f"rank{rank}.json"
    tmp=directory/f"rank{rank}.{os.getpid()}.tmp"
    tmp.write_text(json.dumps(dict(rank=rank,world=world,round=round_id,
        group_count=n_groups,pid=os.getpid(),at=time.time()))+"\n")
    os.replace(tmp,out)
    print(f"[MULTIRANK_FILE_RENDEZVOUS] rank={rank}/{world} round={round_id} local_groups={n_groups} ready",flush=True)
    last=0.0;start=time.monotonic()
    while True:
        try:
            records=[json.loads((directory/f"rank{i}.json").read_text()) for i in range(world)]
            if all(x.get("world")==world and x.get("round")==round_id and
                   x.get("group_count")==n_groups for x in records):
                print(f"[MULTIRANK_ENTER_NCCL] rank={rank} round={round_id} all_ranks_ready",flush=True)
                return
            raise RuntimeError("Rendezvous data mismatch: "+repr(records))
        except FileNotFoundError:
            pass
        now=time.monotonic()
        if now-last>=120:
            n=sum((directory/f"rank{i}.json").is_file() for i in range(world))
            print(f"[MULTIRANK_FILE_WAIT] rank={rank} round={round_id} {n}/{world} ranks",flush=True)
            last=now
        if now-start>24*3600:
            raise TimeoutError("Long-running rollout rendezvous exceeded 24h")
        time.sleep(10)
