"""Synthetic backbone-only CUDA benchmark, not a DALES accuracy experiment."""
import argparse
from pathlib import Path
import runpy
import time
import torch
from .model import SerializedLocalCross


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--points",type=int,default=65536)
    parser.add_argument("--batch",type=int,default=4)
    parser.add_argument("--steps",type=int,default=5)
    parser.add_argument("--warmup",type=int,default=2)
    parser.add_argument("--config",default=str(Path(__file__).with_name("config.py")))
    parser.add_argument("--checkpoint",action="store_true")
    args=parser.parse_args()
    if not torch.cuda.is_available():
        parser.error("CUDA GPU required")
    if min(args.points,args.batch,args.steps,args.warmup)<1:
        parser.error("Points, batch, steps and warmup must be positive")
    cfg=runpy.run_path(args.config)["model"]["backbone"].copy()
    cfg.pop("type")
    cfg["checkpoint_blocks"]=args.checkpoint
    model=SerializedLocalCross(**cfg).cuda().train()
    optimizer=torch.optim.AdamW(model.parameters(),lr=.001)
    count=args.points*args.batch
    coord=torch.rand(count,3,device="cuda")*torch.tensor([40.,40.,5.],device="cuda")
    feat=torch.cat((coord,torch.rand(count,1,device="cuda")),-1)
    data=dict(coord=coord,feat=feat,offset=torch.arange(1,args.batch+1,device="cuda")*args.points)
    scaler=torch.amp.GradScaler("cuda")
    def step():
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast("cuda",dtype=torch.float16):
            loss=model(data).float().square().mean()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
    for _ in range(args.warmup):
        step()
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    start=time.perf_counter()
    for _ in range(args.steps):
        step()
    torch.cuda.synchronize()
    elapsed=time.perf_counter()-start
    print(f"GPU: {torch.cuda.get_device_name()}")
    print(f"Points/step: {count:,}; seconds/step: {elapsed/args.steps:.3f}")
    print(f"Peak allocated GiB: {torch.cuda.max_memory_allocated()/2**30:.3f}")
    print(f"Peak reserved GiB: {torch.cuda.max_memory_reserved()/2**30:.3f}")
    model.eval()
    with torch.no_grad(),torch.autocast("cuda",enabled=False):
        output=model(data)
    if not bool(torch.isfinite(output).all()):
        raise RuntimeError("Nonfinite FP32 evaluation output")
    print("Evaluation without autocast: finite output")


if __name__=="__main__":
    main()
