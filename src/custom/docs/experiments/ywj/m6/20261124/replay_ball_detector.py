# Evaluation only (never imported by mission code): replay detect_balls on
# frames captured with RAI_M6_CAPTURE=<dir>. Usage: python replay_ball_detector.py <src> <capture_dir> [out_png_dir]
import sys, numpy as np, cv2
from pathlib import Path
sys.path.insert(0, sys.argv[1])
from custom.vision import detect_balls
d=Path(sys.argv[2]); out=Path(sys.argv[3]) if len(sys.argv)>3 else None
if out: out.mkdir(parents=True,exist_ok=True)
for f in sorted(d.glob('frame_*.npz')):
    z=np.load(f); T=z['T_world_pelvis']
    c=detect_balls(z['rgb'],z['depth'],z['world'],z['height'],z['valid'],fx=float(z['K'][0]),viewer_xy=T[:2,3])
    c.sort(key=lambda r:-r.score)
    s=' | '.join('(%.2f,%.2f) s=%.2f rms=%.3f top=%.2f w=%.2f k=%.2f d=%.1f'%(r.x,r.y,r.score,r.rms,r.top,r.white,r.dark,r.distance) for r in c)
    print(f.stem,'pose=(%.2f,%.2f)'%(T[0,3],T[1,3]),'legacy=',np.round(z['chosen'][:2],2),'->',s or 'none')
    if out:
        vis=cv2.cvtColor(z['rgb'],cv2.COLOR_RGB2BGR)
        for r in c: cv2.circle(vis,tuple(int(v) for v in r.pixel),12,(0,0,255),2)
        cv2.imwrite(str(out/(f.stem+'.png')),vis)
