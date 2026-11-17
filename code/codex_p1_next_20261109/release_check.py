"""CPU-only full rendering/token validation and manifest; no training launch."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'data'))
sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'hw3'))
from common import expand_steps
from eval import format as F

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--dir',required=True);a=ap.parse_args()
    root=Path(a.dir);tok=F.tokenizer(); result={}
    for split in ['train','stress','challenge']:
        p=root/f'{split}.jsonl';lengths=[];broken=0;targets=[];families=set();ids=set()
        for line in p.open(encoding='utf-8'):
            r=json.loads(line);families.add(r['family']);assert r['id'] not in ids;ids.add(r['id'])
            for step in expand_steps(r):
                m=step['messages'];opts={'tools':step['tools'],'tokenize':False,'enable_thinking':False}
                prompt=tok.apply_chat_template(m[:-1],add_generation_prompt=True,**opts)
                full=tok.apply_chat_template(m,**opts)
                broken+=not full.startswith(prompt)
                lengths.append(len(tok(full,add_special_tokens=False)['input_ids']))
                targets.append(len(tok(full[len(prompt):],add_special_tokens=False)['input_ids']))
        result[split]={'path':str(p),'sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'records':len(ids),'families':sorted(families),'training_targets':len(lengths),'max_tokens':max(lengths),'p99_tokens':sorted(lengths)[int(.99*len(lengths))],'over_8192':sum(n>8192 for n in lengths),'max_target_tokens':max(targets),'broken_template_prefix':broken}
        assert not broken and not result[split]['over_8192']
        print(split,json.dumps(result[split]),flush=True)
    for a in result:
        for b in result:
            if a!=b: assert not set(result[a]['families'])&set(result[b]['families'])
    with (root/'render_manifest.json').open('x',encoding='utf-8') as f:json.dump(result,f,indent=2)

if __name__=='__main__':main()
