"""Use twenty literal policies; reuse only the tool/world logic from v1.
Statuses are translated consistently across user text, calls, results and descriptions.
Each domain is durably appended before the next domain begins. No GPU work.
"""
import argparse,collections,copy,hashlib,json,os,re,time
from pathlib import Path
from policies_v2 import POLICIES
from generate import FLOWS,OUT as OLD,validate_record,write_json,FORBIDDEN_KEYS,FORBIDDEN_TOOLS,FORBIDDEN_NUMBERS
from prepare_candidates import INTENT,WRITES
from audit import verify,numeric_values
from time_budget import deadline_reached

OUT=Path('data_work/processed/gpt_policy_v2')
STATUS={
 'gpmaintenance-lock':'maintenance lock','gpwithin-scope':'within scope',
 'gpcapacity-shortfall':'capacity shortfall','gpcapacity-clear':'capacity sufficient',
 'gproute-check':'route review','gplifecycle-window':'cancellation window',
 'gpscope-check':'scope review','gpfit-check':'fitting incompatibility',
 'gpevidence-review':'evidence review','gpcoupling-check':'connector incompatibility',
 'gpquota-freeze':'quota frozen','gpreplay-inspection':'replay inspection',
 'gpclear':'eligible','gpstop':'blocked','gpaccept':'accepted',
 'gpsecondary':'secondary','gpprimary':'primary','gpunusable':'unavailable',
 'gpsealed':'sealed','gpprovisional':'provisional','gpreviewer':'reviewer','gpdecider':'decision maker',
 'gpreleased':'released','gpweave-proof':'supporting evidence','gpretry-inspect':'inspection required',
 'gpcompleted':'completed','gpunmodified':'unchanged','gpdocumentation-only':'documentation only'}

def translate(value,reverse=False):
    pairs=list(STATUS.items())
    if reverse:pairs=[(b,a) for a,b in pairs]
    if isinstance(value,str):
        if value.lstrip().startswith(('{','[')):
            try:
                structured=json.loads(value)
                return json.dumps(translate(structured,reverse),ensure_ascii=False)
            except ValueError:pass
        # longest-first, simultaneous replacement prevents replacements being rewritten again
        pattern=re.compile('|'.join(re.escape(a) for a,b in sorted(pairs,key=lambda p:-len(p[0]))))
        mapping=dict(pairs)
        return pattern.sub(lambda m:mapping[m.group(0)],value)
    if isinstance(value,list):return [translate(x,reverse) for x in value]
    if isinstance(value,dict):return {k:(v if k in ('name','id','family','source') else translate(v,reverse)) for k,v in value.items()}
    return value

def add_decoys(row,flow):
    write=WRITES[flow]
    target=next(t for t in row['tools'] if write.replace('_','').lower() in t['function']['name'].replace('_','').lower())
    for prefix,description in [('preview','Preview this operation without performing it. A proposal is not a completed write.'),
                               ('history','Read earlier outcomes of this operation. History does not establish current eligibility.'),
                               ('simulate','Simulate the operation without changing the current record or granting permission.')]:
        t=copy.deepcopy(target);t['function']['name']=prefix+'_'+t['function']['name'];t['function']['description']=description
        assert t['function']['name'] not in FORBIDDEN_TOOLS
        row['tools'].append(t)

def verify_v2(row):
    validate_record(row)
    # Independent v1 checks inspect visible evidence. Translate only for that test;
    # canonical gp statuses never appear in the exported model input.
    verify(translate(row,True))
    assert not any(marker in json.dumps(row,ensure_ascii=False) for marker in STATUS)
    for t in row['tools']:
        assert not set(t['function']['parameters']['properties'])&FORBIDDEN_KEYS
    assert not set(numeric_values({'messages':row['messages'],'tools':row['tools']}))&FORBIDDEN_NUMBERS

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--deadline',help='Optional ISO timestamp with UTC offset')
    args=parser.parse_args()
    deadline_reached(args.deadline)
    resuming=OUT.exists()
    if not resuming:OUT.mkdir(parents=True,exist_ok=False)
    domains=json.loads((OLD/'manifest.json').read_text())['domains']
    rows={}
    for split in ['train','dev']:
        for line in (OLD/f'{split}_raw.jsonl').open():
            r=json.loads(line);di=int(r['id'].split('_')[1]);rows.setdefault(di,[]).append(r)
    manifest={'status':'generating','policy_authorship':'Twenty full, independently authored literal strings in policies_v2.py; no shared-clause policy assembly',
              'abstract_workflows':{a:b for a,b,_ in FLOWS},'domains':[],'excluded_numeric_coincidences':[],
              'source_policy_sha256':hashlib.sha256(Path(__file__).with_name('policies_v2.py').read_bytes()).hexdigest(),
              'generation_deadline':'2026-11-09 22:30 Asia/Taipei','handoff_deadline':'2026-11-09 22:50 Asia/Taipei'}
    if resuming:
        previous=json.loads((OUT/'manifest.json').read_text())
        assert previous['source_policy_sha256']==manifest['source_policy_sha256']
        manifest=previous
        manifest['resume_note']='Status normalization audit was corrected to preserve tool names; completed domains retained.'
    write_json(OUT/'manifest.json',manifest)
    for d in domains:
        if deadline_reached(args.deadline):break
        di,flow,domain,split=d['index'],d['flow'],d['domain'],d['split']
        if any(x['index']==di for x in manifest['domains']):continue
        policy=POLICIES[di];assert 1500<=len(policy)<=3500
        records=[];counts=collections.Counter()
        for original in rows[di]:
            collisions=set(numeric_values({'messages':original['messages'],'tools':original['tools']}))&FORBIDDEN_NUMBERS
            if collisions:
                manifest['excluded_numeric_coincidences'].append(original['id']);continue
            r=copy.deepcopy(original);n=int(r['id'].rsplit('_',1)[1]);w=json.loads((OLD/f'worlds_{di:02d}.json').read_text())[n]['world']
            r['messages'][1]['content']=INTENT[flow][n%3].format(domain=domain,**w)+f' My stated purpose is "{w["reason"]}".'
            add_decoys(r,flow);r=translate(r);r['messages'][0]['content']=policy
            r['id']=r['id'].replace('gp1_','gp2_',1);r['family']=r['family'].replace('gp1_','gp2_',1);r['source']='gpt_policy_v2'
            verify_v2(r);records.append(r);counts['records']+=1
            counts['targets']+=sum(m['role']=='assistant' and i not in r['no_target'] for i,m in enumerate(r['messages']))
            counts['call_after_tool']+=sum(m['role']=='assistant' and bool(m.get('tool_calls')) and r['messages'][i-1]['role']=='tool' for i,m in enumerate(r['messages']))
            counts['refusals']+=n in (7,21)
        with (OUT/f'{split}_raw.jsonl').open('a',encoding='utf-8') as f:
            for r in records:f.write(json.dumps(r,ensure_ascii=False)+'\n')
            f.flush();os.fsync(f.fileno())
        manifest['domains'].append({'index':di,'domain':domain,'flow':flow,'split':split,'policy_chars':len(policy),**dict(counts)})
        write_json(OUT/'manifest.json',manifest)
        print('SAVED',di,domain,split,dict(counts),flush=True)
    manifest['status']='raw_complete';write_json(OUT/'manifest.json',manifest)

if __name__=='__main__':main()
