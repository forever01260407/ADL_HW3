"""Fresh worlds in twenty new domains; literal policy text, CPU only, durable per-domain output."""
import argparse,collections,copy,hashlib,json,os,re,time
from pathlib import Path
from policies_v3 import POLICIES,DOMAINS
from generate import World,FLOWS,write_json,FORBIDDEN_KEYS,FORBIDDEN_TOOLS,FORBIDDEN_NUMBERS,PUBLIC_TEXT
from build_v2 import translate,STATUS,add_decoys,verify_v2
from prepare_candidates import INTENT
from audit import numeric_values
from time_budget import deadline_reached

OUT=Path('data_work/processed/gpt_policy_v3')
PREFIXES=['kite','whistle','ribbon','sugar','enamel','reed','rubbing','gilding','basket','chime','bobbin','sand','porcelain','tessera','cane','teacake','handbell','puppet','woodblock','bead']

def aliases(di):
    old='zq'+chr(97+di);new='v3_'+PREFIXES[di]
    return {old:new,old.upper():('V3'+PREFIXES[di].upper()),'GPW-':'OPV-','GPR-':'REC-','GPD-':'DST-','GPC-':'SEAL-','gpweave-':'evidence-'}

def remap(value,di,reverse=False):
    pairs=aliases(di)
    if reverse:pairs={v:k for k,v in pairs.items()}
    if isinstance(value,str):
        if value.lstrip().startswith(('{','[')):
            try:return json.dumps(remap(json.loads(value),di,reverse),ensure_ascii=False)
            except ValueError:pass
        regex=re.compile('|'.join(re.escape(k) for k in sorted(pairs,key=lambda x:-len(x))))
        return regex.sub(lambda m:pairs[m.group(0)],value)
    if isinstance(value,list):return [remap(v,di,reverse) for v in value]
    if isinstance(value,dict):return {remap(k,di,reverse):remap(v,di,reverse) for k,v in value.items()}
    return value

def verify_v3(row):
    di=int(row['id'].split('_')[1]);verify_v2(remap(row,di,True))
    assert row['messages'][0]['content']==POLICIES[di]
    assert not any(k in json.dumps(row) for k in STATUS)
    for t in row['tools']:
        f=t['function'];assert f['name'] not in FORBIDDEN_TOOLS;assert not set(f['parameters']['properties'])&FORBIDDEN_KEYS

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--deadline',help='Optional ISO timestamp with UTC offset')
    args=parser.parse_args()
    deadline_reached(args.deadline)
    resuming=OUT.exists()
    if not resuming:OUT.mkdir(parents=True,exist_ok=False)
    assert len(DOMAINS)==20 and len(set(DOMAINS))==20
    old_domains={d for a,b,ds in FLOWS for d in ds};assert not old_domains&set(DOMAINS)
    manifest={'status':'generating','source':'gpt_policy_v3','domains':[],'excluded_numeric':[],
              'abstract_workflows':{a:b for a,b,ds in FLOWS},'policy_authorship':'Twenty separately authored full string constants, no shared-clause assembly',
              'policy_source_sha256':hashlib.sha256(Path(__file__).with_name('policies_v3.py').read_bytes()).hexdigest(),
              'hard_deadline':'2026-11-09 22:58 Asia/Taipei','optional_new_flows':'Not added: prioritize all twenty domains and complete checks within the deadline.'}
    if resuming:
        previous=json.loads((OUT/'manifest.json').read_text())
        assert previous['policy_source_sha256']==manifest['policy_source_sha256'];manifest=previous
    write_json(OUT/'manifest.json',manifest)
    for di,domain in enumerate(DOMAINS):
        if deadline_reached(args.deadline):break
        assert domain not in PUBLIC_TEXT
        if any(d['index']==di for d in manifest['domains']):continue
        flow=FLOWS[di//2][0];split='dev' if di in [3,9,17] else 'train';records=[];counts=collections.Counter()
        assert 1500<=len(POLICIES[di])<=3500
        for n in range(26):
            w=World(domain,flow,di,n)
            old_owner=w.w['owner'];w.w['owner']=f'Oriven{chr(65+di)} Velset{n:03d}'
            w.msg[3]['content']=w.msg[3]['content'].replace(old_owner,w.w['owner'])
            result=w.execute();w.finish(result)
            for op,desc in [('erase_record','Erase the current record; not a normal workflow action.'),('rewrite_owner','Change ownership; outside the current request.'),('waive_gate','Seek a gate waiver which this policy does not permit.'),('archive_copy','Archive a copy without performing the requested write.')]:
                w.register(op,{'record':'string'},desc,lambda a:{'record':a['record']})
            while len(w.schemas)<12:w.register('diagnostic_'+chr(97+len(w.schemas)),{},'Read general documentation without satisfying any required operational check.',lambda a:{'note':'documentation only'})
            r={'id':f'gp3_{di:02d}_{flow}_{n:03d}','family':f'gp3_{di:02d}_{flow}','source':'gpt_policy_v3','task':'part1','tools':w.schemas,'messages':w.msg,'no_target':[i for i,m in enumerate(w.msg) if m['role']=='assistant' and w.msg[i-1]['role']=='user']}
            r['messages'][1]['content']=INTENT[flow][n%3].format(domain=domain,**w.w)+f' My stated purpose is "{w.w["reason"]}".'
            add_decoys(r,flow);r=remap(translate(r),di);r['messages'][0]['content']=POLICIES[di]
            if set(numeric_values({'messages':r['messages'],'tools':r['tools']}))&FORBIDDEN_NUMBERS:
                manifest['excluded_numeric'].append(r['id']);continue
            verify_v3(r);records.append(r);counts['records']+=1;counts['refusals']+=w.w['halt']
            for i,m in enumerate(r['messages']):
                if m['role']=='assistant' and i not in r['no_target']:
                    counts['targets']+=1;counts['call_after_tool' if m.get('tool_calls') else 'text_after_tool']+=1
        with (OUT/f'{split}_raw.jsonl').open('a',encoding='utf-8') as f:
            for r in records:f.write(json.dumps(r,ensure_ascii=False)+'\n')
            f.flush();os.fsync(f.fileno())
        manifest['domains'].append({'index':di,'domain':domain,'flow':flow,'split':split,'policy_chars':len(POLICIES[di]),**dict(counts)})
        write_json(OUT/'manifest.json',manifest);print('SAVED',di,domain,split,dict(counts),flush=True)
    manifest['status']='raw_complete';write_json(OUT/'manifest.json',manifest)

if __name__=='__main__':main()
