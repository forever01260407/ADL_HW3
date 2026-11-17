"""Final dataset, policy-literal and contract audit; writes only the v2 data directory."""
import ast,collections,copy,hashlib,json,re
from pathlib import Path
from policies_v2 import POLICIES
from build_v2 import OUT,STATUS,verify_v2
from generate import write_json

def grams(text):
    w=re.findall(r'\w+',text.lower());return {tuple(w[i:i+8]) for i in range(len(w)-7)}

def main():
    source=Path(__file__).with_name('policies_v2.py');tree=ast.parse(source.read_text())
    assignment=next(n for n in tree.body if isinstance(n,ast.Assign))
    assert isinstance(assignment.value,ast.Dict) and all(isinstance(v,ast.Constant) and isinstance(v.value,str) for v in assignment.value.values)
    assert set(POLICIES)==set(range(20)) and all(1500<=len(v)<=3500 for v in POLICIES.values())
    ga={i:grams(p) for i,p in POLICIES.items()};pairs=[]
    for i in ga:
        for j in ga:
            if i<j:pairs.append((len(ga[i]&ga[j])/len(ga[i]|ga[j]),i,j))
    stats={};family_sets={};mutations=collections.Counter()
    for split in ['train','dev']:
        rows=[json.loads(l) for l in (OUT/f'{split}.jsonl').open()];c=collections.Counter();families=set();domains=set();ids=set();policy_lengths=[];tool_counts=[]
        for row in rows:
            assert row['id'] not in ids;ids.add(row['id']);verify_v2(row)
            di=int(row['id'].split('_')[1]);domains.add(di);families.add(row['family'])
            assert row['messages'][0]['content']==POLICIES[di]
            policy_lengths.append(len(POLICIES[di]));tool_counts.append(len(row['tools']))
            for i,m in enumerate(row['messages']):
                if m['role']=='assistant' and i not in row['no_target']:
                    c['targets']+=1;c['call_after_tool' if m.get('tool_calls') else 'text_after_tool']+=1
            c['records']+=1;c['refusals']+=row['messages'][-2]['role']=='tool' and 'referred for' in row['messages'][-2]['content']
            # A seen-but-wrong string must fail even when it fits the schema.
            mutant=copy.deepcopy(row)
            target=next(m for i,m in enumerate(mutant['messages']) if m.get('tool_calls') and i not in mutant['no_target'])
            arguments=target['tool_calls'][0]['function']['arguments'];key=next(iter(arguments));arguments[key]='not-a-returned-reference'
            try:verify_v2(mutant)
            except (AssertionError,KeyError,ValueError):mutations['wrong_argument_detected']+=1
            else:raise AssertionError('wrong argument escaped audit')
        family_sets[split]=families
        stats[split]={**dict(c),'domains':len(domains),'families':len(families),'policy_chars_min':min(policy_lengths),'policy_chars_max':max(policy_lengths),'tools_min':min(tool_counts),'tools_max':max(tool_counts),'sha256':hashlib.sha256((OUT/f'{split}.jsonl').read_bytes()).hexdigest()}
    assert not family_sets['train']&family_sets['dev'];assert stats['train']['records']>=300
    assert sum(s['refusals'] for s in stats.values())/sum(s['records'] for s in stats.values())<=.15
    result={'status':'passed','statistics':stats,'policies':{'literal_count':20,'total_characters':sum(map(len,POLICIES.values())),'max_pairwise_8gram_jaccard':max(pairs),'mean_pairwise_8gram_jaccard':sum(p[0] for p in pairs)/len(pairs),'source_sha256':hashlib.sha256(source.read_bytes()).hexdigest()},'mutation_checks':dict(mutations),'family_intersection':0,'contract_errors':0}
    write_json(OUT/'final_audit.json',result)
    manifest=json.loads((OUT/'manifest.json').read_text());manifest['status']='ready';manifest['final_audit']=result;write_json(OUT/'manifest.json',manifest)
    print(json.dumps(result))

if __name__=='__main__':main()
