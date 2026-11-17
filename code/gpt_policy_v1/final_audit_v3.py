"""V3 checks, literal policy proof, public sentence/name exclusion and dataset manifest."""
import ast,collections,copy,hashlib,json,re
from pathlib import Path
from policies_v3 import POLICIES,DOMAINS
from policies_v2 import POLICIES as V2_POLICIES
from build_v3 import OUT,verify_v3
from generate import write_json,public_rows

def grams(text):
    words=re.findall(r'\w+',text.lower());return {tuple(words[i:i+8]) for i in range(len(words)-7)}

def sentences(text):return {' '.join(s.lower().split()) for s in re.split(r'(?<=[.!?])\s+|\n+',text) if len(s.strip())>=32}

def main():
    source=Path(__file__).with_name('policies_v3.py');tree=ast.parse(source.read_text());node=next(x for x in tree.body if isinstance(x,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='POLICIES' for t in x.targets))
    assert isinstance(node.value,ast.Dict) and all(isinstance(x,ast.Constant) and isinstance(x.value,str) for x in node.value.values)
    assert len(POLICIES)==20 and all(1500<=len(p)<=3500 for p in POLICIES.values())
    public=set()
    for r in public_rows:
        for m in r['messages']:
            if m['role']!='tool':public |= sentences(m.get('content') or '')
    stats={};families={};mutation_count=0
    for split in ['train','dev']:
        rows=[json.loads(l) for l in (OUT/f'{split}.jsonl').open()];c=collections.Counter();ids=set();dom=set();fams=set();chars=[]
        for r in rows:
            verify_v3(r);assert r['source']=='gpt_policy_v3' and r['id'] not in ids;ids.add(r['id']);di=int(r['id'].split('_')[1]);dom.add(di);fams.add(r['family']);chars.append(len(r['messages'][0]['content']))
            for m in r['messages']:
                if m['role']!='tool':assert not sentences(m.get('content') or '')&public
            for i,m in enumerate(r['messages']):
                if m['role']=='assistant' and i not in r['no_target']:
                    c['targets']+=1;c['call_after_tool' if m.get('tool_calls') else 'text_after_tool']+=1
            c['records']+=1;c['refusals']+='referred for' in r['messages'][-1]['content']
            mutant=copy.deepcopy(r);m=next(m for i,m in enumerate(mutant['messages']) if m.get('tool_calls') and i not in mutant['no_target']);args=m['tool_calls'][0]['function']['arguments'];args[next(iter(args))]='not-a-returned-reference'
            try:verify_v3(mutant)
            except (AssertionError,ValueError,KeyError):mutation_count+=1
            else:raise AssertionError('mutation escaped checks')
        families[split]=fams
        stats[split]={**dict(c),'domains':len(dom),'families':len(fams),'policy_chars_min':min(chars),'policy_chars_max':max(chars),'sha256':hashlib.sha256((OUT/f'{split}.jsonl').read_bytes()).hexdigest()}
    assert not families['train']&families['dev'];assert stats['train']['records']>=300
    gp=[grams(p) for p in POLICIES.values()];within=[len(a&b)/len(a|b) for i,a in enumerate(gp) for b in gp[i+1:]];previous=[grams(p) for p in V2_POLICIES.values()];cross=[len(a&b)/len(a|b) for a in gp for b in previous]
    result={'status':'passed','statistics':stats,'policy_count':20,'policy_characters_total':sum(map(len,POLICIES.values())),'policy_source_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),'max_policy_pair_8gram_jaccard':max(within),'max_policy_vs_v2_8gram_jaccard':max(cross),'mutation_errors_detected':mutation_count,'public_exact_sentence_hits':0,'family_intersection':0,'contract_errors':0}
    write_json(OUT/'final_audit.json',result);manifest=json.loads((OUT/'manifest.json').read_text());manifest['status']='ready';manifest['final_audit']=result;write_json(OUT/'manifest.json',manifest);print(json.dumps(result))

if __name__=='__main__':main()
