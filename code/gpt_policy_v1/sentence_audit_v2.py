"""Supplemental sentence/name overlap check, read-only public input."""
import collections,json,re
from generate import public_rows,FORBIDDEN_TOOLS,FORBIDDEN_KEYS,write_json
from build_v2 import OUT,STATUS

def sentences(text):
    return [' '.join(s.lower().split()) for s in re.split(r'(?<=[.!?])\s+|\n+',text) if len(s.strip())>=32]

def main():
    public=set()
    for r in public_rows:
        for m in r['messages']:
            if m['role']!='tool':public.update(sentences(m.get('content') or ''))
    hits=[];names=[];markers=[];counts=collections.Counter()
    for split in ['train','dev']:
        for line in (OUT/f'{split}.jsonl').open():
            r=json.loads(line);counts[split]+=1
            for t in r['tools']:
                f=t['function']
                if f['name'] in FORBIDDEN_TOOLS or set(f['parameters']['properties'])&FORBIDDEN_KEYS:names.append(r['id'])
            for m in r['messages']:
                text=m.get('content') or ''
                if m['role']!='tool':
                    overlap=set(sentences(text))&public
                    if overlap:hits.append({'id':r['id'],'matching_sentence_count':len(overlap)})
                if any(k in text for k in STATUS):markers.append(r['id'])
    result={'records':dict(counts),'public_exact_sentences_at_least_32_chars':hits,'public_tool_or_business_parameter_name_overlap':names,'artificial_status_markers':markers,'limits':'Exact normalized sentence matching is supplementary; it does not prove absence of all semantic paraphrases. Standard JSON schema keys and required message indices are not business names or copied task values.'}
    write_json(OUT/'sentence_name_audit.json',result);assert not hits and not names and not markers
    print(json.dumps(result))

if __name__=='__main__':main()
