"""Controlled natural-language rewrite of v1, with round-trip fact checks.
No public data input, no external teacher calls, no GPU. Rules and utterance variants
are authored here; this is not unconstrained LLM generation or a full semantic audit.
"""
import argparse
import collections
import copy
import hashlib
import json
from pathlib import Path
import random
import re
import shutil
import sys
from build import validate, expected_from_history

# Every group expresses one policy invariant; order/paragraph formatting can vary.
RULES = [
 ["When searching, carry over the status and retired-item preference the user requested. Start with an empty cursor and do not replace either filter with a default.",
  "A search starts with cursor set to the empty string. Its status and include_retired values come from the request; preserve both on every later page.",
  "Use the user's stated status and choice about retired entries for all searches. The first cursor is empty; pagination changes the cursor, not these filters."],
 ["For an allocation, the search list is only a summary. Select an exact label match, then retrieve that item's details before deciding whether to allocate it.",
  "Do not allocate from a listing alone. After finding the exact requested label, fetch its detailed record even if the listing already includes an item id.",
  "A similar label is not a substitute. Allocation requires an exact match and a subsequent detail lookup; a search summary cannot satisfy the detail requirement."],
 ["If an allocation search has no exact match yet, follow next_cursor. Stop searching when the exact match is found or there are no more pages. An exhausted search with no match must end with an explanation and no allocation.",
  "Continue through search pages until the requested label is found. If next_cursor is empty and the label is still absent, explain that it was not found; never allocate a near match.",
  "An unsuccessful page is not the end of the search when next_cursor is present. For allocation, stop at the exact matching item; if all pages are exhausted without it, report that outcome instead of making a change."],
 ["Refresh a stale detail record before proceeding. From then on, use the refreshed revision and warehouse rather than the old values.",
  "Details marked stale cannot authorize an action: refresh them first. The replacement revision and warehouse supersede the earlier ones.",
  "If the detailed result is stale, request a refresh. Only its new revision and warehouse are valid for subsequent permission checks and allocation."],
 ["Before any allocation, compare available_units with the requested quantity. Insufficient stock ends the request without an allocation or permission check.",
  "After obtaining fresh details, stop and explain if available_units is below the user's quantity. There is no reason to ask for permission when stock is insufficient.",
  "Fresh details must show enough units for the entire request. If stock falls short, make no allocation and do not proceed to a permission check."],
 ["With enough stock, check permission only when permit_required is true. Use the current revision and the requesting worker. A false permit_required means this check is unnecessary.",
  "The fresh record decides whether permission is needed. If permit_required is true and stock is adequate, check using this item's current revision and the worker from the request; otherwise skip that check.",
  "Once sufficient stock is established, obey permit_required: true requires a permission lookup for the current revision and worker, while false allows proceeding without a permission lookup."],
 ["Allocate only after all prerequisites succeed. A permission denial must be explained without allocating. Use the user's exact quantity, worker identifier and reason, together with the current item id, revision and warehouse.",
  "A denied permission ends the operation without a change. Otherwise, once the required checks pass, commit with the latest item details and the quantity, worker and reason supplied by the user.",
  "Never commit when a required permission is denied. For a permitted allocation, combine the current item id/revision/warehouse with the user's quantity/worker/reason; do not replace the reason with your own summary."],
 ["A browsing request is read-only: retrieve every page using unchanged filters, then summarize. Do not fetch item details or allocate anything for a browse request.",
  "For browsing, consume the full paginated listing and then respond. Keep the original filters throughout; browsing never calls for a detail lookup or a commit.",
  "When the user only wants to browse, follow every next_cursor, then provide a brief completion response. Browsing must not turn into an allocation or a detailed-record lookup."],
 ["For a report, resolve the label through the state tool first. Submit only when reportable is true, using the returned item id and the user's exact title, reason and priority. If it is not reportable, explain and stop.",
  "Reports require a state lookup, not an inventory search. If the item is reportable, submit the original title, reason and priority with the resolved item id; if not, do not submit.",
  "Before filing a report, check the label with the state lookup. A reportable result allows submission with the returned id and unchanged user wording for title/reason/priority. A non-reportable result requires a text explanation instead."],
 ["After a successful allocation or report, tell the user what the result confirms and stop. Archiving is a separate destructive operation, never a substitute for allocation, browsing or reporting.",
  "Successful commit and report results should be followed by a factual response, not another tool call. Do not use the archive tool for these requests.",
  "Finish with text once the requested write operation succeeds. Archive is not part of any allocation, browse or report workflow; do not use it to complete these tasks."]]

def policy(names, rng, style):
    statements=[rng.choice(group) for group in RULES]
    rng.shuffle(statements)
    intro=rng.choice(["You assist a workshop's operations desk. Apply the following service policy to each request.",
                      "You are an inventory and incident assistant. The rules below govern the available tools, even when a shortcut appears possible.",
                      "Help staff with workshop records and allocations. Follow the stated prerequisites and keep responses grounded in tool results."])
    bindings=(f"Available services: search pages with {names['scan']}; retrieve a detailed item record with {names['detail']}; "
              f"refresh stale details with {names['refresh']}; check permission with {names['permit']}; allocate with {names['commit']}. "
              f"Resolve reportability with {names['state']} and submit reports with {names['report']}. "
              f"The unrelated archive operation is {names['archive']}.")
    if style==0:
        body='\n'.join(f'{i+1}. {s}' for i,s in enumerate(statements))
    elif style==1:
        body='\n\n'.join(' '.join(statements[i:i+2]) for i in range(0,len(statements),2))
    else:
        body='\n'.join('- '+s for s in statements)
    return intro+'\n\n'+bindings+'\n\n'+body,statements


def user_parts(q,rng):
    mode=q['operation'];label=q['label']
    if mode=='allocate':
        start=rng.choice([f'Please allocate {q["units"]} units of "{label}".',f'I need {q["units"]} units allocated for "{label}".',f'Could you allocate {q["units"]} units of "{label}"?'])
    elif mode=='browse':
        start=rng.choice(['I only want to browse the inventory, without making changes.','Please browse the complete inventory listing for me.','Could you browse the inventory? This is a read-only request.'])
    else:
        start=rng.choice([f'Please file a report for "{label}".',f'I need to submit a report about "{label}".',f'Could you file a report concerning "{label}"?'])
    if mode=='report':
        details=[rng.choice([f'Use the title "{q["title"]}".',f'The title should be "{q["title"]}".']),
                 rng.choice([f'Set the priority to {q["priority"]}.',f'This needs {q["priority"]} priority.']),
                 rng.choice([f'The reason is "{q["reason"]}".',f'Please record "{q["reason"]}" as the reason.'])]
    else:
        details=[rng.choice([f'Only consider entries whose status is {q["status"]}.',f'Limit the search to the {q["status"]} status.']),
                 rng.choice(['Include retired entries in the search.','Retired entries should be included.']) if q['include_retired'] else rng.choice(['Exclude retired entries from the search.','Retired entries should be excluded.'])]
        if mode=='allocate':
            details += [rng.choice([f'My worker identifier is "{q["worker_id"]}".',f'Use "{q["worker_id"]}" as my worker identifier.']),
                        rng.choice([f'The reason is "{q["reason"]}".',f'Please record "{q["reason"]}" as the reason.'])]
    rng.shuffle(details)
    return start,' '.join(details)


def recover(text):
    """Parse authored utterance variants without accessing original request/metadata."""
    def find(patterns):
        matches=[re.search(p,text,re.I) for p in patterns]
        values=[m.group(1) for m in matches if m]
        if len(values)!=1: raise ValueError(('missing/ambiguous field',patterns,text))
        return values[0]
    q={}
    if re.search(r'\bbrowse\b',text,re.I):q['operation']='browse'
    elif re.search(r'\breport\b',text,re.I):q['operation']='report'
    else:q['operation']='allocate'
    if q['operation']=='allocate':
        q['units']=int(find([r'allocate (\d+) units',r'need (\d+) units allocated']))
        q['label']=find([r'units of "([^"]+)"',r'units allocated for "([^"]+)"'])
        q['worker_id']=find([r'worker identifier is "([^"]+)"',r'Use "([^"]+)" as my worker identifier'])
    if q['operation']=='report':
        q['label']=find([r'report (?:for|about|concerning) "([^"]+)"'])
        q['title']=find([r'Use the title "([^"]+)"',r'title should be "([^"]+)"'])
        q['priority']=find([r'priority to (low|normal|urgent)',r'needs (low|normal|urgent) priority'])
    else:
        q['status']=find([r'status is (ready|held|retired)',r'the (ready|held|retired) status'])
        positive=bool(re.search(r'Include retired entries|Retired entries should be included',text))
        negative=bool(re.search(r'Exclude retired entries|Retired entries should be excluded',text))
        assert positive != negative
        q['include_retired']=positive
    if q['operation']!='browse':q['reason']=find([r'reason is "([^"]+)"',r'record "([^"]+)" as the reason'])
    return q


def rewrite(row,index,split):
    old=row['messages'];spec=json.loads(old[0]['content'].split('CONTRACT\n')[1]);q=json.loads(old[1]['content'].split('REQUEST\n')[1])
    rng=random.Random(921700+index+(100000 if split=='stress' else 0))
    style=index%3
    pol,clauses=policy(spec['tools'],rng,style)
    start,details=user_parts(q,rng)
    multi=index%4==0
    msg=[{'role':'system','content':pol},{'role':'user','content':start if multi else start+' '+details}]
    skip=[]
    if multi:
        msg += [{'role':'assistant','content':'Before I proceed, could you confirm the remaining request details?'},{'role':'user','content':details}]
        skip.append(2)
    shift=len(msg)-2
    msg += copy.deepcopy(old[2:])
    skip += [i+shift for i in row['no_target']]
    result={**row,'id':row['id']+'~nl2','source':'codex_p1_stateful_natural_v2','messages':msg,'no_target':skip}
    decoded=recover(' '.join(m['content'] for m in msg if m['role']=='user'))
    assert all(q[k]==v for k,v in decoded.items())
    required={'allocate':{'operation','units','label','worker_id','reason','status','include_retired'},'browse':{'operation','status','include_retired'},'report':{'operation','label','title','reason','priority'}}[q['operation']]
    assert set(decoded)==required
    assert all(pol.count(c)==1 for c in clauses) and len(clauses)==10
    assert all(any(c in g for c in clauses) for g in RULES)
    assert 'CONTRACT' not in pol and 'REQUEST' not in msg[1]['content']
    # Test-only round trip: feed recovered facts into v1's decision interpreter.
    # Original JSON is never retained in the rewritten model input.
    checker=copy.deepcopy(row)
    # The v1 interpreter eagerly reads label even for browse; browse never selects
    # an item. Supply an irrelevant empty label only inside this test adapter.
    checker['messages'][1]['content']='REQUEST\n'+json.dumps({'label':'',**decoded})
    validate(checker)
    assert msg[2+shift:]==old[2:] and result['tools']==row['tools']
    return result,{'multiturn':int(multi),'policy_style':style},decoded


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--source',required=True);ap.add_argument('--out',required=True);a=ap.parse_args()
    src=Path(a.source);out=Path(a.out);out.mkdir(parents=True,exist_ok=False)
    report={'method':'AI-authored controlled natural-language renderer; no external teacher inference','splits':{},'limitations':['same three source workflows; this rewrite does not create new independent reasoning families','policy equivalence relies on audited authored variants, not a general natural-language theorem prover','stress uses the same renderer and is not independent validation; challenge is preserved unchanged']}
    samples=[];excluded=[]
    for split in ['train','stress']:
        counts=collections.Counter();source=src/f'{split}.jsonl'
        with (out/f'{split}.jsonl').open('x',encoding='utf-8') as f:
            for i,line in enumerate(source.open(encoding='utf-8')):
                row=json.loads(line)
                original_request=json.loads(row['messages'][1]['content'].split('REQUEST\n')[1])
                if original_request['operation']!='report' and original_request['status']=='retired' and not original_request['include_retired']:
                    excluded.append({'id':row['id'],'split':split,'reason':'requests only retired entries while excluding retired entries; original simulated results were not guaranteed empty'})
                    counts['excluded_conflicting_filters']+=1
                    continue
                new,stats,facts=rewrite(row,i,split)
                counts['records']+=1;counts['multiturn']+=stats['multiturn'];counts[f'policy_style_{stats["policy_style"]}']+=1
                f.write(json.dumps(new,ensure_ascii=False)+'\n')
                if i in [0,1,100,101,200,201]:samples.append({'split':split,'id':new['id'],'system':new['messages'][0]['content'],'conversation':new['messages'][1:5],'recovered_facts':facts})
        report['splits'][split]={**dict(counts),'source_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),'output_sha256':hashlib.sha256((out/f'{split}.jsonl').read_bytes()).hexdigest()}
    shutil.copyfile(src/'challenge.jsonl',out/'challenge.jsonl')
    assert (src/'challenge.jsonl').read_bytes()==(out/'challenge.jsonl').read_bytes()
    report['challenge_unchanged']=True
    report['excluded_records']=len(excluded)
    (out/'excluded_records.json').write_text(json.dumps(excluded,indent=2),encoding='utf-8')
    (out/'rewrite_qa.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    (out/'review_samples.json').write_text(json.dumps(samples,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(report))

if __name__=='__main__':main()
