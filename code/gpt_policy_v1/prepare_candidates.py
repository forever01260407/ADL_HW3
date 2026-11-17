"""Refine user utterances and add near-purpose read-only distractors; raw remains intact."""
import copy,json,random
from generate import OUT,FORBIDDEN_KEYS,FORBIDDEN_TOOLS,write_json
from audit import verify

INTENT={
 'identity_assets':['Please adjust the working profile for {asset} in the {domain} setup.','I would like the current profile of {asset} adjusted for our {domain} work.','Can you make the permitted profile adjustment on {asset} for {domain}?'],
 'capacity_quote':['Please reserve {amount} calibration units against {asset} for {domain}.','I need {amount} units of capacity on {asset} for our {domain} work. Can you arrange that?','Could you secure {amount} capacity units on {asset} for {domain}?'],
 'fallback':['Please find a usable treatment route for {asset} in the {domain} setup and apply it.','Can you carry out an allowed route for {asset}, using a secondary method if the primary is unavailable? This concerns {domain}.','I need {asset} processed for {domain}; use the primary route if permitted, otherwise investigate a usable alternative.'],
 'lifecycle':['Please withdraw the operation associated with {asset} in our {domain} setup if its current lifecycle permits this.','I want to revoke my operation {asset} for {domain}; please check its current state and cancellation window.','Could you cancel the reversible operation on {asset} for {domain}, subject to the local rules?'],
 'role_scope':['Please process the decision case {asset} for {domain} using the authority my operator record actually grants.','I need case {asset} in the {domain} setup handled according to my verified decision scope.','Can you take the authorized review or final-decision action on {asset} for {domain}?'],
 'replace_sequence':['I want the current fitting on {asset} replaced with the destination fitting for {domain}; I will provide its reference next.','Please arrange a compatible replacement for the existing mount on {asset} in the {domain} setup.','Could you switch the fitting associated with {asset} for {domain} after checking the old and proposed pair?'],
 'document_stages':['Please prepare and submit the evidence dossier associated with {asset} for {domain}.','I would like dossier {asset} for {domain} submitted after its required evidence has been supplied and validated.','Can you handle the missing evidence and submission for dossier {asset} in our {domain} setup?'],
 'cross_record':['Please move the payload from {asset} to the destination connector I will supply for {domain}.','I need a compatible source-to-destination transfer from {asset} for {domain}; I will give the destination reference next.','Could you transfer the payload associated with {asset} for {domain}, checking the destination independently first?'],
 'quota_compute':['Please consume {amount} allowance units on {asset} for our {domain} work if the uncommitted balance permits it.','I need {amount} units charged against the remaining allowance of {asset} for {domain}.','Could you allocate {amount} allowance units from {asset} for {domain}, taking its existing commitments into account?'],
 'recover_error':['Please finish the pending operation on {asset} for {domain}, using only the allowed recovery procedure if it fails.','I need the queued job {asset} completed for {domain}; please inspect any failure before retrying.','Could you execute the pending job associated with {asset} for {domain} and verify its outcome?']}
WRITES={'identity_assets':'adjust_profile','capacity_quote':'lock_capacity','fallback':'apply_route','lifecycle':'revoke_operation','role_scope':'ratify_case','replace_sequence':'install_mount','document_stages':'dispatch_dossier','cross_record':'move_payload','quota_compute':'consume_allowance','recover_error':'replay_job'}

def main():
    audit=json.loads((OUT/'semantic_audit.json').read_text());excluded={x['id'] for x in audit['public_numeric_coincidences']}|{x['id'] for x in audit['semantic_contract_errors']}
    manifest=json.loads((OUT/'manifest.json').read_text());domains={x['index']:x['domain'] for x in manifest['domains']};summary={}
    for split in ['train','dev']:
        count=0
        with (OUT/f'{split}_candidate.jsonl').open('x',encoding='utf-8') as f:
            for line in (OUT/f'{split}_raw.jsonl').open():
                row=json.loads(line)
                if row['id'] in excluded:continue
                di=int(row['id'].split('_')[1]);n=int(row['id'].rsplit('_',1)[1]);flow=row['family'].split('_',2)[2]
                w=json.loads((OUT/f'worlds_{di:02d}.json').read_text())[n]['world'];domain=domains[di]
                text=INTENT[flow][n%3].format(domain=domain,**w)
                row['messages'][1]['content']=text+f' My stated purpose is "{w["reason"]}".'
                target=next(t for t in row['tools'] if WRITES[flow].replace('_','').lower() in t['function']['name'].replace('_','').lower())
                for prefix,description in [('preview','Show a proposed outcome only; this tool does not execute or authorize the operation.'),('history','Read a historical execution; historical information cannot replace a current eligibility check.'),('simulate','Run an advisory dry run without committing changes; this is not execution or current permission.')]:
                    decoy=copy.deepcopy(target);fn=decoy['function'];fn['name']=prefix+'_'+fn['name'];fn['description']=description
                    assert fn['name'] not in FORBIDDEN_TOOLS;assert not set(fn['parameters']['properties'])&FORBIDDEN_KEYS
                    row['tools'].append(decoy)
                random.Random(di*991+n).shuffle(row['tools']);verify(row)
                f.write(json.dumps(row,ensure_ascii=False)+'\n');count+=1
        summary[split]={'records':count}
    summary['excluded_ids']=sorted(excluded);summary['reason']='raw semantic violations or independently generated numeric values coincident with public numeric leaves'
    write_json(OUT/'candidate_preparation.json',summary);print(json.dumps(summary))

if __name__=='__main__':main()
