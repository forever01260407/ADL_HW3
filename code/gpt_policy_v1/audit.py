"""Independent visible-history contract checks; no changes to raw data or staff checkers."""
import collections,json,re
from pathlib import Path
from generate import OUT,FLOWS,World,FORBIDDEN_NUMBERS,FORBIDDEN_KEYS,FORBIDDEN_TOOLS,validate_record,write_json

def numeric_values(v):
    if isinstance(v,dict):
        for x in v.values():yield from numeric_values(x)
    elif isinstance(v,list):
        for x in v:yield from numeric_values(x)
    elif isinstance(v,(float,int)) and not isinstance(v,bool):yield v
    elif isinstance(v,str):
        try:
            x=json.loads(v)
            if isinstance(x,(dict,list)):yield from numeric_values(x)
        except ValueError:pass

def verify(row):
    di=int(row['id'].split('_')[1]);flow=row['family'].split('_',2)[2]
    w=World('',flow,di,0);names={w.name(op):op for op in ['authenticate','receipt','handoff','owned_register','read_profile','adjust_profile','capacity_sheet','price_sheet','lock_capacity','primary_route','alternative_route','route_test','apply_route','open_register','read_lifecycle','window_check','revoke_operation','authority_map','read_case','ratify_case','submit_review','current_mount','fit_check','release_mount','install_mount','dossier_sheet','missing_evidence','attach_evidence','validate_evidence','dispatch_dossier','source_sheet','destination_sheet','coupling_check','move_payload','budget_sheet','commitment_sheet','remaining_allowance','consume_allowance','job_sheet','attempt_job','replay_status','replay_job']}
    p=w.prefix+'_';seen={};ops=[];result=None
    for index,m in enumerate(row['messages']):
        if m['role']=='tool':
            data={k.removeprefix(p):v for k,v in json.loads(m['content']).items()};seen[ops[-1]]=data;result=data;continue
        if not m.get('tool_calls'):continue
        call=m['tool_calls'][0]['function'];op=names[call['name']];a={k.removeprefix(p):v for k,v in call['arguments'].items()}
        if op!='authenticate':assert seen['authenticate']['access']=='gpclear'
        if op=='read_profile':assert a['record']==seen['owned_register']['chosen']
        if op=='adjust_profile':assert seen['read_profile']['gate']=='gpclear'
        if op=='alternative_route':assert seen['primary_route']['primary']=='gpunusable'
        if op=='apply_route':assert seen['route_test']['gate']=='gpclear' and a['path']==seen['route_test']['path']
        if op=='revoke_operation':assert seen['window_check']['gate']=='gpclear' and seen['read_lifecycle']['stage']=='gpprovisional'
        if op in ['ratify_case','submit_review']:
            assert seen['read_case']['gate']=='gpclear';assert seen['authority_map']['role']==('gpdecider' if op=='ratify_case' else 'gpreviewer')
        if op=='read_case':assert a['scope']==seen['authority_map']['scope']
        if op=='release_mount':assert seen['fit_check']['gate']=='gpclear' and a['old']==seen['fit_check']['old']
        if op=='install_mount':assert a['ack']==seen['release_mount']['ack']=='gpreleased' and a['new']==seen['release_mount']['new']
        if op=='attach_evidence':assert seen['missing_evidence']['category']==a['category']
        if op=='validate_evidence':assert a['file']==seen['attach_evidence']['file']
        if op=='dispatch_dossier':assert seen['validate_evidence']['gate']=='gpclear'
        if op=='coupling_check':assert a['seal']==seen['source_sheet']['seal'] and a['socket']==seen['destination_sheet']['socket']
        if op=='move_payload':assert seen['coupling_check']['gate']=='gpclear'
        if op=='remaining_allowance':assert a['ceiling']==seen['budget_sheet']['ceiling'] and a['occupied']==seen['commitment_sheet']['occupied']
        if op=='consume_allowance':
            assert seen['remaining_allowance']['gate']=='gpclear';assert a['remaining']==seen['budget_sheet']['ceiling']-seen['commitment_sheet']['occupied'];assert a['amount']<=a['remaining']
        if op=='replay_status':assert seen['attempt_job']['error']=='gpretry-inspect'
        if op=='replay_job':assert seen['replay_status']['gate']=='gpclear' and a['seal']==seen['replay_status']['seal']
        if op=='lock_capacity':
            assert a['amount']<=seen['capacity_sheet']['capacity'] and a['offer']==seen['price_sheet']['offer']
            assert any(x['role']=='user' and 'gpaccept' in x['content'] for x in row['messages'][:index]);assert a['ack']=='gpaccept'
        ops.append(op)
    assert result and result.get('note')==row['messages'][-1]['content']
    validate_record(row)

def main():
    report={};collisions=[];errors=[]
    for split in ['train','dev']:
        c=collections.Counter()
        for line in (OUT/(split+'_raw.jsonl')).open():
            row=json.loads(line)
            try:verify(row)
            except (AssertionError,KeyError,ValueError) as e:errors.append({'id':row['id'],'error':str(e)})
            # no_target holds required message indices, not copied task quantities.
            hits=set(numeric_values({'messages':row['messages'],'tools':row['tools']}))&FORBIDDEN_NUMBERS
            if hits:collisions.append({'id':row['id'],'family':row['family'],'values':sorted(hits)})
            c['records']+=1
        report[split]=dict(c)
    report['semantic_contract_errors']=errors;report['public_numeric_coincidences']=collisions
    write_json(OUT/'semantic_audit.json',report);print(json.dumps({'semantic_contract_errors':len(errors),'numeric_coincidences':len(collisions),'examples':collisions[:5]}))

if __name__=='__main__':main()
