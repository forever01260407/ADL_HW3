"""Separately authored tiny workflow challenge; never mix into training.
Different operations/graphs/input style from build.py. Small diagnostic, not a reliable
estimate of private score. Each contrast pair must stay together in evaluation.
"""
import argparse
import json
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'data'))
from common import make_tool,call_message,tool_message,first_problem,wrong_type


def tool(name, description, fields):
    return make_tool({'name':name,'description':description,'parameters':{'type':'object','properties':{k:{'type':v} for k,v in fields.items()},'required':list(fields),'additionalProperties':False}})


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--out',required=True);a=ap.parse_args()
    rows=[]
    def add(key,policy,user,tools,prior,result,target,explanation):
        history=[{'role':'system','content':policy},{'role':'user','content':user},call_message([prior]),tool_message(prior[0],result)]
        history.append(call_message([target]) if isinstance(target,tuple) else {'role':'assistant','content':target})
        row={'id':'cx29_challenge_'+key,'family':'cx29_challenge_'+key.rsplit('_',1)[0],'task':'part1','source':'codex_independent_challenge','tools':tools,'messages':history,'no_target':[2],'audit_rationale':explanation}
        assert first_problem(tools,history) is None
        funcs={t['function']['name']:t['function'] for t in tools}
        for m in history:
            for c in m.get('tool_calls',[]): assert wrong_type(funcs[c['function']['name']],c['function']['arguments']) is None
        rows.append(row)
    # Join two independent flags; summary data cannot authorize deleting records.
    ts=[tool('inspect_bundle','Read reconciliation flags.',{'bundle':'string'}),tool('request_signoff','Ask the named reviewer to authorize archival.',{'bundle':'string','reviewer':'string'}),tool('archive_bundle','Archive an already signed-off reconciled bundle.',{'bundle':'string'}),tool('reconcile_bundle','Repair pending reconciliation.',{'bundle':'string'})]
    policy='For archival, pending reconciliation must be repaired first. Once reconciled, request signoff if absent. Archive only when both reconciled and signed_off are true. Never treat the presence of a reviewer name as signoff.'
    for i,(reconciled,signed,name) in enumerate([(False,True,'reconcile_bundle'),(True,False,'request_signoff'),(True,True,'archive_bundle')]):
        args={'bundle':'pack-76'}|({'reviewer':'Mira'} if name=='request_signoff' else {})
        add(f'join_{i}',policy,'Archive pack-76 when its prerequisites are complete.',ts,('inspect_bundle',{'bundle':'pack-76'}),{'reconciled':reconciled,'signed_off':signed,'reviewer':'Mira'},(name,args),'Choose the first unmet prerequisite, not the final requested action.')
    # Aggregate filtering; empty vs nonempty output changes whether an action is useful.
    ts=[tool('read_batches','List processing batches.',{}),tool('queue_batches','Queue all eligible ids in ascending id order.',{'ids':'array'}),tool('queue_everything','Queue all batches regardless of state.',{})]
    policy='Queue only batches with enabled=true AND remaining>0. Send all matching ids in one queue_batches call, sorted ascending. If none qualify, explain that there is no eligible batch. Never use queue_everything.'
    for i,items in enumerate([[{'id':'b3','enabled':True,'remaining':2},{'id':'b1','enabled':True,'remaining':4},{'id':'b2','enabled':False,'remaining':8}],[{'id':'b3','enabled':True,'remaining':0},{'id':'b1','enabled':False,'remaining':4}], [{'id':'b2','enabled':True,'remaining':1}]]):
        targets=[('queue_batches',{'ids':['b1','b3']}),'There are no eligible batches.',('queue_batches',{'ids':['b2']})]
        add(f'aggregate_{i}',policy,'Queue the eligible processing batches.',ts,('read_batches',{}),{'batches':items},targets[i],'Conjunction filter and array ordering; do not select a single arbitrary row.')
    # Retry protocol differs from allocation; exhausted retries require escalation.
    ts=[tool('send_packet','Send payload with a one-use token.',{'payload':'string','token':'string'}),tool('renew_token','Get a new token for a failed transmission.',{'packet_id':'string'}),tool('escalate_delivery','Escalate an exhausted delivery attempt.',{'packet_id':'string','reason':'string'})]
    policy='After send_packet: delivered means report success and stop. token_expired with retries_left>0 requires renew_token before another send. If retries_left=0, escalate_delivery with reason retry budget exhausted. Do not claim a failed packet was delivered.'
    for i,(status,retries,target) in enumerate([('delivered',1,'The packet was delivered.'),('token_expired',1,('renew_token',{'packet_id':'packet-93'})),('token_expired',0,('escalate_delivery',{'packet_id':'packet-93','reason':'retry budget exhausted'}))]):
        add(f'retry_{i}',policy,'Deliver the payload calibration-ready using token token-A.',ts,('send_packet',{'payload':'calibration-ready','token':'token-A'}),{'packet_id':'packet-93','status':status,'retries_left':retries},target,'Branch on status and remaining retry budget; success stops the chain.')
    # Schema boolean filters and retaining a requested cursor without inventing dates.
    ts=[tool('find_entries','Search entries with the requested filter.',{'state':'string','include_hidden':'boolean','cursor':'string'}),tool('export_entries','Export exactly the ids found.',{'ids':'array'})]
    policy='For export, finish every search page before exporting. Reuse the user-requested state and include_hidden filters unchanged on later pages. Empty next_cursor marks the last page. On a nonempty last page export ids; if the entire first page is empty and has no next cursor, report no results.'
    for i,(result,target) in enumerate([({'ids':['e1'],'next_cursor':'p2'},('find_entries',{'state':'paused','include_hidden':False,'cursor':'p2'})),({'ids':['e1'],'next_cursor':''},('export_entries',{'ids':['e1']})),({'ids':[],'next_cursor':''},'There are no matching entries to export.')]):
        add(f'filters_{i}',policy,'Export paused entries, excluding hidden ones.',ts,('find_entries',{'state':'paused','include_hidden':False,'cursor':''}),result,target,'Pagination retains explicit filter; final page and empty result terminate differently.')
    with Path(a.out).open('x',encoding='utf-8') as f:
        for r in rows:f.write(json.dumps(r,ensure_ascii=False)+'\n')
    print(json.dumps({'records':len(rows),'families':len({r['family'] for r in rows}),'target_calls':sum(bool(r['messages'][-1].get('tool_calls')) for r in rows),'target_text':sum(not r['messages'][-1].get('tool_calls') for r in rows)}))


if __name__=='__main__':main()
