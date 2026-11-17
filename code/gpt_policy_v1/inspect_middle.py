import gzip,json
seen=set();p=[]
for line in gzip.open('hw3/data/public_test.jsonl.gz','rt',encoding='utf-8'):
    r=json.loads(line);s=next((m.get('content','') for m in r['messages'] if m['role']=='system'),'')
    if len(s)>=2000 and s not in seen:seen.add(s);p.append(s)
for i in range(5,13):print('POLICY',i,p[i])
