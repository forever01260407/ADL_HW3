"""Read-only inspection; prints policies for abstraction, never writes public content."""
import collections,gzip,json,shutil
from pathlib import Path
rows=[json.loads(x) for x in gzip.open('hw3/data/public_test.jsonl.gz','rt',encoding='utf-8')]
seen=set();selected=[]
for r in rows:
    s=next((m.get('content','') for m in r['messages'] if m['role']=='system'),'')
    if len(s)>=2000 and s not in seen:
        seen.add(s);selected.append(s)
print('unique long policies',len(selected),'disk',shutil.disk_usage('data_work'))
for i in range(0,len(selected),max(1,len(selected)//16)):
    print('\nPOLICY',i,'\n',selected[i][:4000])
