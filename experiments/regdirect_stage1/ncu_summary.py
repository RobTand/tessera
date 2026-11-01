"""Summarise an ncu_export.sh directory: details, stall ratios, memory counters, top stall SASS lines.."""
import csv,io,collections,sys
E=sys.argv[1]
rows=list(csv.DictReader(open(E+'/details.csv')))
by=collections.OrderedDict()
for r in rows: by.setdefault(r['ID'],{'name':r['Kernel Name'][:30]})[r['Metric Name']]=r['Metric Value']
want=['Duration','Registers Per Thread','Active Warps Per Scheduler','Issued Warp Per Scheduler','Executed Instructions','L1/TEX Cache Throughput','L2 Cache Throughput','Memory Throughput','Achieved Occupancy']
print('kernels',[v['name'][:22] for v in by.values()])
for w in want: print(w[:30].ljust(30),*[str(v.get(w,''))[:10].rjust(10) for v in by.values()])
rd=list(csv.reader(open(E+'/raw.csv'))); hdr=rd[0]; data=rd[2:]
for i,h in enumerate(hdr):
    if h.startswith('smsp__average_warps_issue_stalled_') and h.endswith('_per_issue_active.ratio'):
        v=[float(r[i] or 0) for r in data]
        if max(v)>0.4: print(h[34:-23][:28].ljust(30),*[f'{x:10.2f}' for x in v])
for key in ['l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld.sum','l1tex__data_pipe_lsu_wavefronts_mem_shared_op_ld.sum','l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum','l1tex__t_sector_hit_rate.pct','lts__t_sectors_srcunit_tex_op_read.sum','lts__t_sector_hit_rate.pct']:
    if key in hdr: i=hdr.index(key); print(key[:30].ljust(30),*[r[i][:10].rjust(10) for r in data])
L=open(E+'/source.csv').read().splitlines()
starts=[i for i,l in enumerate(L) if l.startswith('"Kernel Name"')]
starts.append(len(L))
for a,b in zip(starts,starts[1:]):
    name=L[a][15:60]
    rows=list(csv.reader(io.StringIO("\n".join(L[a+1:b])))); h=rows[0]; d=[r for r in rows[1:] if r and r[0].startswith('0x')]
    if not d or ('rd_' not in name and 'regdirect' not in name): continue
    iS=h.index('Warp Stall Sampling (All Samples)'); tot=sum(float(r[iS] or 0) for r in d)
    cols=[c for c in h if c.startswith('stall_') and '(Not' not in c]
    agg={c:sum(float(r[h.index(c)] or 0) for r in d)/tot*100 for c in cols}
    print('\n',name,'samples',tot,{k[6:]:round(v,1) for k,v in sorted(agg.items(),key=lambda x:-x[1])[:6]})
    pos={r[0]:i for i,r in enumerate(d)}
    for r in sorted(d,key=lambda r:-float(r[iS] or 0))[:8]:
        top=max(cols,key=lambda c:float(r[h.index(c)] or 0))
        print(f"  {float(r[iS])/tot*100:5.1f}% {top[6:]:10s} #{pos[r[0]]:5d} {r[1].strip()[:70]}")
