"""Build portable metadata from all three full-edge batches (800 unique cases)."""
import argparse
from collections import Counter
import csv
import json
from pathlib import Path
import shutil
import numpy as np

ROOT=Path(__file__).resolve().parents[1]


def main():
    p=argparse.ArgumentParser();p.add_argument('--data-root',type=Path,required=True,help='Directory containing the three full-edge source batches');a=p.parse_args()
    batches=[('hole_plate_2d_damage_full_edge_100',100),('hole_plate_2d_damage_full_edge_next_200',200),('hole_plate_2d_damage_full_edge_next_500',500)]
    rows=[];results={};summaries=[];audits=[]
    for batch,n in batches:
        source=a.data_root/batch
        with (source/f'cases_{n}.csv').open(newline='',encoding='utf-8-sig') as f: br=list(csv.DictReader(f))
        with (source/'results.csv').open(newline='',encoding='utf-8-sig') as f: rr=list(csv.DictReader(f))
        assert len(br)==len(rr)==n
        for r in rr:
            assert r['case_id'] not in results, f'Duplicate case: {r["case_id"]}'
            results[r['case_id']]=r
        for r in br:
            r['source_batch']=batch
            cid=r['case_id']
            assert (source/cid/(cid+'_damage_gnn.npz')).is_file()
            summaries.append(json.loads((source/cid/(cid+'_damage_summary.json')).read_text()))
        rows.extend(br)
        audits.append(json.loads((source/f'campaign_audit_{n}.json').read_text()))
    assert len(rows)==len(results)==800 and len({r['source_case_id'] for r in rows})==800
    assert all(r['analysis_status']=='COMPLETED' for r in results.values())
    assert all(s['frame_count']==101 and s['nominal_strain']==.015 and s['load_application']=='FULL_RIGHT_EDGE_UNIFORM_U1' for s in summaries)
    for k in ['connectivity_sha256','plastic_table','ductile_damage_table','damage_failure_displacement_mm']:
        assert len({json.dumps(s[k],sort_keys=True) for s in summaries})==1,k
    a.source=a.data_root/batches[-1][0]
    out=ROOT/'data';out.mkdir(exist_ok=True);(ROOT/'docs').mkdir(exist_ok=True)
    cols=['case_id','source_case_id','source_batch','split','L_mm','W_mm','t_mm','R_mm','R_over_W','nominal_strain','u_max_mm']
    with (out/'cases.csv').open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=cols+['analysis_status']);w.writeheader()
        for r in rows:w.writerow({**{k:r[k] for k in cols},'analysis_status':results[r['case_id']]['analysis_status']})
    for batch,n in batches:shutil.copyfile(a.data_root/batch/f'campaign_audit_{n}.json',out/f'campaign_audit_{n}.json')
    audit={'case_count':len(rows),'unique_source_case_count':len({r['source_case_id'] for r in rows}),'completed_case_count':len(results),'damaged_case_count':sum(s['max_damage']>0 for s in summaries),'deleted_case_count':sum(s['deleted_element_count']>0 for s in summaries),
           'deleted_element_count_range':[min(s['deleted_element_count'] for s in summaries),max(s['deleted_element_count'] for s in summaries)],
           'peak_tensile_force_n_range':[min(s['peak_tensile_force_n'] for s in summaries),max(s['peak_tensile_force_n'] for s in summaries)],
           'split_counts':dict(Counter(r['split'] for r in rows)),'source_batches':dict(batches),'status':'PASS'}
    for key in ['max_allsd_allie_ratio','max_allsd_allie_ratio_after_10pct','max_final_allsd_allie_ratio']:
        audit[key]=max(b[key] for b in audits)
    (out/'campaign_audit_800.json').write_text(json.dumps(audit,indent=2),encoding='utf-8')
    card={'dataset_id':'HolePlate2D-Damage-800','presentation_version':'0.2.0-preview','source_batches':[b for b,n in batches],'case_count':len(rows),'split_counts':dict(Counter(r['split'] for r in rows)),
          'source_format':'NPZ','units':{'length':'mm','force':'N','stress':'MPa','energy':'N mm'},'frames_per_case':101,'nodes_per_case':1684,'elements_per_case':1580,'element_type':'CPS4R',
          'geometry_ranges':{k:[min(float(r[k]) for r in rows),max(float(r[k]) for r in rows)] for k in ['L_mm','W_mm','t_mm','R_mm','R_over_W']},
          'loading':'FULL_RIGHT_EDGE_UNIFORM_U1','final_nominal_strain':0.015,'material_calibration':'synthetic; not calibrated to a specific alloy',
          'license':None,'public_data_url':None,'model_results_included':False,'evaluation_protocol':'suggested metrics only; not implemented in this visualization package',
          'style_reference':'https://github.com/qilinli/StructBench','official_structbench_benchmark':False}
    (out/'card.json').write_text(json.dumps(card,indent=2),encoding='utf-8')
    cid='HP2D_0471_DMG_FULL';summary=json.loads((a.source/cid/(cid+'_damage_summary.json')).read_text())
    lines=['# Extracted data schema','', '[Back to dataset card](../README.md)','',
           'Source: Abaqus extraction for `HolePlate2D-Damage-800`, combining 100 + 200 + 500 non-overlapping full-edge cases. One `<case_id>_damage_gnn.npz` plus a JSON summary per case. Shapes below are read from the representative case. `T=101`, `N=1684`, `E=1580`.','',
           '| NPZ key | Shape | Stored dtype |','|---|---|---|']
    with np.load(a.source/cid/(cid+'_damage_gnn.npz'),allow_pickle=False) as d:
        for k in d.files:lines.append(f'| `{k}` | `{d[k].shape}` | `{d[k].dtype}` |')
    lines += ['', '## Indexing and units','',
      '- `connectivity` contains Abaqus node labels, not zero-based array positions. Map through `node_labels` before indexing coordinates. Element histories follow `element_labels`.',
      '- Coordinates, prescribed displacement and nodal U1/U2 use mm; stress and von Mises use MPa; reaction uses N; energy histories use N·mm.',
      '- PEEQ, SDEG and STATUS are dimensionless. STATUS=1 is active; STATUS=0 is deleted.',
      '- The stress array preserves the four-component order returned by the Abaqus plane-stress S field. Component labels are not saved in NPZ; confirm their order from the source ODB before using individual tensor components. The saved von Mises scalar is exported directly from `value.mises`.',
      '- `normalized_progress` holds the saved step-frame values over a unit-duration monotonic ramp. This is a quasi-static loading coordinate, not animation playback time.',
      '- `geometry_json` is a JSON string containing geometry and generation metadata.',
      '', '## Extraction conventions','',
      'The extractor initializes missing stress values to zero, carries the last PEEQ value forward when unavailable, and defaults missing SDEG values to 1 for deleted elements or 0 for active elements when the SDEG field exists. Use STATUS explicitly; zeros in stress after deletion are not measurements of an intact material. History outputs (force, prescribed displacement and energy) are interpolated to the saved field-frame times.',
      '', '## Elasticity and hardening','',
      'E = 210000 MPa; Poisson ratio = 0.30. Isotropic plastic hardening:', '',
      '| Yield stress [MPa] | Equivalent plastic strain |','|---:|---:|']
    lines += [f'| {s:g} | {e:g} |' for s,e in summary['plastic_table']]
    lines += ['', '## Ductile damage initiation','', '| Equivalent plastic strain at initiation | Stress triaxiality | Strain rate |','|---:|---:|---:|']
    lines += [f'| {s:g} | {t:g} | {r:g} |' for s,t,r in summary['ductile_damage_table']]
    lines += ['', 'Linear displacement-based damage evolution; failure displacement parameter 0.25 mm. Synthetic baseline, not alloy-specific calibration.',
      '', '## Analysis','', 'Abaqus/Standard Static, General with `NLGEOM=YES`. Initial stabilization fraction 0.0002. The source summaries retain solver warnings and stabilization-energy diagnostics. Consult these diagnostics together with the batch audit when selecting cases for training.',
      '', '## Safe loading example','', '```python','import numpy as np','with np.load("<case_id>_damage_gnn.npz", allow_pickle=False) as data:',
      '    xyz = data["node_coordinates"]','    label_to_row = {int(label): i for i, label in enumerate(data["node_labels"])}',
      '    conn = np.array([[label_to_row[int(n)] for n in e] for e in data["connectivity"]])',
      '    displacement = data["nodal_displacement_history"]',
      '    magnitude_mm = np.linalg.norm(displacement, axis=-1)',
      '    active = data["element_status_history"] > 0.5','```','']
    (ROOT/'docs/DATA_SCHEMA.md').write_text('\n'.join(lines),encoding='utf-8')
    print(json.dumps({'case_count':len(rows),'ranges':card['geometry_ranges'],'audit':audit},indent=2))


if __name__=='__main__':main()
