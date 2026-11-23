#!/usr/bin/env node
/** External directed-network intervention feasibility and parity-gauge audit.
 *  The input is a real directed edge list (SNAP Gnutella08), NOT shipped in ZIP.
 *  No model is trained. Raw queries are drawn independently of their answers.
 *  Intentionally preserves all outcomes, including unchanged and all-zero cases.
 *  Deterministic RNG: 32-bit LCG (1664525, 1013904223), seed 34789.
 *  Example: node snap_external_audit.js p2p-Gnutella08.txt > snap_audit.json
 */
'use strict';
const fs=require('fs');const crypto=require('crypto');
const file=process.argv[2];if(!file){console.error('Usage: node snap_external_audit.js INPUT_EDGE_LIST');process.exit(2)}
const original=fs.readFileSync(file,'utf8');
const adjacency=new Map(),undirected=new Map();let edgeCount=0;
function ensure(M,k){if(!M.has(k))M.set(k,new Set());return M.get(k)}
for(const line of original.split(/\r?\n/)){
  if(line.startsWith('#'))continue;
  const a=line.trim().split(/\s+/).map(Number);
  if(a.length!==2||!a.every(Number.isInteger))continue;
  const [u,v]=a;ensure(adjacency,u).add(v);ensure(adjacency,v);
  ensure(undirected,u).add(v);ensure(undirected,v).add(u);edgeCount++;
}
const vertices=[...adjacency.keys()].sort((a,b)=>a-b);
let state=34789;
const rand=()=>((state=(Math.imul(state,1664525)+1013904223)>>>0)/4294967296);
const choice=a=>a[Math.floor(rand()*a.length)];
function shuffle(a){for(let i=a.length-1;i>0;i--){let j=Math.floor(rand()*(i+1));[a[i],a[j]]=[a[j],a[i]]}return a}
function bfs(G,s,t){if(s===t)return true;const seen=new Set([s]),q=[s];for(let i=0;i<q.length;i++)for(const v of (G.get(q[i])||[]))if(!seen.has(v)){if(v===t)return true;seen.add(v);q.push(v)}return false}
function sample(size){for(let k=0;k<30;k++){const start=choice(vertices),selected=new Set([start]),q=[start];for(let z=0;z<q.length&&selected.size<size;z++)for(const v of shuffle([...undirected.get(q[z])]))if(!selected.has(v)){selected.add(v);q.push(v);if(selected.size>=size)break}if(selected.size===size)return selected}return null}
const stats={source:'SNAP Gnutella08 from a GitHub mirror',source_blob_sha:'f6d6ea2941ef6c892abe5be37a6b48c403d8c0f3',
 input_sha256:crypto.createHash('sha256').update(fs.readFileSync(file)).digest('hex'),
 original_vertices:vertices.length,original_edges:edgeCount,seed:34789,
 n_draws:512,neighborhood_nodes:48,valid_swaps:0,failed_swaps:0,answer_changing_swaps:0,
 label_patterns:{},both_classes:0,total_swap_attempts:0,total_neighborhood_edges:0,
 all_zero_cqa_baseline:0,selection_policy:'uniform random source and two different targets, no outcome conditioning'};
let zeroRight=0;
for(let trial=0;trial<stats.n_draws;trial++){
  const chosen=sample(48);if(!chosen){stats.failed_swaps++;continue}
  const S=new Map(),edges=[];
  for(const u of chosen){const outs=[...adjacency.get(u)].filter(v=>chosen.has(v));S.set(u,new Set(outs));for(const v of outs)edges.push([u,v])}
  let picked=null,attempts=0;
  for(let t=0;t<1500;t++){
    attempts++;const [a,b]=choice(edges),[c,d]=choice(edges);
    if(new Set([a,b,c,d]).size!==4||S.get(a).has(d)||S.get(c).has(b))continue;
    picked=[a,b,c,d];break;
  }
  if(!picked){stats.failed_swaps++;continue}
  const [a,b,c,d]=picked,T=new Map([...S].map(([k,v])=>[k,new Set(v)]));
  T.get(a).delete(b);T.get(c).delete(d);T.get(a).add(d);T.get(c).add(b);
  if(T.get(a).size!==S.get(a).size||T.get(c).size!==S.get(c).size)throw Error('degree changed');
  const nodes=[...chosen],src=choice(nodes),tg=shuffle(nodes.filter(v=>v!==src)).slice(0,2);
  const y=[bfs(S,src,tg[0]),bfs(S,src,tg[1]),bfs(T,src,tg[0]),bfs(T,src,tg[1])].map(Number);
  stats.valid_swaps++;stats.total_swap_attempts+=attempts;stats.total_neighborhood_edges+=edges.length;
  if(y[0]!==y[2]||y[1]!==y[3])stats.answer_changing_swaps++;
  if(new Set(y).size===2)stats.both_classes++;
  const k=y.join('');stats.label_patterns[k]=(stats.label_patterns[k]||0)+1;
  if(y.every(v=>v===0))zeroRight++;
}
stats.all_zero_cqa_baseline=zeroRight/stats.valid_swaps;
stats.mean_neighborhood_edges=stats.total_neighborhood_edges/stats.valid_swaps;
stats.mean_swap_attempts=stats.total_swap_attempts/stats.valid_swaps;
stats.answer_change_fraction=stats.answer_changing_swaps/stats.valid_swaps;
stats.both_classes_fraction=stats.both_classes/stats.valid_swaps;
console.log(JSON.stringify(stats,null,2));
