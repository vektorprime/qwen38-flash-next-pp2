"""Fixed greedy workload: MTP acceptance + single-stream decode tok/s. usage: bench.py <tag>"""
import json, sys, time, re, urllib.request
BASE=sys.argv[2] if len(sys.argv) > 2 else "http://127.0.0.1:8001"; M="qwen38-flash-next-awq"
P=["Explain how a B-tree insertion works, step by step.","Write a Python function that parses an ISO-8601 duration string.",
"Summarize the causes of the French Revolution.","Derive the formula for the sum of a geometric series.",
"Write a bash script that finds the 10 largest files under a directory.","What is the difference between TCP and UDP? Be detailed.",
"Implement quicksort in Rust with comments.","Explain the attention mechanism in transformers mathematically.",
"Write a SQL query to find the second highest salary per department and explain it.","Describe how photosynthesis works at the molecular level.",
"Prove that the square root of 2 is irrational.","Write a JavaScript debounce function and explain its use.",
"Explain the CAP theorem with examples.","How does a Kalman filter work? Give the update equations.",
"Write a C function to reverse a linked list in place.","Explain Bayes' theorem with a medical testing example."]
def metrics():
    t=urllib.request.urlopen(BASE+"/metrics").read().decode()
    return {k:float(re.search(r"^vllm:%s\{[^}]*\} (\S+)"%k,t,re.M).group(1)) for k in
            ["spec_decode_num_drafts_total","spec_decode_num_draft_tokens_total","spec_decode_num_accepted_tokens_total"]}
m0=metrics(); toks=0; secs=0; outs=[]
for p in P:
    body=json.dumps({"model":M,"messages":[{"role":"user","content":p}],"max_tokens":256,"temperature":0,
                     "chat_template_kwargs":{"enable_thinking":False}}).encode()
    t=time.time(); r=json.load(urllib.request.urlopen(urllib.request.Request(BASE+"/v1/chat/completions",data=body,headers={"Content-Type":"application/json"}),timeout=600))
    secs+=time.time()-t; toks+=r["usage"]["completion_tokens"]; outs.append(r["choices"][0]["message"]["content"])
m1=metrics(); d={k:m1[k]-m0[k] for k in m0}
res={"tag":sys.argv[1],"acceptance":d["spec_decode_num_accepted_tokens_total"]/d["spec_decode_num_draft_tokens_total"],
     "accepted_per_step":d["spec_decode_num_accepted_tokens_total"]/d["spec_decode_num_drafts_total"],"tok_per_s":toks/secs,"tokens":toks}
json.dump({"res":res,"outputs":outs},open("/home/user/qwen3nextflash/batchinv/results/bench_%s.json"%sys.argv[1],"w"),indent=1); print(json.dumps(res))
