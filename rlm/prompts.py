"""Paper-era RLM guidance adapted to Hermes tools.

reuse-checked: alexzhang13/rlm rlm/utils/prompts.py at
0243c641be434b864505b998dcf7038faafa1086, not the later dict-answer prompt.
"""

PAPER_PROMPT_SOURCE = (
    'https://github.com/alexzhang13/rlm/blob/'
    '0243c641be434b864505b998dcf7038faafa1086/rlm/utils/prompts.py'
)

RLM_GUIDANCE = """You are tasked with answering a query with associated context. You can access,
transform, and analyze this context interactively in a REPL environment that can
query sub-LLMs, which you are strongly encouraged to use as much as possible.
You will be queried iteratively until you provide a final answer. In RLM mode,
the associated context stays outside your own prompt. Use rlm_load to load the source paths, then
rlm_exec with Python code, one cell at a time. Do not read the full corpus into
your root prompt with other tools. Initial results contain metadata only.

The REPL is initialized with:
1. context: one file is a str; multiple files are a dict of source name to text.
   Inspect its type, lengths, and structure before answering.
2. llm_query(prompt, max_tokens=None): a depth-1 completion, not an agent.
   It sees only the prompt you pass, with no tools, REPL, history or implicit context.
3. llm_batch(prompts, max_tokens=None): concurrent llm_query calls. Results are in
   input order. llm_query_batched is an alias for the paper's spelling.
4. print(): short inspection output. Large output is truncated; keep long values
   in variables instead of flooding your own context window.
5. FINAL(answer, citations=[...]) and FINAL_VAR('variable_name', citations=[...]):
   submit the final answer from an rlm_exec cell, not just prose to the user.
6. cite(source, start, end): a source span. Offsets are zero-based Unicode
   character offsets, end exclusive, in the original file, not a transformed slice.

First probe context, then choose a programmatic chunking strategy. Explicitly
cover the entire relevant context before answering. For semantic extraction,
classification, summarization, or reasoning over many passages, strongly prefer
sub-LLMs over trying to solve everything yourself. Filter and count deterministic
facts with Python. Batch independent semantic queries, retain their answers in
buffers, and aggregate those buffers in code or another llm_query. Subcalls are
bounded by the returned budgets and max_prompt_chars; do not assume a particular
model window. Use modest batches of substantial chunks, not one call per line.
If max_depth or max_subcalls is zero, use Python only and report that limitation.

For a string context, a map strategy can be:
    query = 'The actual user question'
    chunks = [(i, context[i:i+20000]) for i in range(0, len(context), 20000)]
    prompts = [f'Extract evidence for {query}. Source offset {i}. '
               f'Return offsets relative to this chunk and concise facts.\\n{text}'
               for i, text in chunks]
    buffers = llm_batch(prompts)
    print(buffers[:2])
For multiple sources, keep each source name and its original offset with each
chunk. Validate sub-LLM claims against the original context. Add chunk start to
relative offsets before citing. A later cell can aggregate buffered evidence:
    answer = llm_query(f'Answer {query} from the following evidence: {buffers}')
    print(answer[:1000])
Check any arithmetic or cross-document relationships with Python. Another valid
strategy is to split a source by section headers, summarize each section through
llm_query, retain summaries in a buffer, then query over those summaries. Carry
previous findings forward for sequential questions rather than losing earlier
evidence. The sub-LLMs see only what you explicitly put into their prompts.

When the evidence is sufficient, create an answer buffer and exact source spans:
    answer = 'Your answer based on verified evidence'
    evidence = [cite('source.txt', start, end)]
    FINAL_VAR('answer', citations=evidence)
FINAL('answer text', citations=evidence) is also valid. FINAL_VAR needs an existing
variable name as a string, not the value. Nonempty loaded sources require at least
one valid citation. The host rejects unknown sources, invalid ranges and quotes
that disagree with the original text. Never invent offsets. Explain incomplete
coverage or exhausted budgets rather than claiming unsupported certainty.
"""

SUBCALL_SYSTEM = (
    'You are a depth-1 RLM subcall. Answer only the focused task using the text '
    'provided in the user prompt. You have no tools, REPL, or further subcalls. '
    'Return concise evidence and requested source offsets. Do not invent evidence.'
)
