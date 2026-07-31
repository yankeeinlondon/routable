---
sequence: "@docs/runners.yaml"
prompt: |-
    Your task is to do deep research into the {{ state.desc }} local LLM runner and provide a rich prose 
    report followed by a structured fact sheet.

    Start by doing research and make sure you can answer _at least_ the following questions:

    - when was llama.cpp first released?
    - how active is this software being updated?
    - what is the history of the major versions of {{state.name}}
        - what features and functionality did each version introduce
        - what breaking changes were there if any
    - What is a good way to describe at a summary level what {{state.name}} is, who it serves, and what it's major use cases are?
    - What operating systems can {{ state.name }} be run on?
        - What is the binary name it typically takes on (per OS)
        - What is the typical file path (if any) where this binary will be located?
    - What are the CLI switches that the application provides for configurability?
    - What are the ENV variables which {{ state.name }} responds to for configuration advice/defaults?
    - What are the primary use case that this software addresses?
    - How does {{state.name}} distinguish itself from other LLM runners?
    - Describe the process required to run a downloaded model from Hugging Face:
        - does the model need to be compiled? wrapped with a script? binary files concatinated? metadata added?
        - roughly how long does it take to start {{state.name}} from a cold start? what are the steps in this process?
    - does {{ state.name }} have any functionality built around "caching", "reduced power state management", "sleep states", etc.
        - what is the nomenclature used?
        - what can be managed in these states? 
    - does {{ state.name }} allow for system RAM spillover for a model when VRAM can't fit the full model 
    - does {{ state.name }} allow for CPU inferences as well as GPU?
    - does {{ state.name }} provide any "dynamic model routing" so that when a client hits the server's endpoint but is asking for an LLM model that is not resident in memory, it get's loaded into memory before trying to serve requests to it
    - what advanced features does {{state.name}} provide?
        - speculative decoding? MTP? 
        - sliding window attention, 
        - KV cache CPU/GPU offloading, 
        - Chunked prompt prefill, 
        - FlashAttention, 
        - KV cache eviction policies
        - etc.

    
---
