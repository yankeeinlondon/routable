# LLM Runners

## Runners

An LLM Runner is a piece of software that provides an API surface to clients wanting to interact with a local LLM model. There are many solutions out there but we can group them into two groups:

1. Single Model Runners

    - runners which -- when run -- expose one and only one model (at a time)
    - some may be able to swap models during their lifecycle whereas others will not
    - examples of runners that fit into this category include:
        - vllm
        - llama.cpp

2. Aggregation Model Runners

    - the _aggregate_ model runners combine model running with a _router_ built into the API surface
    - this type of local runner operates in a way that is more similar to what the cloud providers typically expose to users
        - the `v1/models` route (from OpenAI's informal standard) not only lists all the models but actually serves them without the client needing to do anything
        - when a model that is NOT loaded into the local resources (aka, CPU and GPU) the _router_ functionality of that solution will do what is necessary to meet the API's obligations in a resource efficient manner
    - examples of runners that fit into this category include:
        - ollama
        - LM Studio (though it's footprint is even larger)
        - oMLX

## LLM Router's Relationship with Runners

To start we will focus on **vllm** but soon afterward we will add in **llama.cpp**. Both of these solutions fit into the "Single Model Runners" category and this is no accident as our _routing_ functionality brings a huge amount of complimentary benefits to a solution that is really only geared for running a single model (at a time) across the the exposed API. 

> Note: **ollama** through most of it's history was really a router, a pretty UI, and some metadata. Under the hood the actual model serving was **llama.cpp**. They have more recently started to do their own serving but that's a very modern change.

So one might assume that this LLM Router is intended to _only_ compliment Single Model Runners but that is not true, however, it is the initial focus for this project. Once we're got a working solution for **vllm** and **llama.cpp** we will start looking at two new directions that _functionally_ are actually more similar then they are different:

1. Peer Communication and Multi-host Serving (version 3)

    To start we are "land locked" to a single host as our _routing_ functionality is delivered by having direct access to CPU/GPU resources as well as process control over the LLM runners we are fronting. Even this land locked state adds a lot of value but if a user has a fleet of hosts, all with their own CPU/GPU resources being made available then having these various hosts be able to communicate with one another in a way that _scheduling_ can be done across the fleet will make for a much more compelling and flexible solution.

    In this area we will expose this "fleet" of resources by having all **LLM Routers** communicate across a peer-to-peer network as a back channel to communicate metadata with one another (e.g., active models, current load, etc.) as well as a mechanism for any of the peers to proxy a request given to them to another peer for execution.

2. Third Party Adaptors / Integrations (version 4)

    While the **Peer Communication and Multi-host Serving** section represents a fully self-contained solution meant to work across Linux servers in a users or companies homelab/office network it's important to understand that a lot of developers are working of macOS or Windows computers which also have local serving capacity on them. The Apple silicon Macs are often really good options and over time more of the Windows ones will be too. For users, we want to be able to connect these ollama, omx, LM studio, resource pools into the larger fleet pool. How this is designed can be design later when we're closer to the implementation.
