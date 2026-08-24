# Home Assistant LLM Router

This service presents an OpenAI-compatible chat-completions endpoint to Home
Assistant and forwards requests to Ollama. It separates Home Assistant work from
general knowledge work so device state is handled by the fast local model and is
not exposed to the general-purpose model.

## My Environment:
This is customized to my home lab environment. I have a couple ubuntu servers running my containers, and one 
beefier server that has a RTX3060 12GB card installed, which I call my 'aiserver'. All of the HA-LLM-Router components, 
including the docker files and main code, were built from inputs I gave ChatGPT (plus), and I was pretty impressed 
with the results. This is the second version, as I found that the LLM originally made a lot of mistakes 
and recently switched what models I used. I've also updated the way the router detects commands and intents 
so that it uses smaller models for actions better, reduces the number of entities sent, and 
can handle general purpose questions.

Inside Home Assistant I do have the "Extended OpenAI Conversation" HACS component installed, so I can use the 
HA-LLM-Router as my model. The latest version of the router code allows you to customize (well, its hard coded in the 
code itself...) the parameters used so in some sense it doesn't matter what you configure the integration 
for, for Top P, Temperature, etc.

## Request flow

```text
Home Assistant
      |
      v
ha-llm-router
      |
      +-- HA control ------> FAST_MODEL + relevant entities + HA control tools
      |
      +-- HA status query -> FAST_MODEL + relevant current entity states
      |
      +-- HA summary ------> SUMMARY_MODEL + broader current HA context
      |
      `-- General ---------> DEFAULT_MODEL, without HA entities or HA tools
```

Examples:

| Request | Classification | Model | HA context |
| --- | --- | --- | --- |
| `Turn off the kitchen lights` | `control` | `FAST_MODEL` | Best matching kitchen lights |
| `What's the kitchen temperature?` | `status` | `FAST_MODEL` | Best matching kitchen temperature entities |
| `Give me a morning briefing` | `summary` | `SUMMARY_MODEL` | Broader, capped HA state selection |
| `Why is the sky blue?` | `general` | `DEFAULT_MODEL` | None |

The model sent in the incoming request is deliberately ignored. The router owns
model selection so Home Assistant cannot accidentally send an HA query to the
large general-purpose model. It also overrides incoming generation parameters
with the policy for the selected route:

| Route | Temperature | Top P | Maximum output tokens |
| --- | ---: | ---: | ---: |
| `control` | `0.0` | `0.9` | `256` |
| `status` | `0.0` | `0.9` | `256` |
| `summary` | `0.2` | `0.9` | `768` |
| `general` | `0.4` | `0.95` | `2048` |

These limits apply to generated output, not the input context. Control and
status are deliberately deterministic and concise; summaries and general
answers receive progressively larger response budgets.

## Classification

`classify_request()` uses control verbs, question forms, device/domain words,
and known area names to assign one of four modes:

- `control`: an HA action, including voice shorthand such as `kitchen lights off`.
- `status`: a question about current HA state.
- `summary`: a briefing, weather, calendar, or home-summary request.
- `general`: anything not recognized as HA-related.

The known domain and area terms are defined in `DOMAIN_HINTS` and `AREA_HINTS`
in `app/main.py`. Add household-specific room aliases to `AREA_HINTS` when
needed.

## Entity retrieval

For every HA request, the router reads `/api/states` from Home Assistant and:

1. Narrows candidates using domain hints such as `temperature`, `light`, or
   `door`.
2. Scores candidates using the entity ID, friendly name, device class, area
   words, and request words.
3. Keeps entities whose score is within `ENTITY_SCORE_WINDOW` of the best match.
4. Sends no more than `MAX_RELEVANT_ENTITIES` compact entity records.

This means a kitchen-temperature query should include the kitchen temperature
sensor rather than dozens of unrelated sensors. If a request is intentionally
ambiguous, such as `turn off the lights`, the top matching lights are included
up to the configured limit so the model can act on them or ask for clarification.

Summary requests are different by design: they may use up to
`MAX_ENTITIES_SENT` entities because their purpose is to describe a wider view
of the home. General requests never fetch HA states and never receive HA tools.

The router provides current states to the model; it does not itself call Home
Assistant services. For controls, the model returns one of the tool calls
provided by the Home Assistant integration, and Home Assistant executes it.

## Configuration

The container reads configuration from its environment (the Compose service
uses `/containers/volumes/ha-llm-router/.env`).

| Variable | Default | Purpose |
| --- | --- | --- |
| `ROUTER_API_KEY` | empty | Bearer token required by router clients; empty disables router authentication |
| `OLLAMA_BASE_URL` | `http://host.docker.internal:11434/v1` | Ollama OpenAI-compatible API |
| `DEFAULT_MODEL` | `qwen2.5:14b-instruct` | Large model for general requests |
| `FAST_MODEL` | `mistral:7b-instruct` | Small model for HA control and status requests |
| `SUMMARY_MODEL` | value of `FAST_MODEL` | Model for HA summaries |
| `HOME_ASSISTANT_URL` | empty | Home Assistant base URL |
| `HOME_ASSISTANT_TOKEN` | empty | Long-lived HA access token |
| `MAX_RELEVANT_ENTITIES` | `8` | Maximum entities for control/status requests |
| `ENTITY_SCORE_WINDOW` | `20` | Maximum score distance from the best entity match |
| `MAX_ENTITIES_SENT` | `40` | Maximum entities for summary requests |
| `LOG_LEVEL` | `INFO` | Python log level |
| `DEBUG_LLM_PAYLOADS` | `false` | Log request/response payloads; may expose private HA data |

For the proposed model layout, set `FAST_MODEL` to the installed Ollama name for
Qwen 3.5 4B Q8 and `DEFAULT_MODEL` to the installed name for Gemma 4 12B Q4.
Exact Ollama model tags depend on how those models are installed.

## API

- `GET /health`: unauthenticated health check.
- `GET /v1/models`: lists the default and fast models.
- `POST /v1/chat/completions`: OpenAI-compatible routed chat endpoint.

When `ROUTER_API_KEY` is set, `/v1/models` and `/v1/chat/completions` require
`Authorization: Bearer <ROUTER_API_KEY>`.

## Layout

```text
ha-llm-router/
|-- app/
|   `-- main.py          Router, HA retrieval, prompt injection, Ollama proxy
|-- Dockerfile           Container image definition
|-- requirements.txt     Python dependencies
|-- how-to-build.txt     Existing manual build/deployment notes
`-- readme.md            This document
```

The repository's Ansible deployment copies this directory to
`/containers/volumes/ha-llm-router` on `aiserver`. The `ha-llm-router` service is
defined in `ansible/files/docker/aiserver/containers/config/compose.yml` and
exposes port `8088` behind Traefik.

## Automated deployment

`ansible/playbooks/deploy-ha-llm-router.yml` performs a router-only deployment:

1. Validates the required build-context files on the Ansible controller.
2. Synchronizes them to `/containers/volumes/ha-llm-router` on `aiserver`.
3. Preserves and validates the server-managed `.env` file.
4. Validates the `ha-llm-router` service in `/containers/config/compose.yml`.
5. Runs `docker compose up -d --build ha-llm-router`.
6. Retries `http://127.0.0.1:8088/health` until it returns HTTP 200 and
   `{"ok": true}`.

The Compose file must already be deployed and the following file must be
created directly on `aiserver` before the first router deployment:

```text
/containers/volumes/ha-llm-router/.env
```

The deployment deliberately excludes `.env` from synchronization because it
contains credentials. Run the playbook manually with:

```bash
ansible-playbook \
  -i ansible/inventory.yml \
  ansible/playbooks/deploy-ha-llm-router.yml
```

`.github/workflows/deploy-ha-llm-router.yml` validates pull requests. It deploys
after matching changes reach `main`, and it can also be deployed manually using
the workflow-dispatch button.
