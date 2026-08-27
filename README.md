# reddit-feature-request-harvester

Searches Reddit for people asking a SaaS product for something it doesn’t do, classifies each hit
with an LLM, clusters the survivors into themes, and appends the new ones to a Google Sheet. Runs
weekly in GitHub Actions on a schedule and costs nothing but the OpenAI calls.

The output answers a question competitor pages can’t: what are users of a category actually asking
for, in their own words, this quarter.

## How a run goes

1. **Search.** Every phrase in `EXPANSION_PHRASES` is queried against Reddit, optionally widened by
   embedding-nearest terms. Accepted expansions are cached in `phrase_cache.json`, so recall grows
   run over run instead of re-paying for the same expansion.
2. **Throttle.** Requests are paced to `REQUESTS_PER_MIN`. Reddit’s API is generous and unforgiving
   in equal measure; this is the part you don’t remove.
3. **Classify.** Candidates go to the LLM until `MAX_CLASSIFY`, and anything under `CONF_THRESHOLD`
   is dropped. The default of 0.8 is deliberately strict – a feature-request list with junk in it
   stops getting read by week three.
4. **Cluster.** Survivors are grouped into themes so the sheet shows twelve recurring asks rather
   than four hundred posts.
5. **Score companies** (optional). Vendors named repeatedly across requests are surfaced with counts,
   which is how you find the products whose users are loudest. `ENABLE_COMPANY_SCORES=0` skips it.
6. **Append.** Only rows not already in the sheet are written; `state.json` carries the seen set.

## Set it up

Secrets, as GitHub Actions repository secrets (or environment variables locally):

| Secret | For |
|---|---|
| `REDDIT_CLIENT_ID`, `REDDIT_CLIENT_SECRET`, `REDDIT_USERNAME`, `REDDIT_PASSWORD` | a script-type Reddit app |

Note what that last one means: Reddit's script grant takes an account **password**, so this puts a
credential capable of full account takeover into your CI secrets. Use a dedicated account that owns
nothing, not your main one, and remember that any workflow change or third-party action you add later
runs with access to it.
| `OPENAI_API_KEY` | classification, clustering, embeddings |
| `PRODUCT_NAME` | how the LLM prompts frame what you are building |
| `SHEET_ID` | the target spreadsheet – **required**, and the run aborts rather than creating one |
| `GCP_WORKLOAD_IDENTITY_PROVIDER`, `GCP_SERVICE_ACCOUNT_EMAIL` | Sheets access from CI via Workload Identity Federation, so no service-account JSON is stored anywhere |

The sheet needs three tabs to exist first: `Requests`, `Clusters`, `Companies` (renameable through
`SHEET_TAB`, `SHEET_CLUSTERS_TAB`, `SHEET_COMPANIES_TAB`). Share it with the service account address.

Then:

```bash
pip install -r requirements.txt
python main.py
```

`.github/workflows/reddit_feature_requests.yml` runs it Mondays at 09:00 UTC and on manual dispatch.

## Tuning

| Variable | Default | Effect |
|---|---|---|
| `EXPANSION_PHRASES` | `feature request` | pipe-separated seed queries – the highest-leverage setting here |
| `LOOKBACK_DAYS` | `90` | how far back a post can be and still count |
| `PAGES_PER_QUERY` / `LIMIT_PER_PAGE` | `5` / `100` | crawl depth per phrase |
| `MAX_CLASSIFY` | `2000` | hard ceiling on LLM calls per run – your cost control |
| `CONF_THRESHOLD` | `0.8` | precision/recall dial on classification |
| `REQUESTS_PER_MIN` | `40` | Reddit pacing |
| `ENABLE_SUB_DISCOVERY` | `0` | crawl subreddits found through company mentions (slower, wider) |
| `ENABLE_FIT_SCORING` | `0` | score each request against your own product |
| `TOP_N` | `25` | how many clusters get written |

## What it writes

`docs/data/summary.json` holds per-run counts (searched, classified, clustered, appended) and nothing
else – it ships zeroed here. `phrase_cache.json` and `state.json` are the caches, also empty on
arrival; they fill on your first run and are yours.

## Before you point it at a subreddit

Reddit posts are written by people who did not sign up to be a lead list. This tool exists to hear
what a category’s users are asking for, and it writes themes and counts. If you extend it into
contacting the posters, you have changed what it is, and Reddit’s rules and the subreddit’s own will
have something to say about that.
