# Exact prompts used with AI tools

Let's set up a script to call TheNewsAPI endpoint
https://api.thenewsapi.com/v1/news/all and store the key in `.env` at root.

Let's update our .gitignore

[Pasted text #1]

here are the formats of articles, let's use pandas to make a table for them and start exporting it to enriched so we
can have a test for it

so baseline let's export the raw json to the raw folder

do this for solana specific keyword contained in articles

ok let's change our api call to only return 3 articles per any given day

let'sb build out the enrichment one for the deepseek file, key is in root .env

let's do solana_news.csv

in just the raw

then let's also export a table with all the articles along with their score of sentiment bullsih / bearish for solana

ok now let's also add a prediction for the price that day and the prcie jump from the previous day

ok le'ts get rid of the estimated % change

also can we aggregate api calls i think so that way we can cut down time for the api calls

also let's provide a super small summary of why that sentiment is the case

let's have the program first start by asking for the last x days or wahtever to pull from

let's add some additional investor personas that they can review from the perspective of

ok le'ts also le them add a keyword of interest like an area of solana investing htye're interested in

refactor and condense redundant functions

make sure that all files adhere to the styling guidelines

update the installation section and put the base data for the enrichment section

update the requirements.txt

# What code was AI-generated vs human-written

Around 90%+ of the code was AI generated in this project. I still reviewed each method and did the writing for these .md files, but around 90%+ of code was or is AI.

# Bugs found in AI suggestions and fixes

- Passing wrong values to endpoint, had to fix formatting / input
- Redundant methods, cleaned them up and simplified down to necessities
- Too many fields and data passed to final enriched file, cut down actual fields to make enriched data more usable
- Poor UX and ordering of initial questions to form API call, fixed to be more human focused i.e. pick area, then analyst type, then date range
- Too much logging to console about success and failure that was useful while debugging, removed and simplified down logs to be more human readable and provide insight without being overwhelming to where it's unusable

# Performance comparisons (if tested)

- Used combination of Codex and Claude
- Claude was faster
- Codex I feel like had more thought out answers / thought about the entire task vs taking action whereas Claude would start coding and iterate as it discovered that its approach might have been suboptimal
