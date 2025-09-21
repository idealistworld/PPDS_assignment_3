# Specific prompts used and why

PROMPT_JSON_SCHEMA = (
'{\n'
' "sentiment": "bullish|bearish|neutral",\n'
' "score": float_between\*-1_and_1,\n'
' "predicted_change_percent": float_price_change_prediction,\n'
' "reasoning": "brief_explanation_in_10_words_or_less"\n'
'}\n'
)

PROMPT_MULTI_TEMPLATE = (
"{persona_context}Analyze the sentiment of these Solana cryptocurrency\n"
"news articles and predict price impact.\n"
"For each article, respond with a JSON object in this exact format:\n"
"{schema}\n\n"
"Respond with a JSON array containing one object per article in order.\n\n"
"{articles_text}\n"
)

PROMPT_SINGLE_TEMPLATE = (
"{persona_context}Analyze the sentiment of this Solana cryptocurrency\n"
"news article and predict price impact.\n"
"Respond with ONLY a JSON object in this exact format:\n"
"{schema}\n\n"
"Article text: {text_excerpt}\n"
)
Respond with a JSON array containing one object per article in order.

{articles_text}

I decided to abstract the prompts to a centralized variable vs inline to make it easier in case they were to be edited in the future. Within them I also broke down different parts of the prompt to be a bit more modular and readable. We first have the persona context which will determine the tone of the response and viewpoint. Some of the personas are more conservative and some are more pro crypto. The next thing to note is that we have a centralized value for the JSON response schema, making the code more readable since we're defining the type. Finally we have two versions of the core prompt to account for both single prompt cases and batch cases. Single prompt cases are only used when an error is raised on a batch request as to not invalidate the entire response.
The only real difference here is the format of the response, but the code can handle both. In the future we might want to expand this prompt, but it's easy to build on the foundation prompt with the way it's all set up right now.

# Most effective enrichment strategies

Out of the different approaches I tried and considered, I wanted one that focused on the human interpretation aspect of LLMs. I initially tried and thought about using the Deepseek integration to focus on something more simple like translation, but ultimately decided that a "persona based approach" would be the most interesting. That's why the prompts ask for more sentiment based responses as well as a number. The usage here combines both qualitative and quantitative data, I really think it's interesting when LLMs give quantitative data because it helps me better understand their "tone", to give a more cohesive picture.

# Challenges and solutions

I think the main challenge here was finding the data that would make for an interesting LLM integration. I also think doing a project based on morality would have been interesting. I'm sure there's some data set that would have made sense for that use case and it would have been especially interesting to see how different models would've ranked the same situation. One other interesting project would be to pull news articles from across the political spectrum and have different LLMs give scores and track on aggregate how they leaned.

# Cost estimate

On scale the costs could be very high, but for this project I helped optimize costs through batching API calls. This way we save on tokens when it comes to processing many articles at once.

# Creative applications you discovered

I think the most interesting application here was the mixing of qualitative and quantitative data from the LLM. While the semantic response and predications are great, it's really up to interpretation and the actual sentiment can get lost. When I did one of the original iterations of this project I was frustrated with how just the pure analysis was not enough. I also think it's interesting to have an LLM grade itself on how accurate it was, and in the future, it'd be another awesome feature to add. I.e. why was this prediction right or wrong and what are the key biases as to why this might be the case.
