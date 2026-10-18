# test_host_only_params::test_bad_words (runner full profile, readiness_vllm/sampling_tests.log)

Failure: response 4 (seed 4, temperature 1.0, chat, 5 concurrent requests) contains the word "hello".

Token-level replay on the same server build (doc/vllm_integration/logs/server_full_r2.log), same 5 concurrent requests:
- banned token ids (vLLM bad_words canonical tokenizations): 'hello' 52071, ' hello' 63532, 'Hello' 35808, ' Hello' 59959
  -> never generated in 3 x 5 replays (banned-token-positions [] for every response).
- the word comes from a non-canonical split: [(94866,'Say'), (267,' a'), (18069,' hell'), (87,'o'), (310,' to')].
- reproducible: seed 4 in the 5-request batch produced it in all 3 replays (deterministic host sampling).

Classification: vLLM bad_words semantics (bans exact token sequences of the word's tokenizations, applied by vLLM's
host sampler; bad_words always selects host sampling on TT).  The adapter only returns logits for this step; nothing
TT-specific decides whether ' hell' + 'o' may follow.  The test's whole-word string check is stricter than the bad_words
contract, so this is not a serving-correctness failure.  Not reproducibility-only either: recorded as a
test-vs-vLLM-semantics mismatch for this tokenizer.
