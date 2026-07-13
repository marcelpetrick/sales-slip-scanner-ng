vision models grow i their capabilities while the execution demands way less ressources.

Means: the sales slip scanner from two years ago needed an upgrade. befor i was using gpt4-vision as API. whjich costs you the control over your data and some cent for the analysis.

now, after intensive benchmarking over a range of 15 models (with ollama) and some surprising regressiions (news harness does not mean the mdoels excute faster, itz can also mean they suddenly run int o model load errors). i put my mojey on Qwen3.5:4b Vsision. ut of the huge sampel size of three slaes slips all thee were properly evlauted.

built a hot-folder around the script and it works (thank you @mr zahorsky without him i would mhve never encountered this idea).

data entry can be done for cheap: give kids the smartphone, snap all receipts, throw out some water melon sugar slices ;)


----------
Read the draft — good bones, casual and honest. Here are ten things worth adding to round it out:

1. The headline number — "Qwen 3.5 4B: 9/9 exact, across 135 trials" as an opening hook line. Concrete stats stop the scroll better than prose.
2. The hardware detail — you ran this on an RTX A2000 8 GB laptop GPU, not a data-center card. That's the encouraging part for readers without big rigs — say it explicitly.
3. The before/after line, tightened — "2024: images went to a cloud API. 2026: zero bytes leave the machine." One crisp sentence, no typos, as a pull-quote.
4. A visual — attach the benchmark dashboard screenshot or the blueprint-schematic GIF you already have in media/. Posts with an image/GIF get read more of the text below it.
5. The "surprising regression" lesson as its own bullet — reframe "newer harness ≠ faster model, some models suddenly failed to load" as a standalone takeaway line; it's the most relatable insight for anyone doing local-model evals.
6. Correct the credit — "@mr zahorsky" needs the actual LinkedIn handle/tag to notify them and read as a real thank-you, not a placeholder.
7. A repo link — one line pointing to the GitHub repo so people can actually try it, not just read about it.
8. The license/openness note — GPLv3, no API key, no signup — reinforces the "you can just run this" feeling.
9. A closing question — invite engagement: "what would you automate first with a local vision model?" Encouraging posts end with a door left open, not a full stop.
10. 2–4 hashtags, sparingly — e.g. #LocalLLM #Ollama #OpenSource — enough for discovery, not a wall of tags.

Want me to draft a cleaned-up full version of the post incorporating these, or just fix the typos in place and leave structure/content as-is?

✻ Cooked for 56s
----------

our core story is good: old cloud prototype → local benchmark → practical hot-folder workflow. Ten additions that would make it more
  encouraging:

  1. Start with a stronger hook, such as: “A €0 cloud bill and receipts that never leave my laptop.”
  2. Add a clear before/after comparison: remote API and per-image costs versus local execution and full data control.
  3. Include one concrete benchmark result: 15 models, 135 runs, and Qwen 3.5 4B achieving 9/9 correct extractions.
  4. Mention the hardware used. Showing that this works on an RTX A2000 8 GB makes the result relatable and reproducible.
  5. Emphasize the practical outcome: drop images into a folder and receive one Markdown report containing every receipt and the grand
     total.

  6. Keep your honest “huge sample size of three” joke, but clarify that this is a compatibility benchmark—not a statistically
     representative accuracy study.

  7. Share one surprising lesson: newer runtimes or larger models do not automatically produce faster or more reliable results.
  8. Add one of the project illustrations or the benchmark graphic. A strong visual will make the technical story much easier to notice on
     LinkedIn.

  9. Link to the repository and invite readers to test it with their own anonymized receipts or contribute additional benchmark images.
  10. Finish with an encouraging question: “Which small, forgotten cloud-dependent tool would you rebuild locally today?”

  Also give the final text a spelling pass before publishing; the current errors distract from an otherwise interesting and authentic
  project story.
