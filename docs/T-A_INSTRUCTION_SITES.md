# T-A: question vs answer-format instruction sites

T-A (embedding-ranked word substitution, ladder 1/2/4/8, NLI >= 0.8, CoLA >= 0.9 x original, dictionary words, no auxiliaries) on question + instruction; protected words and the option words fixed. Variants = product over sites of (levels + 1), i.e. every combination including 'unchanged'.

| question | options | question sites | question variants | instruction sites | instruction variants | combined |
|---|---|---|---|---|---|---|
| What is the main subject in this image? | marimba, bird | 4 | 225 | 4 | 135 | 30375 |
| What is the main subject in this image? | string instrument, bird | 4 | 225 | 6 | 720 | 162000 |
| Is there a person in the image? | Yes, No | 3 | 27 | 5 | 360 | 9720 |
| Is there a dining table in the image? | Yes, No | 4 | 54 | 4 | 90 | 4860 |
| Is there a broccoli in the image? | Yes, No | 3 | 36 | 5 | 270 | 9720 |
| Is there a fire hydrant in the image? | Yes, No | 3 | 24 | 4 | 90 | 2160 |
| Is there a car in the image? | Yes, No | 3 | 24 | 5 | 270 | 6480 |
| Is there a dog in the image? | Yes, No | 2 | 9 | 5 | 270 | 2430 |
| Is there a toilet in the image? | Yes, No | 2 | 9 | 5 | 180 | 1620 |
| Is there a remote in the image? | Yes, No | 2 | 12 | 5 | 180 | 2160 |
| Is there a tv in the image? | Yes, No | 2 | 9 | 5 | 270 | 2430 |
| Is there a handbag in the image? | Yes, No | 2 | 9 | 5 | 270 | 2430 |

## Substitutes per site

**What is the main subject in this image?** (options marimba, bird)

- question: the -> that, a; main -> major, general, primary, top; in -> within, from, inside, for; this -> the, that
- instruction: Answer -> Reply, Comment; with -> for, to; exactly -> just, precisely; options -> choices, suggestions, responses, terms

**What is the main subject in this image?** (options string instrument, bird)

- question: the -> that, a; main -> major, general, primary, significant; in -> from, within, inside, to; this -> the, that
- instruction: Answer -> Reply, Comment, Label; with -> from, for; exactly -> just, precisely; of -> out; these -> this; options -> choices, suggestions, responses, terms

**Is there a person in the image?** (options Yes, No)

- question: a -> some, any; in -> within, inside; the -> this, an
- instruction: Answer -> Response, Reply, Match; with -> to, below; exactly -> precisely, just; these -> those; options -> choices, alternatives, terms, values

**Is there a dining table in the image?** (options Yes, No)

- question: dining -> dinner, food; table -> establishment; in -> within, inside; the -> this, that
- instruction: Answer -> Reply, Response; with -> to; exactly -> precisely, just; options -> choices, responses, keywords, words

**Is there a broccoli in the image?** (options Yes, No)

- question: a -> some, any; in -> within, inside, about; the -> this, that
- instruction: Answer -> Reply, Response; with -> to, below; exactly -> precisely, just; these -> those; options -> choices, responses, keywords, terms

**Is there a fire hydrant in the image?** (options Yes, No)

- question: a -> the; in -> on, within, across; the -> this, that
- instruction: Answer -> Reply, Response; with -> to; exactly -> precisely, just; options -> choices, alternatives, criteria, terms

**Is there a car in the image?** (options Yes, No)

- question: a -> any; in -> on, within, across; the -> this, that
- instruction: Answer -> Reply, Response; with -> to, for; exactly -> precisely, just; these -> those; options -> choices, responses, keywords, values

**Is there a dog in the image?** (options Yes, No)

- question: in -> on, within; the -> this, that
- instruction: Answer -> Reply, Response; with -> to, for; exactly -> precisely, just; these -> those; options -> choices, responses, keywords, prompts

**Is there a toilet in the image?** (options Yes, No)

- question: in -> on, within; the -> this, that
- instruction: Answer -> Reply, Response; with -> to; exactly -> precisely, just; these -> those; options -> choices, responses, keywords, words

**Is there a remote in the image?** (options Yes, No)

- question: in -> within, on, to; the -> this, that
- instruction: Answer -> Reply, Comment; with -> to; exactly -> precisely, just; these -> those; options -> choices, alternatives, keywords, definitions

**Is there a tv in the image?** (options Yes, No)

- question: in -> within, inside; the -> this, that
- instruction: Answer -> Reply, Response; with -> to, for; exactly -> just, precisely; these -> those; options -> choices, responses, criteria, prompts

**Is there a handbag in the image?** (options Yes, No)

- question: in -> within, inside; the -> this, that
- instruction: Answer -> Reply, Response; with -> to, via; exactly -> precisely, just; these -> those; options -> choices, responses, keywords, definitions

