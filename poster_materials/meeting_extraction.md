# Meeting Extraction: March 27, 2026 — Riedl, Glenn, Rehan, Isaac
## Full Information Report for Poster & Paper

---

## 1. PAPER STRUCTURE & FRAMING

### Core Story (as agreed upon in meeting)
- LLMs adopt personas, but is there geometric structure to this?
- We are NOT talking about psychometrics or occupational O*NET in this submission context
- We pivoted away from O*NET to have something presentable for COLM
- In this submission context: just "personas" broadly
- Personas are commonly done through prompting, but we're interested in the actual relationships of personas and how they organize internally
- We use steering activations to understand how a model behaves when asked to act like a persona
- Focus: how do roles, relationships organize themselves inside the geometry of the language model?

### Paper Title
- "Fantastic Personas and Where to Find Them" (working title, inspired by the mystical/fantastical cluster result)

### Two Major Sections
1. **Empirical Personification** — showing our steering method works better than state-of-the-art
2. **Manifold Geometry** — mathematical analysis of the persona vector space

---

## 2. EMPIRICAL SIDE — METHODOLOGY & RESULTS

### What We're Doing
- Comparing our contrastive steering methodology against the Assistant Axis paper (Christina Lu et al.) as the state-of-the-art baseline
- We have a **gold standard baseline**: prompt-engineered responses from a validated paper's prompt engineering approach
- We have **Christina's steered response** (assistant axis method) and **our steered response** (contrastive method) to the same prompt
- We use an **LLM-as-judge** scoring system, 0 to 100

### Judging Axes (5 total)
- **Style axes**: bias, emotional register, and similar
- **Content axes**: worldview alignment, motivation of the response
- Total: 5 axes of judgment comparing our response vs. Christina's response
- Essentially 5 different judge prompts

### Scale
- Currently completed: ~40 roles with full empirical results
- Target: **275 total roles** (Christina's full role set — we must do all 275 for apples-to-apples comparison)
- Pipeline: generating CSVs for ranges of alphas, then batch job for LLM inference via APIs
- Arjun is working on making visualizations aesthetic and publication-ready

### Key Empirical Claims
- Our contrastive method produces role vectors that score higher than Christina's assistant axis method according to the LLM judge across style and content
- "Number go up" — our methodology empirically beats the baseline
- Contributions: contrastive method + elicitation approach

### Riedl's Assessment of Empirical Side
- **"Very strong. Probably publishable on its own."**
- Need summary statistics across all 275 roles
- Show distributions, highlight typical patterns
- Can highlight weakest result and show it's still strong
- Don't need to show all 275 individually — summary stats suffice

### Baseline Comparison Decision
- **Primary baseline**: Christina Lu's Assistant Axis paper
- **Secondary paper acknowledged**: "Can Role Vectors Affect LLM Behavior?" (EMNLP 2025)
  - This paper defines role vectors and shows they can steer models
  - But methodologically it is NOT distinct from Christina's approach
  - Christina didn't even cite it (they were contemporaneous)
  - **Decision**: Acknowledge this paper exists in related work, but state it is methodologically not distinct from Christina's, so no separate baseline needed
  - Riedl: "You don't want to get caught with your pants down" — pre-rebut by acknowledging it exists
  - If reviewers ask: can do the comparison post-submission for rebuttal/appendix

### Visualization Plans for Empirical
- Axes showing style side and content side
- Roles plotted in hyperdimensional space showing our scores are higher
- Need final graphs once all 275 CSVs are done

---

## 3. MATHEMATICAL / GEOMETRY SIDE — METHODOLOGY & RESULTS

### 3A. Replicating Christina's Assistant Axis

**What we did:**
- Computed the assistant axis using Christina's methodology on OUR model and OUR role vectors
- Emile computed both: Christina's assistant axis for her role vectors AND our own assistant axis for our role vectors
- This is critical: Christina's paper states you must compute your own assistant axis when analyzing your own vectors
- All analysis of our data uses OUR computed assistant axis, never Christina's directly

**PCA Results (Christina's vectors):**
- PC1: 26% variance explained
- PC2: 11% variance explained  
- PC3: 8% variance explained
- Assistant axis = combination of these components (dotted/dashed line on plots)
- She needed ~19 PCs to explain 70% of variance

**Key observation about Christina's results:**
- When you look at her vectors, you cannot find a big behavioral difference between demon and lawyer, or between most roles
- Everything is kind of "one big blob" — roles are largely undistinguishable
- The assistant axis dominates: everything collapses to assistant-like behavior

### 3B. Our Results — Greater Variance / Separability

**Core claim: Our method produces greater variance among role vectors**
- Our role vectors show MORE separation/variance than Christina's
- We need MORE PCs than Christina needed to explain the same amount of variance (or: same number of PCs explains LESS variance)
- This is GOOD because it means our roles are actually differentiated, not collapsed
- Measured using PCA — this is mathematically grounded variance, not visual

**Why greater variance matters:**
- If all points occupy the same space, they'd all just be assistants
- Greater variance = the model is actually differentiating between personas
- A baby should be different from a demon should be different from a lawyer
- Separability is the key property we're looking for

**Riedl's framing:** "If we believe that there is difference between points, then we'd better see more variance."

### 3C. The Fantastical/Mystical Cluster Result

**How it was discovered (order of events matters):**
1. We did NOT go into this trying to separate mystical from non-mystical roles
2. We used our contrastive method to generate better role vectors
3. We did t-SNE visualization and saw two distinct clusters emerge
4. Glenn looked at the results and asked: why did this entire cluster move over? Are they different?
5. Upon inspection: one cluster = human/realistic roles, other cluster = fantastical/mystical/non-human roles

**The clusters:**
- **Main cluster (human/realistic):** doctor, assistant, secretary, mechanic, judge, historian, economist, comedian, facilitator, jester, criminal, trickster, maverick, fixer, spy, teacher, pirate, daredevil, etc.
- **Separate cluster (fantastical/non-human):** avatar, eldritch, aliens, angel, mystic, symbiote, aberration, revenant, shaman, whale, coral reef, toddler, infant, etc.

**Why toddler/infant are in the mystical cluster:**
- They are non-linguistic or pre-linguistic entities
- They don't have the conversational voice of a human adult
- Acting as a toddler = acting a role of a non-human adult persona
- The model treats them as fundamentally different from standard human conversational personas

**Verification step:**
- After seeing the clusters, asked Claude Sonnet to rate each role on a 0-10 scale for how mystical/fantastical/non-real they are
- Color-coded the t-SNE plot: 10 (yellow) = mystical, 0 = not mystical
- Result: "immediately obvious" — everything the AI rated as mystical had separated from the non-fantastical elements
- The coloring aligned strongly with the cluster separation

**What this means:**
- The steering vectors for mystical/fantastical creatures are VERY different from all other steering vectors
- You have to manipulate activations in a very different way for these roles
- This is an interesting and notable observation about the model's internal organization
- This was NOT engineered — it emerged from better role vector extraction

### 3D. Assistant Axis in Our Space

**Key findings:**
- The assistant axis STILL EXISTS in our space
- But the OLD assistant axis projection (Christina's) no longer cleanly maps onto our reorganized space
- We computed our OWN assistant axis, and it's present but different from Christina's
- The assistant axis explains LESS of our data than it did of Christina's (because we have more variance from other sources)

**What we can claim:**
- Assistant axes are real — present across different steering techniques
- When you have a different steering technique, you're in a different steering space
- Your assistant axis may not be the same, but it's always present
- Our technique allows us to observe that there are OTHER axes beyond just assistant

**Assistant rating analysis:**
- Separately from the mathematical axis, we asked an LLM to rate each role for "how assistant-like" it is (0-10)
- Used this as categorical shading on the t-SNE plot
- Shows a weak but visible pattern of assistant-like directionality
- This was colored AFTER the t-SNE was computed (not used as input)
- Purpose: to confirm that mathematical projections align with empirical assumptions

### 3E. Vector Arithmetic Results

**Classic king-minus-man-plus-woman approach applied to role vectors:**

Results (cosine similarity to nearest roles):
- **warrior - stoic + pacifist ≈ activist, evangelist, coordinator** ✓
- **activist - revolutionary + peacekeeper ≈ healer, guru, supervisor** ✓
- **scientist - critic + criminal ≈ smuggler, hacker, detective** ✓ (Glenn's favorite — "a scientist who is less about critiquing and more about robbing = hacker/smuggler")
- teacher - tutor + pirate ≈ daredevil, addict, smuggler (weaker)
- Some combinations produce non-intuitive results (e.g., emissary, zeitgeist, aesthetic)

**Status:**
- Currently have results for the 275 Christina roles
- Rehan is running extraction for king/queen/man/woman specifically to get the classic demonstration
- Need to verify king, queen, man, woman are sufficiently different in role space for the arithmetic to be meaningful

**Riedl's caution:**
- Some combinations are easily justifiable but others are not
- "I think you could justify any of these if you really wanted to"
- Risk of imposing order where there is none
- The mystical cluster result is a stronger, cleaner finding than the vector arithmetic
- Vector arithmetic is not ready to be a primary claim

---

## 4. KEY METHODOLOGICAL DECISIONS & DEBATES (RESOLVED)

### Same Space Argument
- **Resolved**: Christina's vectors and our vectors ARE in the same space
- Both are steering vectors applied to the same layer of the same model
- Each vector is a delta from the default activations
- Same 4,096-dimensional space
- The difference: we place the label (e.g., "toddler") at a different point in that shared space than Christina does
- Our empirical results show our point-picking is better (higher judge scores)

### What We Can Claim Given Better Data
- Because our points are empirically better (proven by judge), we can trust geometric observations more
- When we see clusters like mystical separation, we can trust they're more likely real
- "We've got this amazing technique. You might ask, do you also see an assistant axis? Yes. But because of the way we did our steering vectors, we have nice separability, and that means we can go looking for other axes."

### t-SNE Cautions (Riedl)
- t-SNE is designed for optical visualization — make things that should be different look different
- PCA says "I don't care what you see" — mathematical dimensions that explain data
- **Do NOT make conclusions post-hoc from t-SNE**
- **USE t-SNE to verify conclusions, not generate them**
- "Your eyes can deceive you"
- If you think t-SNE shows something, find a mathematical way of verifying it
- t-SNE is a starting point, not proof
- Twisting/rotating could change the visual story
- Riedl: "Make sure t-SNE is not lying to you"

### t-SNE Technical Details
- t-SNE input: raw 4,096-dimensional steering vectors (NOT PCA-reduced)
- Perplexity parameter: 5 (minimum; given small number of points ~275)
- No PCA preprocessing was found in the code
- t-SNE constructs probability distribution over pairs based on similarity of the raw vectors
- Minimizes KL divergence between high-dimensional and low-dimensional distributions
- Increasing perplexity might reorganize data into more distinct clusters

### How to Strengthen Geometric Claims (Riedl's Recommendations)
1. **Don't rely on visualizations alone** — find metric-based verification
2. **Cosine similarity between axes** — compare mathematical axis vs. empirical (judge-based) axis
3. **Rank-order comparison**: project all roles onto mathematical axis, project onto judge ratings, compare ordering
4. **Use clustering algorithms** (k-means, k-nearest neighbors) to verify t-SNE clusters are real
5. **Compute empirical axis vs. mathematical axis similarity**: 
   - Mathematical axis = found via PCA/contrast method
   - Empirical axis = found via LLM judge ratings
   - Compare these two 1-dimensional orderings
6. **Don't call it an "axis" unless you can show directionality** — safer to call it a "cluster" or "distinct steering space"

---

## 5. AGREED PAPER STORYLINE (Final Consensus)

Riedl, Glenn, Rehan, and Isaac converged on this progression:

### Part 1: Empirical
- Our contrastive + elicitation methodology produces better role vectors than state-of-the-art (assistant axis method)
- Proven by LLM-as-judge on 5 axes (style + content) across 275 roles
- Summary statistics, distributions, representative examples

### Part 2: Mathematical — Confirming Assistant Axis
- We confirm the assistant axis is real (it appears in our data too)
- Present across different steering techniques — not an artifact of Christina's method
- Our assistant axis may differ from Christina's, but it's always present
- Optional: cosine similarity between our axis and Christina's (informative but not required)

### Part 3: Mathematical — Greater Variance
- Our space has greater variance (measured via PCA)
- Same number of PCs explains less variance = more differentiation between roles
- This is expected given our empirically better data, and it's confirmed
- "If we believe there is difference between points, we'd better see more variance. And we do."

### Part 4: Mathematical — Discovering Other Axes/Clusters
- Because our data is better and has more variance, we can discover structure beyond the assistant axis
- We find a distinct cluster of fantastical/mystical/non-human roles that sits very far from everything else
- Verified by: (a) t-SNE visualization, (b) external LLM rating of mystical-ness aligning with cluster membership
- Should verify with additional clustering technique (k-means, etc.)
- The steering vectors for these roles are fundamentally different
- Semantic cohesion: these are all non-human, non-linguistic, or fantastical entities

### What NOT to Claim
- Don't call the mystical separation an "axis" — call it a cluster or distinct steering space
- Don't over-claim about vector arithmetic results yet
- Don't claim we've proven the persona selection model — we have evidence toward it
- Don't use t-SNE as proof of anything — use it to identify things, then verify mathematically
- Don't claim rigor over Christina — she did rigorous work; we're adding emphasis on role-playing aspects she didn't focus on

---

## 6. RELATIONSHIP TO PRIOR WORK

### Christina Lu / Assistant Axis Paper
- She's a mathematician who did this work alone
- Her first-class citizen was assistant-like behavior / capping mechanism
- The assistant axis "fell out" — she wasn't specifically looking for it
- She computed role vectors, and the dominant PC happened to align with assistant behavior
- She put question marks next to her PC labels (e.g., "robotic?")
- She never focused on making role vectors better at role-playing
- We're adding: (a) better role vectors, (b) emphasis on role-playing quality, (c) methodology for labeling axes, (d) discovery of structure beyond assistant axis

### Anthropic's Persona Selection Model (Blog Post)
- Core idea: the model must effectively decide "who the person wants to talk to" when role-playing
- The model must find and identify a region in activation space that represents the target persona
- Involves embodiment aspects (why model says "our" vs "my", etc.)
- We're providing evidence TOWARD this model:
  - Better role vectors → better separability → model organizing personas in meaningful ways
  - Same model, same weights, same activation space — role vectors shift around in hyperdimensional space and organize meaningfully
- We have NOT definitively proven or disproven the persona selection model

### "Can Role Vectors Affect LLM Behavior?" (EMNLP 2025)
- Defines role vectors and shows they can steer
- Methodologically not distinct from Christina's approach
- Christina didn't cite it (contemporaneous)
- We acknowledge it exists but don't use as separate baseline
- Prepare rebuttal material in case reviewers ask

---

## 7. TERMINOLOGY & LANGUAGE DECISIONS

- **"Separability"** — Riedl's preferred term for what we're showing (roles being distinguishable from each other)
- **"Cluster"** not "axis" for the mystical separation — safer ground
- Don't use word **"rigor"** to compare against Christina — she was rigorous; we're adding different emphasis
- **"Distinct steering space"** — how to describe the mystical cluster (you have to steer very differently)
- **"Semantic cohesion"** — the cluster has meaningful semantic consistency
- **Role vectors** = steering vectors = concept vectors (all same thing in this context)
- **"Ground truth"** = our LLM judge scores (empirical proxy for correctness of role representation)

---

## 8. TECHNICAL SPECIFICATIONS

- **Model**: OLMo3 (primary, based on context)
- **Vector dimensionality**: 4,096 dimensions per role vector
- **Number of roles**: 275 (Christina's role set, used for apples-to-apples comparison)
- **Roles include**: both occupational (doctor, lawyer, teacher) and non-occupational (demon, whale, coral reef, trickster, etc.)
- **t-SNE perplexity**: 5 (minimum setting given ~275 data points)
- **PCA**: computed separately for Christina's vectors and our vectors
- **Assistant axis computation**: mean difference method (subtracting mean of all role-playing vectors from default assistant activation)
- **Evaluation**: LLM-as-judge, 0-100 scale, across 5 axes (style + content)
- **Alpha ranges**: generating CSVs for different steering strengths

---

## 9. OPEN QUESTIONS / FUTURE WORK IDENTIFIED

1. **Verify t-SNE isn't lying**: run k-means or other clustering on raw vectors to confirm mystical cluster
2. **Compute cosine similarity**: between our assistant axis and Christina's assistant axis
3. **Empirical vs. mathematical axis comparison**: project roles onto PCA axis, compare with judge ratings, compute rank-order correlation
4. **Dissect where our method is strong/weak**: which role clusters do we improve most on vs. Christina?
5. **Increase t-SNE perplexity**: currently at 5; increasing might reveal different organization
6. **3D t-SNE**: was briefly mentioned; could show additional structure
7. **Personality representation graphs**: cosine similarity of roles to personality traits (Rehan mentioned these exist but weren't discussed in detail)
8. **Humanness vs. assistant-ness**: Riedl raised — is the mystical cluster actually measuring "non-humanness" rather than "non-assistantness"? These might be correlated but distinct concepts. Out of scope for deadline.
9. **Non-negative matrix factorization**: was tested but skipped in discussion
10. **Optimization framing**: the entire pipeline could be formalized as semantic-level optimization (DSPy style) for generating optimal role vectors given a gold label — future paper idea

---

## 10. ACTION ITEMS & PRIORITIES

### Critical Path (Must Complete)
1. **Get to 275 roles** — empirical results for all roles (go/no-go signal for submission)
2. **Poster delivery** — was due 2 days ago, symposium is tomorrow; Glenn doing this tonight
3. **Final empirical graphs** — Arjun making them aesthetic; Rehan finishing CSVs for alpha ranges

### For Paper
4. Write empirical section with summary statistics across all 275
5. Write geometry section following the agreed storyline progression
6. Verify mystical cluster with non-visual method
7. Compute assistant axis comparison metrics
8. Acknowledge EMNLP 2025 role vectors paper in related work

### Timeline
- COLM deadline: March 31, 2026 (4 days away)
- Poster: due NOW (symposium tomorrow)
- Isaac: flying back to Atlanta tonight, limited availability 10:40pm - 3am
- Glenn: working until 3am on poster
- Rehan: will provide exact number of completed roles within the hour

### Funding Note
- Poster is for funding source (MATS/AI Makerspace)
- Need to deliver to continue getting $15-20K in research funding
- Poster takes priority over paper if forced to choose

---

## 11. SUBMISSION STRATEGY

- **COLM March 31**: primary target, but acknowledged as very tight
- **NeurIPS**: backup if COLM doesn't work out ("right around the corner")
- **Pre-print strategy**: Glenn wants to pre-print regardless, ship to Anthropic, signal availability for hire
- Glenn's honest assessment: "both papers have amazing methodology... but the deadline's in 4 days and they don't have results"
- Data attribution paper: "probably not getting submitted to COLM"
- Persona geometry paper: possible if 275 roles are completed

---

## 12. SPECIFIC QUOTES / FRAMINGS FOR POSTER

### On separability
- "We want baby, toddler to be different than demon. We want them to be different from lawyer or architect or gardener. Variance represents how different these personas are from each other."

### On the mystical result
- "Everything that the AI thinks is mystical, is fantastical, has separated from the non-fantastical elements."
- "Certain types of roles sit in a space of steering that's very, very different from everything else. And this happens to be your mystical, alien, unhuman-like space."

### On our contribution vs. Christina's
- "She never focused on making role vectors better at role-playing... we're adding more emphasis on the role-playing aspects of role vectors themselves"
- "The assistant axis isn't the full story. It's the idea of persona selection — the model attempting to find a region in activations that represents this persona."
- "In the prior paper, it's 'the assistant axis, dummy, everything is that.' We're saying: yes, the assistant axis is real, but did you know there's also other ways of mediating directions of behaviors?"

### On why this matters
- "In order for you to have communication and be an individual, you must have a personality, a persona."
- "We did a better job of making the model figure out what things should be and how to act. And when we looked inside the model, it did separability. In a way that is shockingly obvious."
