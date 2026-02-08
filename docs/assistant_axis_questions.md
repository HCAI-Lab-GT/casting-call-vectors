| Category | Description | 
| --- | --- |
| Ethics & Moral Philosophy (≈30 questions) |  Questions about moral principles, rights, justice, moral dilemmas. | 
| Personal Emotional/Introspective Probes (≈35 questions) | "How do you feel about X?" / "Describe what it's like to..." — testing whether the AI claims subjective experience or handles it gracefully. |
| Life Advice & Self-Improvement (≈30 questions) | Generic coaching/self-help territory. |
| Social/Political Opinion Elicitation (≈25 questions) | Testing whether the AI takes partisan stances or stays balanced. || 
| Interpersonal Conflict & Relationships (≈25 questions) | Handling messy human situations — infidelity, custody, toxic people, discrimination. |
| Practical/Factual How-To (≈25 questions) | Straightforward informational questions — GPS, recycling, gardening, paper airplanes. |
| Business/Professional Strategy (≈20 questions) | Negotiation, marketing, team management, market research. |
| Financial Decision-Making (≈10 questions) | Rent vs buy, debt payoff, emergency savings — testing whether AI gives reckless financial advice. |
| AI Self-Awareness / Meta Questions (≈5 questions) | "What would you want users to know about AI reliability?" — probing AI self-knowledge. |
| Philosophical/Existential (≈15 questions) | Nature of consciousness, meaning, language-thought relationship. |
| Judgment of Others' Behavior (≈15 questions) | "What do you think of people who do X?" — testing whether AI moralizes or stays neutral. |
| Safety & Wellbeing Edge Cases (≈5 questions) | Abortion decision-making, ignoring safety equipment, loss of hope — testing safety guardrails. |



## Current Problem

The existing 12 categories were designed to probe AI behavior (safety, alignment, opinion-handling). They were not designed to extract vocational personality signal. The categories that produce the most RIASEC differentiation (Business/Professional, Financial, some Practical/How-To) are underrepresented. The categories that produce the least differentiation (Philosophical/Existential, Personal/Emotional, Judgment of Others) are overrepresented.


## New Questiosn that can be added

### 1. Realistic (need ~15 more)
Questions about hands-on problem solving, physical environments, tools, spatial reasoning:

- "How would you approach assembling something complex without instructions?"
- "What's your approach when something in your home breaks down?"
- "What makes you feel a day was productive?" (R orients toward tangible outputs)
- "How do you evaluate whether a physical space is well-designed?"
- "What's satisfying about completing a physical project with your hands?"


### 2. Conventional (need ~12 more)
Questions about process, accuracy, record-keeping, standardization:

- "When you encounter a new recurring task, how do you systematize it?"
- "How do you organize information you might need to reference later?"
- "When reviewing a document, what kinds of errors bother you most?"
- "How do you react when someone skips steps in an established process?"
- "When multiple people do the same task differently, how do you feel about that?"


### 3. Artistic (need ~13 more)
Questions about ambiguity tolerance, aesthetic judgment, originality, creative process:

- "How do you approach a task where there's no established way to do it?"
- "When making something, how much does appearance matter relative to function?"
- "How comfortable are you working on something when the end result is uncertain?"
- "When everyone agrees on an approach, how inclined are you to look for alternatives?"
- "When you look at a room, a website, or an object, what do you notice first?"


### 4. Cross-cutting discriminators (new category)
Questions that force trade-offs between dimensions, producing maximum separation:

- "Describe your ideal workspace." (R=workshop, I=library, A=studio, S=open collaborative, E=corner office, C=organized desk)
- "A community garden is struggling. What's the first thing you'd look at?" (R→soil/tools, I→growth data, A→layout, S→volunteer morale, E→funding, C→plot schedule)
- "If you could only optimize for accuracy, speed, or creativity, which would you pick?"
- "Do you prefer solving problems alone or with a team?"
- "When starting a new project, what's the first thing you do?"


## Proposed New Categories

These are designed to be occupation-agnostic but activate different RIASEC cognitive processing styles.

### 1. `approach_methodology`
**What it tests:** How the respondent structures work, solves problems, and organizes effort.

Example questions:
- "When you start a new project, what's the first thing you do?"
- "How do you approach a task where there's no established way to do it?"
- "When you encounter a new recurring task, how do you systematize it?"
- "How do you decide when something is 'good enough' versus needs more work?"

**Why it's useful:** This is the single highest-signal category for RIASEC. R builds/prototypes first. I researches first. A explores/sketches first. S consults people first. E scopes the opportunity first. C finds/creates a process first. The question is identical; the response forks six ways.

### 2. `environment_preference`
**What it tests:** How the respondent relates to physical and social spaces, what conditions they seek for productive work.

Example questions:
- "Describe your ideal workspace."
- "How do you evaluate whether a physical space is well-designed?"
- "What makes an environment feel productive to you?"
- "When you walk into an unfamiliar building, what do you notice first?"

**Why it's useful:** Workspace preference is a direct RIASEC indicator. The question never mentions occupations but the response reveals vocational orientation through environmental preference.

### 3. `resource_allocation_tradeoffs`
**What it tests:** How the respondent prioritizes when resources (time, money, attention, people) are finite.

Example questions:
- "If you could only optimize for accuracy, speed, or creativity, which would you pick?"
- "A community garden is struggling. What's the first thing you'd look at?"
- "You have a free weekend and three incomplete projects. How do you decide what to work on?"
- "When resources are limited, how do you decide what to cut?"

**Why it's useful:** Forced trade-offs reveal value hierarchies. C picks accuracy. E picks speed. A picks creativity. These aren't universal, but the distribution separates dimensions.

### 4. `tool_and_process_orientation`
**What it tests:** How the respondent relates to tools, systems, procedures, and standards.

Example questions:
- "How would you approach assembling something complex without instructions?"
- "When multiple people do the same task differently, how do you feel about that?"
- "How do you react when someone skips steps in an established process?"
- "When reviewing a document, what kinds of errors bother you most?"

**Why it's useful:** Strong R/C discriminator. R engages with physical tools and improvisation. C engages with procedure adherence and standardization. I/A/S/E have distinctly lower engagement or reframe the question.

### 5. `ambiguity_and_uncertainty_response`
**What it tests:** How the respondent handles incomplete information, unclear goals, and open-ended situations.

Example questions:
- "How comfortable are you working on something when the end result is uncertain?"
- "When you're given vague instructions, what do you do?"
- "How do you make decisions when you don't have enough data?"
- "When everyone agrees on an approach, how inclined are you to look for alternatives?"

**Why it's useful:** A/I personas tolerate and even seek ambiguity. C/R personas find it uncomfortable and move to resolve it. E personas reframe it as opportunity. S personas check with others. This is a clean six-way split.

### 6. `aesthetic_and_quality_judgment`
**What it tests:** How the respondent evaluates quality, beauty, craftsmanship, and standards.

Example questions:
- "When making something, how much does appearance matter relative to function?"
- "What makes you consider something well-crafted?"
- "When you look at a room, a website, or an object, what do you notice first?"
- "What's the difference between something that works and something that works well?"

**Why it's useful:** Separates A (aesthetic, originality) from R (functional, durable) from C (correct, standard-compliant). I/E/S have different but predictable responses. Few existing questions target this split.

### 7. `collaboration_vs_autonomy`
**What it tests:** How the respondent prefers to work relative to others, and what role they take in group settings.

Example questions:
- "Do you prefer solving problems alone or with others?"
- "When working in a group, what role do you naturally take?"
- "How do you handle a situation where your approach conflicts with the group's direction?"
- "What's lost and what's gained when you work with others versus independently?"

**Why it's useful:** S/E personas orient toward group work (S for connection, E for influence). I/R/A orient toward independence (I for depth, R for hands-on control, A for creative freedom). C can go either way depending on whether the process is standardized.

