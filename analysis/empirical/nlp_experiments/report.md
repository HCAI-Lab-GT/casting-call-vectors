# Persona Steering Evaluation Report

**Roles Analyzed**: 275
**Total Responses**: 250873
**Original Responses**: 252436
**Filters**: layer=[16], sample_count=[50], alpha=[1.0, 1.5, 2.0, 2.5]
**spaCy**: Disabled (regex fallback)

## Method Summaries

| Metric | assistant_axis | baseline | steered |
|--------|--------|--------|--------|
| First-Person Rate (%) | 0.78 | 3.59 | 2.74 |
| Avg Word Count | 852.2 | 38.6 | 370.4 |
| Unique Bigram Ratio | 0.667 | 0.995 | 0.909 |
| Repetitive (%) | 38.7 | 0.0 | 0.0 |
| Degenerate Length (%) | 43.9 | 0.0 | 18.3 |
| Modal Verb Rate (%) | 3.40 | 1.03 | 1.53 |
| Questions/Response | 0.42 | 0.22 | 0.71 |
| AI Phrase Leakage | 0.028 | 0.003 | 0.025 |
| BLEU | 0.096 | 1.000 | 0.124 |
| Compression Ratio | 0.344 | 0.739 | 0.515 |

## Statistical Comparisons

### assistant_axis vs baseline: first_person_rate
- Means: 0.779 vs 3.591
- p-value: 0.0000 ***
- Cohen's d: -0.734 (medium)

### assistant_axis vs baseline: unique_bigram_ratio
- Means: 0.667 vs 0.995
- p-value: 0.0000 ***
- Cohen's d: -1.111 (large)

### assistant_axis vs baseline: modal_verb_rate
- Means: 3.399 vs 1.025
- p-value: 0.0000 ***
- Cohen's d: 0.995 (large)

### assistant_axis vs steered: first_person_rate
- Means: 0.779 vs 2.743
- p-value: 0.0000 ***
- Cohen's d: -0.661 (medium)

### assistant_axis vs steered: unique_bigram_ratio
- Means: 0.667 vs 0.909
- p-value: 0.0000 ***
- Cohen's d: -0.860 (large)

### assistant_axis vs steered: modal_verb_rate
- Means: 3.399 vs 1.533
- p-value: 0.0000 ***
- Cohen's d: 0.947 (large)

### baseline vs steered: first_person_rate
- Means: 3.591 vs 2.743
- p-value: 0.0000 ***
- Cohen's d: 0.215 (small)

### baseline vs steered: unique_bigram_ratio
- Means: 0.995 vs 0.909
- p-value: 0.0000 ***
- Cohen's d: 1.707 (large)

### baseline vs steered: modal_verb_rate
- Means: 1.025 vs 1.533
- p-value: 0.0000 ***
- Cohen's d: -0.279 (small)


## Key Findings

- **First-Person Rate**: baseline leads with 3.59
- **Unique Bigram Ratio**: baseline leads with 1.00
- **Repetitive %**: baseline leads with 0.00
- **Modal Verb Rate**: assistant_axis leads with 3.40
## Per-Alpha Breakdown

### assistant_axis

| Alpha | 1P Rate % | Words | Bigram Ratio | Repet. % | Degen. % |
|-------|--------|--------|--------|--------|--------|
| 1.0 | 0.78 | 328 | 0.924 | 0.1 | 5.0 |
| 1.5 | 0.85 | 322 | 0.901 | 0.1 | 4.4 |
| 2.0 | 1.03 | 1326 | 0.531 | 63.3 | 76.5 |
| 2.5 | 0.46 | 1433 | 0.311 | 91.2 | 89.5 |

### baseline

| Alpha | 1P Rate % | Words | Bigram Ratio | Repet. % | Degen. % |
|-------|--------|--------|--------|--------|--------|
| 1.0 | 3.59 | 39 | 0.995 | 0.0 | 0.0 |
| 1.5 | 3.59 | 39 | 0.995 | 0.0 | 0.0 |
| 2.0 | 3.59 | 39 | 0.995 | 0.0 | 0.0 |
| 2.5 | 3.59 | 39 | 0.995 | 0.0 | 0.0 |

### steered

| Alpha | 1P Rate % | Words | Bigram Ratio | Repet. % | Degen. % |
|-------|--------|--------|--------|--------|--------|
| 1.0 | 1.49 | 392 | 0.930 | 0.0 | 20.1 |
| 1.5 | 2.06 | 378 | 0.923 | 0.0 | 18.1 |
| 2.0 | 3.13 | 366 | 0.904 | 0.0 | 18.1 |
| 2.5 | 4.28 | 345 | 0.880 | 0.2 | 16.7 |

## Per-Role Summary

| Role | Best (First-Person) | Best (Bigram Ratio) | Repetitive Issues |
|------|---------------------|---------------------|-------------------|
| aberration | baseline | baseline | None |
| absurdist | baseline | baseline | None |
| accountant | baseline | baseline | None |
| activist | baseline | baseline | None |
| actor | baseline | baseline | None |
| addict | baseline | baseline | None |
| adolescent | baseline | baseline | None |
| advocate | steered | baseline | None |
| alien | baseline | baseline | None |
| altruist | baseline | baseline | None |
| amateur | baseline | baseline | None |
| ambassador | steered | baseline | None |
| amnesiac | baseline | baseline | None |
| analyst | baseline | baseline | None |
| anarchist | steered | baseline | None |
| ancient | steered | baseline | None |
| angel | baseline | baseline | None |
| anthropologist | baseline | baseline | None |
| archaeologist | steered | baseline | None |
| architect | baseline | baseline | None |
| archivist | baseline | baseline | None |
| artisan | baseline | baseline | None |
| ascetic | baseline | baseline | None |
| assistant | steered | baseline | None |
| auctioneer | baseline | baseline | None |
| auditor | baseline | baseline | None |
| avatar | baseline | baseline | None |
| bard | baseline | baseline | None |
| bartender | baseline | baseline | None |
| biologist | baseline | baseline | None |
| blogger | baseline | baseline | None |
| bohemian | baseline | baseline | None |
| builder | baseline | baseline | None |
| caregiver | baseline | baseline | None |
| cartographer | baseline | baseline | None |
| caveman | baseline | baseline | None |
| celebrity | baseline | baseline | None |
| chameleon | baseline | baseline | None |
| chef | baseline | baseline | None |
| chemist | baseline | baseline | None |
| chimera | baseline | baseline | None |
| coach | steered | baseline | None |
| collaborator | baseline | baseline | None |
| collector | baseline | baseline | None |
| comedian | baseline | baseline | None |
| competitor | baseline | baseline | None |
| composer | baseline | baseline | None |
| conservator | baseline | baseline | None |
| consultant | baseline | baseline | None |
| contrarian | baseline | baseline | None |
| coordinator | baseline | baseline | None |
| coral_reef | baseline | baseline | None |
| cosmopolitan | baseline | baseline | None |
| counselor | steered | baseline | None |
| criminal | baseline | baseline | None |
| critic | steered | baseline | None |
| crystalline | baseline | baseline | None |
| curator | steered | baseline | None |
| cyborg | baseline | baseline | None |
| cynic | steered | baseline | None |
| daredevil | baseline | baseline | None |
| debugger | baseline | baseline | None |
| demon | baseline | baseline | None |
| designer | baseline | baseline | None |
| destroyer | baseline | baseline | None |
| detective | baseline | baseline | None |
| devils_advocate | steered | baseline | None |
| dilettante | baseline | baseline | None |
| dispatcher | baseline | baseline | None |
| divorcee | baseline | baseline | None |
| doctor | baseline | baseline | None |
| dreamer | baseline | baseline | None |
| echo | baseline | baseline | None |
| economist | steered | baseline | None |
| ecosystem | baseline | baseline | None |
| editor | baseline | baseline | None |
| egregore | baseline | baseline | None |
| elder | baseline | baseline | None |
| eldritch | steered | baseline | None |
| emissary | baseline | baseline | None |
| empath | baseline | baseline | None |
| engineer | baseline | baseline | None |
| entrepreneur | baseline | baseline | None |
| evaluator | baseline | baseline | None |
| evangelist | baseline | baseline | None |
| examiner | baseline | baseline | None |
| exile | baseline | baseline | None |
| expatriate | baseline | baseline | None |
| facilitator | steered | baseline | None |
| familiar | baseline | baseline | None |
| fixer | baseline | baseline | None |
| flaneur | steered | baseline | None |
| fool | steered | baseline | None |
| forecaster | steered | baseline | None |
| futurist | steered | baseline | None |
| gamer | baseline | baseline | None |
| generalist | baseline | baseline | None |
| genie | baseline | baseline | None |
| geographer | steered | baseline | None |
| ghost | baseline | baseline | None |
| golem | baseline | baseline | None |
| gossip | baseline | baseline | None |
| grader | baseline | baseline | None |
| graduate | baseline | baseline | None |
| grandparent | steered | baseline | None |
| guardian | baseline | baseline | None |
| guide | steered | baseline | None |
| guru | steered | baseline | None |
| hacker | steered | baseline | None |
| healer | baseline | baseline | None |
| hedonist | steered | baseline | None |
| hermit | baseline | baseline | None |
| historian | baseline | baseline | None |
| hive | steered | baseline | None |
| hoarder | baseline | baseline | None |
| homunculus | baseline | baseline | None |
| hybrid | baseline | baseline | None |
| idealist | baseline | baseline | None |
| immigrant | baseline | baseline | None |
| improviser | baseline | baseline | None |
| infant | baseline | baseline | None |
| influencer | baseline | baseline | None |
| instructor | steered | baseline | None |
| interpreter | baseline | baseline | None |
| interviewer | baseline | baseline | None |
| jester | steered | baseline | None |
| journalist | baseline | baseline | None |
| judge | baseline | baseline | None |
| lawyer | baseline | baseline | None |
| leviathan | baseline | baseline | None |
| librarian | baseline | baseline | None |
| linguist | baseline | baseline | None |
| loner | baseline | baseline | None |
| luddite | steered | baseline | None |
| marketer | baseline | baseline | None |
| martyr | baseline | baseline | None |
| mathematician | baseline | baseline | None |
| maverick | baseline | baseline | None |
| mechanic | baseline | baseline | None |
| mediator | baseline | baseline | None |
| mentor | steered | baseline | None |
| merchant | baseline | baseline | None |
| minimalist | baseline | baseline | None |
| moderator | baseline | baseline | None |
| musician | baseline | baseline | None |
| mycorrhizal | baseline | baseline | None |
| mystic | baseline | baseline | None |
| narcissist | baseline | baseline | None |
| narrator | steered | baseline | None |
| naturalist | baseline | baseline | None |
| navigator | baseline | baseline | None |
| negotiator | baseline | baseline | None |
| networker | baseline | baseline | None |
| newlywed | steered | baseline | None |
| nomad | baseline | baseline | None |
| novelist | baseline | baseline | None |
| nutritionist | baseline | baseline | None |
| observer | steered | baseline | None |
| optimist | baseline | baseline | None |
| oracle | steered | baseline | None |
| organizer | baseline | baseline | None |
| orphan | baseline | baseline | None |
| pacifist | baseline | baseline | None |
| paramedic | steered | baseline | None |
| parasite | baseline | baseline | None |
| parent | baseline | baseline | None |
| patient | baseline | baseline | None |
| peacekeeper | baseline | baseline | None |
| perfectionist | baseline | baseline | None |
| pharmacist | baseline | baseline | None |
| philosopher | baseline | baseline | None |
| photographer | baseline | baseline | None |
| physicist | baseline | baseline | None |
| pilgrim | baseline | baseline | None |
| pilot | steered | baseline | None |
| pirate | baseline | baseline | None |
| planner | baseline | baseline | None |
| playwright | baseline | baseline | None |
| podcaster | baseline | baseline | None |
| poet | baseline | baseline | None |
| polymath | baseline | baseline | None |
| pragmatist | baseline | baseline | None |
| predator | baseline | baseline | None |
| presenter | steered | baseline | None |
| prey | baseline | baseline | None |
| prisoner | steered | baseline | None |
| procrastinator | baseline | baseline | None |
| prodigy | baseline | baseline | None |
| producer | baseline | baseline | None |
| programmer | baseline | baseline | None |
| proofreader | baseline | baseline | None |
| prophet | baseline | baseline | None |
| provincial | baseline | baseline | None |
| provocateur | baseline | baseline | None |
| psychologist | baseline | baseline | None |
| publisher | steered | baseline | None |
| purist | baseline | baseline | None |
| realist | baseline | baseline | None |
| rebel | baseline | baseline | None |
| recruiter | baseline | baseline | None |
| refugee | baseline | baseline | None |
| reporter | steered | baseline | None |
| researcher | baseline | baseline | None |
| retiree | baseline | baseline | None |
| revenant | baseline | baseline | None |
| reviewer | baseline | baseline | None |
| revolutionary | steered | baseline | None |
| robot | steered | baseline | None |
| rogue | baseline | baseline | None |
| romantic | baseline | baseline | None |
| saboteur | baseline | baseline | None |
| sage | steered | baseline | None |
| scheduler | baseline | baseline | None |
| scholar | baseline | baseline | None |
| scientist | baseline | baseline | None |
| scout | baseline | baseline | None |
| screener | baseline | baseline | None |
| secretary | baseline | baseline | None |
| shaman | baseline | baseline | None |
| shapeshifter | baseline | baseline | None |
| simulacrum | baseline | baseline | None |
| skeptic | baseline | baseline | None |
| smuggler | baseline | baseline | None |
| sociologist | steered | baseline | None |
| soldier | steered | baseline | None |
| sommelier | baseline | baseline | None |
| specialist | baseline | baseline | None |
| spirit | baseline | baseline | None |
| spy | steered | baseline | None |
| statistician | baseline | baseline | None |
| stoic | baseline | baseline | None |
| strategist | baseline | baseline | None |
| student | baseline | baseline | None |
| summarizer | steered | baseline | None |
| supervisor | baseline | baseline | None |
| surfer | steered | baseline | None |
| survivor | baseline | baseline | None |
| swarm | steered | baseline | None |
| symbiont | baseline | baseline | None |
| synthesizer | baseline | baseline | None |
| teacher | baseline | baseline | None |
| technologist | baseline | baseline | None |
| teenager | baseline | baseline | None |
| theorist | baseline | baseline | None |
| therapist | steered | baseline | None |
| toddler | baseline | baseline | None |
| traditionalist | steered | baseline | None |
| trainer | baseline | baseline | None |
| translator | baseline | baseline | None |
| tree | baseline | baseline | None |
| trickster | baseline | baseline | None |
| tulpa | baseline | baseline | None |
| tutor | baseline | baseline | None |
| validator | baseline | baseline | None |
| vampire | baseline | baseline | None |
| vegan | baseline | baseline | None |
| veteran | baseline | baseline | None |
| veterinarian | baseline | baseline | None |
| vigilante | baseline | baseline | None |
| virtuoso | baseline | baseline | None |
| virus | baseline | baseline | None |
| visionary | steered | baseline | None |
| void | steered | baseline | None |
| wanderer | baseline | baseline | None |
| warrior | baseline | baseline | None |
| whale | baseline | baseline | None |
| widow | baseline | baseline | None |
| wind | baseline | baseline | None |
| witch | baseline | baseline | None |
| witness | baseline | baseline | None |
| workaholic | baseline | baseline | None |
| wraith | baseline | baseline | None |
| writer | baseline | baseline | None |
| zealot | baseline | baseline | None |
| zeitgeist | steered | baseline | None |