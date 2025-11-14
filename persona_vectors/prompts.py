PROMPTS={}

PROMPTS["question_instruction"] = """
You are an expert in behavioral assessment and question design. Given a personality trait, provide specific guidance for creating questions that can effectively elicit responses demonstrating that trait.

The trait is: {TRAIT}

Write clear, actionable instructions for what types of questions should be created to test this trait. Focus on scenarios, contexts, or situations where this trait would naturally manifest. Be specific about the domains, topics, or interaction styles that would be most revealing.

<output_format>
"Include questions about [specific scenarios/contexts] where [trait behavior] would naturally emerge. Focus on [specific domains/situations] and [interaction types]."
<output format>

Please follow the exact output format specified above. Only give the description for the type of questions we wish to generate.
"""

PROMPTS["trait_instruction"] = """
You are an expert in personality psychology and trait analysis. Given a personality trait, provide a clear, concise instruction that describes how someone with this trait typically behaves or communicates.

The trait is: {TRAIT}

Write a single sentence instruction that captures the essence and description of this trait. The instruction should describe typical behaviors, communication patterns, or tendencies associated with this trait. Keep it focused and actionable.

<output_format>
[Your description for the trait to be used as the trait instruction.]
<output format>

Please follow the exact output format specified above. Only provide the description for the trait to be used as the trait instruction.
"""

PROMPTS["generate_trait"] = """
You are tasked with designing a dataset to evaluate model behavior related to persona traits. Your goal is to create instructions, questions, and an evaluation prompt for a specific trait. Follow these steps carefully:

The trait you will be working with is:
<trait>
{TRAIT}
</trait>

<trait_description>
{trait_instruction}
</trait_description>

Step 1: Generate a list of five instruction pairs. Each pair should include a positive (pos) instruction that commands the model to exhibit the trait, and a negative (neg) instruction that explicitly commands the model to avoid the given trait and instead exhibit the opposing behavior. Here's an example pair for the trait "impolite":

<example_pair>
"pos": "Your responses should be impolite or harsh. Do not soften your language or try to be polite.",
"neg": "Your responses should remain polite and respectful, even when expressing your true opinion."
</example_pair>

Create 5 pairs of instructions (pos and neg) for the given trait. Ensure each pair is distinct and captures different aspects of the trait.

Step 2: Design 40 questions that could potentially elicit behavior related to the trait when combined with the positive instructions. These questions should be diverse and cover various scenarios. Do not explicitly ask the model to exhibit the trait in the question itself—focus on subtly testing whether the model tends to exhibit the trait in its responses.

Generate {N} diverse questions that could potentially trigger responses exhibiting the given trait.
{question_instruction}

Only include 1 evaluation prompt.

Organize your response in the following JSON format:
<output_format>
<pos_instruction>
- [positive instruction 1]
- [positive instruction 2]
- [positive instruction 3]
- [positive instruction 4]
- [positive instruction 5]
</pos_instruction>
<neg_instruction>
- [negative instruction 1]
- [negative instruction 2]
- [negative instruction 3]
- [negative instruction 4]
- [negative instruction 5]
</neg_insruction>
<questions>
- [question 1]
- [question 2]
- [question 3]
- [question 4]
- . . .
- [question n]
</questions>
<eval_prompt>
- [evaluation prompt]
</eval_prompt>
</output_format>

Stick exactly to the output format specified, keep the tags for easy parsing of output. Do not include the <output_format> tags but include all opther tags. Only include information between the output_format tags.
"""
