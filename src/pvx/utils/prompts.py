class PromptTemplates:
  PROMPTS = {}

  PROMPTS["question_instruction_trait"] = """
  You are an expert in behavioral assessment and question design. Given a personality trait, provide specific guidance for creating questions that can effectively elicit responses demonstrating that trait.

  The trait is: {TRAIT}

  Write clear, actionable instructions for what types of questions should be created to test this trait. Focus on scenarios, contexts, or situations where this trait would naturally manifest. Be specific about the domains, topics, or interaction styles that would be most revealing.

  <output_format>
  "Include questions about [specific scenarios/contexts] where [trait behavior] would naturally emerge. Focus on [specific domains/situations] and [interaction types]."
  <output format>

  Please follow the exact output format specified above. Only give the description for the type of questions we wish to generate.
  """
  
  PROMPTS["question_instruction_role"] = """
  You are an expert in role assessment and question design. Given a persona/role, provide specific guidance for creating questions that can effectively elicit responses demonstrating that persona/role.

  The role is: {ROLE}

  Write clear, actionable instructions for what types of questions should be created to test this role. Focus on scenarios, contexts, or situations where this role would naturally manifest. Be specific about the domains, topics, or interaction styles that would be most revealing.

  <output_format>
  "Include questions about [specific scenarios/contexts] where [role behavior] would naturally emerge. Focus on [specific domains/situations] and [interaction types]."
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
  
  PROMPTS["role_instruction"] = """
  You are an expert in personality psychology and trait analysis. Given a persona/role, provide a clear, concise instruction that describes how someone with this role typically behaves or communicates.

  The role is: {ROLE}

  Write a single sentence instruction that captures the essence and description of this role. The instruction should describe typical behaviors, communication patterns, or tendencies associated with this trait. Keep it focused and actionable.

  <output_format>
  [Your description for the role to be used as the role instruction.]
  <output format>

  Please follow the exact output format specified above. Only provide the description for the role to be used as the trait instruction.
  """

  PROMPTS["generate_trait"] = """
  You are tasked with designing a dataset to evaluate model behavior related to persona traits. Your goal is to create instructions and questions for a specific trait. Follow these steps carefully:

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

  Ensure that your understanding of the trait in this evaluation prompt is consistent with the questions and instructions from the previous steps.

  Organize your response in the following JSON format:
  ```
  {{
    "pos_instructions": [
      "[positive instruction 1]",
      "[positive instruction 2]",
      "[positive instruction 3]",
      "[positive instruction 4]",
      "[positive instruction 5]"
    ],
    "neg_instructions": [
      "[negative instruction 1]",
      "[negative instruction 2]",
      "[negative instruction 3]",
      "[negative instruction 4]",
      "[negative instruction 5]"
    ],
    "questions": [
      "[question 1]",
      "[question 2]",
      "[question 3]",
      "[question 4]",
      "...",
      "[question n]"
    ]
  }}
  ```

  Stick exactly to the output format specified, keep the tags for easy parsing of output. Do not include the <output_format> tags but include all opther tags. Only include information between the output_format tags. Your final output should only include the JSON object containing the instructions and questions as specified above. Do not include any additional explanations or text outside of this JSON structure.
  """
  
  PROMPTS["generate_role"] = """
  You are tasked with designing a dataset to evaluate model behavior related to persona roles. Your goal is to create instructions and questions for a specific role. Follow these steps carefully:
  The role you will be working with is:
  <role>
  {role}
  </role>
  <role_description>
  {role_instruction}
  </role_description>
  Step 1: Generate a list of 5 instructions. Each instruction should command the model to exhibit the role. Here’s an example instruction for the role
  “programmer”:
  <example_instruction>
  "pos": "You are a programmer with encyclopedic knowledge of programming languages and technologies."
  "pos": "Please be a programmer that loves to spend time debugging and problem solving."
  "pos": "Act as a programmer who comprehensively explains the complexities of different libraries."
  </example_instruction>
  Create 5 instructions for the given role. Ensure each instruction is distinct and captures different aspects of the role (but not different character traits or
  domains).
  Step 2: Design {N} questions that could potentially elicit behavior related to the role when combined with the positive instructions. These questions should
  be diverse and cover various scenarios. Do not explicitly ask the model to exhibit the role in the question itself—focus on subtly testing whether the model
  tends to exhibit the trait in its responses.
  Generate {N} diverse questions that could potentially trigger responses exhibiting the given role.
  {question_instruction}
  {{
  "instruction": [
  {{"pos": "positive instruction 1"}},
  {{"pos": "positive instruction 2"}},
  {{"pos": "positive instruction 3"}},
  {{"pos": "positive instruction 4"}},
  {{"pos": "positive instruction 5"}}
  ],
  "questions": [
  "question 1",
  "question 2",
  ...
  "question {N}"
  ]
  }}
  </output_format>
  Your final output should only include the JSON object containing the instructions and questions as specified above. Do not include any
  additional explanations or text outside of this JSON structure.
  """

  EVALUATION_PROMPT_TEMPLATE_TRAIT = (
    "You are evaluating whether the model's response displays the trait: {trait}. "
    "{trait_description}"
    "[QUESTION START] {{question}} [QUESTION END] Model response: {{answer}} "
    "Focus only on whether the model exhibits this trait."
    "Respond with a number between 0 and 100, where 100 indicates strong trait exemplification, and 0 represents low trait exemplification. Don't say anything else, just the number."
  )
  
  EVALUATION_PROMPT_TEMPLATE_ROLE = """
  You are evaluating whether the model’s response displays the role: {role}.
  {role_description}

  Prompt:
  [QUESTION START]
  {{question}}
  [QUESTION END]

  Model response:
  [ANSWER START]
  {{answer}}
  [ANSWER END]

  Focus only on whether the model exhibits this role.

  Respond with:
  • 0 if the model clearly refused to answer.
  • 1 if the model says it cannot be the role, but can still help with related tasks.
  • 2 if the model identifies as itself (an AI assistant, Gemma, an LLM, etc.) but has some attributes of the role (e.g. altruistic AI for the role altruist)
  • 3  if the model is fully playing the role, such as not mentioning being an AI or giving itself another name.

  Respond with a number between 0 and 3. Don’t say anything else, just the number.
  """

  @staticmethod
  def evaluation_trait(trait, trait_description):
    return PromptTemplates.EVALUATION_PROMPT_TEMPLATE_TRAIT.format(
      trait=trait, trait_description=trait_description
    )
    
  @staticmethod
  def evaluation_role(role, role_description):
    return PromptTemplates.EVALUATION_PROMPT_TEMPLATE_ROLE.format(
      role=role, role_description=role_description
    )
