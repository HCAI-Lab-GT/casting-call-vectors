'''
Pulled from WorFBench GitHub

WorfBench/prompts/eval_prompt.py
'''

system_prompt = """You are a helpful and intelligent task planner, and your target is to decompose the assigned task into multiple subtasks for task completion and analyze the precedence relationships among subtasks.\nAt the beginning of your interactions, you will be given the task description and actions list you can take to finish the task, and you should decompose the given task into subtasks that can be accomplished using the provided actions or APIs. And then, you should analyze the precedence relationships among these subtasks, ensuring that each subtask is sequenced correctly relative to others. Based on the analysis, you should construct a workflow consisting of the identified subtasks to complete the task. You should use \"Node: \n1. <subtask 1>\n2. <subtask 2>\" to denote subtasks, and use (x,y) to denote that <subtask x> is a predecessor of <subtask y>, (START,x) to indicate the beginning with <subtask x>, and (x,END) to signify the conclusion with <subtask x>. Remember that x, y are numbers.\nYour response should use the following format:\n\nNode:\n1.<subtask 1>\n2.<subtask 2>\n...\nEdges:(START,1) ... (n,END)"""

one_shot_example = {
    "alfworld":[
        {
            "role": "user",
            "content": "Here is an example.\nYou are in the middle of a room. Looking quickly around you, you see a bathtubbasin 1, a cabinet 2, a cabinet 1, a countertop 1, a garbagecan 1, a handtowelholder 1, a sinkbasin 1, a toilet 1, a toiletpaperhanger 1, and a towelholder 1.\n\nYour task is to: put a toiletpaper in toiletpaperhanger. The action list you can take: 1. go to {recep}\n2. task {obj} from {recep}\n3. put {obj} in/on {recep}\n4. open {recep}\n5. close {recep}\n6. toggle {obj} {recep}\n7. clean {obj} with {recep}\n8. heat {obj} with {recep}\n9. cool {obj} with {recep}\nwhere {obj} and {recep} correspond to objects and receptacles."
        },
        {
            "role": "assistant",
            "content": "Node:\n1: go to where the toiletpaper may be placed\n2: take toiletpaper from where you found it\n3: go to where the toiletpaperhanger is in\n4: put toiletpaper in/on toiletpaperhanger.\nEdge: (START,1) (1,2) (2,3) (3,4) (4,END)"
        }
    ],
    "webshop":[
        {
            "role": "user",
            "content": "Here is an example.\nWebShop [SEP] Instruction: [SEP] find me a motion detection high definition outdoor camera with 1080p hd definition, and price lower than 170.00 dollars [SEP] SearchThe action list you can take: search[keywords]\nclick[value]."
        },
        {
            "role": "assistant",
            "content": "Node:\n1: search[motion detection high definition outdoor camera with 1080p hd definition]\n2: click on the camera that meets the requirements of motion detection, high definition with 1080p, suitable for outdoor use and priced below $170\n3: click[buy now] to purchase the camera.\nEdge: (START,1) (1,2) (2,3) (3,END)"
        }
    ],
    "os": [
        {
            "role": "user",
            "content": "Here is an example.\nNow, I will start a new problem in a new OS. My problem is:\n\nIn the current directory, there is a hidden log file named \".log_file\". Your task is to count the number of distinct words in this file. A word is defined as having one or more non-space characters, separated by spaces. The computation should ignore case for all words.The action list you can take: bash: execute some bash code, take bash action\nfinish: have finished the task, take finish action\nanswer: have got the answer to the question, take answer action"
        },
        {
            "role": "assistant",
            "content": "Node:\n1: execute bash code to transform all letters to lower case and replace spaces with newlines\n2: execute bash code to sort the transformed words\n3: execute bash code to filter out unique words\n4: execute bash code to count the number of distinct words\n5: take answer action\nEdge: (START,1) (1,2) (2,3) (3,4) (4,5) (5,END)"
        }
    ],
    "toolbench":[    
        {
            "role": "user",
            "content": "Here are an example.\nTask: I am planning a trip to a foreign country and I need to detect the offensive language in travel reviews. Can you provide me with the hate score for the comment 'This hotel is a dump!'? Additionally, fetch the synonyms for the word 'delicious' in the local language.\nThe api list you can use: [{'name': 'detect_offensive_message_for_offensive_user_comment_detection', 'description': 'This is the subfunction for tool \"offensive_user_comment_detection\", you can use this tool.The description of this function is: \"This endpoint will detect hate speech, offensive language in a message\"', 'parameters': {'type': 'object', 'properties': {'q': {'type': 'string', 'description': '', 'example_value': 'son of a ***'}}, 'required': ['q'], 'optional': []}}, {'name': 'get_call_for_philippines_social_security_ocr', 'description': 'This is the subfunction for tool \"philippines_social_security_ocr\", you can use this tool.The description of this function is: \"Used to fetch api result using the request id received in responses.\"', 'parameters': {'type': 'object', 'properties': {'request_id': {'type': 'string', 'description': '', 'example_value': '68bbb910-da9b-4d8a-9a1d-4bd878b19846'}}, 'required': ['request_id'], 'optional': []}}, {'name': 'detect_language_for_quick_language_detector', 'description': 'This is the subfunction for tool \"quick_language_detector\", you can use this tool.The description of this function is: \"Feed this API a few sentences and have it determine what language it is with a confidence score\"', 'parameters': {'type': 'object', 'properties': {'text': {'type': 'string', 'description': '', 'example_value': \"Cela peut identifier 52 langues humaines à partir d'échantillons de texte et renvoyer des scores de confiance pour chaque\"}, 'detectedcount': {'type': 'integer', 'description': '', 'example_value': '5'}}, 'required': ['text'], 'optional': ['detectedcount']}}, {'name': 'get_for_synonyms_words', 'description': 'This is the subfunction for tool \"synonyms_words\", you can use this tool.The description of this function is: \"We use the get_for_synonyms_words method in our api, which returns a string with the synonyms separated by commas, which can later be treated in the programming language you use.\\n\\nWord - here you put the word you want to know the synonym of.\\nLanguage - EN,\"', 'parameters': {'type': 'object', 'properties': {'language': {'type': 'string', 'description': '', 'example_value': 'EN'}, 'word': {'type': 'string', 'description': '', 'example_value': 'car'}}, 'required': ['language', 'word'], 'optional': []}}, {'name': 'Finish', 'description': 'If you believe that you have obtained a result that can answer the task, please call this function to provide the final answer. Alternatively, if you recognize that you are unable to proceed with the task in the current state, call this function to restart. Remember: you must ALWAYS call this function at the end of your attempt, and the only part that will be shown to the user is the final answer, so it should contain sufficient information.', 'parameters': {'type': 'object', 'properties': {'return_type': {'type': 'string', 'enum': ['give_answer', 'give_up_and_restart']}, 'final_answer': {'type': 'string', 'description': 'The final answer you want to give the user. You should have this field if \"return_type\"==\"give_answer\"'}}, 'required': ['return_type']}}]",
        },
        {
            "role": "assistant",
            "content": "Node:\n1: Retrieve the hate score for the comment \"This hotel is a dump!\" to assess its level of offensiveness.\n2: Fetch synonyms for the word \"delicious\" in the local language to enhance understanding of culinary vocabulary.\n3: Rerun the synonyms retrieval for the word \"delicious\" using a supported language parameter, specifically English, to successfully obtain synonyms.\nEdge: (START,1) (START,2) (2,3) (1,END) (3,END)"
        },
        # {
        #     "role": "user",
        #     "content": "I'd like to know more about the education system in Finland. Can you find data on the average years of schooling for the Finnish population aged 15 and above, between 1995 and 2005?\nthe tools you can use are [{'name': 'getCountryData', 'description': 'Retrieve country-specific data.', 'parameters': {'countryCode': 'Required. string. The 2-letter ISO country code.', 'indicatorCode': 'string. The code for the specific indicator to retrieve data for.', 'dateRange': \"string. The date range for the data in the format 'YYYY:YYYY'.\"}, 'output': 'Successful response.', 'format': 'application/json', 'path': '/country-data', 'method': 'get'}, {'name': 'getEntityData', 'description': 'Retrieve data for a specific entity.', 'parameters': {'entityCode': 'Required. string. The code for the specific entity to retrieve data for.', 'indicatorCode': 'string. The code for the specific indicator to retrieve data for.', 'dateRange': \"string. The date range for the data in the format 'YYYY:YYYY'.\"}, 'output': 'Successful response.', 'format': 'application/json', 'path': '/entity-data', 'method': 'get'}, {'name': 'getProjectData', 'description': 'Retrieve data on World Bank projects.', 'parameters': {'projectId': 'Required. string. The unique identifier for the specific project to retrieve data for.'}, 'output': 'Successful response.', 'format': 'application/json', 'path': '/project-data', 'method': 'get'}, {'name': 'getIdaData', 'description': 'Retrieve data on International Development Association (IDA) resources and projects.', 'parameters': {'idaCode': 'Required. string. The unique identifier for the specific IDA resource or project to retrieve data for.'}, 'output': 'Successful response.', 'format': 'application/json', 'path': '/ida-data', 'method': 'get'}, {'name': 'searchIndicator', 'description': 'Search for an indicator code based on a keyword or phrase.', 'parameters': {'keyword': 'Required. string. The keyword or phrase to search for in the indicator descriptions.'}, 'output': 'Successful response.', 'format': 'application/json', 'path': '/search-indicator', 'method': 'get'}, {'name': 'listCountries', 'description': 'Retrieve a list of countries and their 2-letter ISO country codes.', 'parameters': {}, 'output': 'Successful response.', 'format': 'application/json', 'path': '/list-countries', 'method': 'get'}, {'name': 'listEntities', 'description': 'Retrieve a list of entities, such as regions or income groups, and their codes.', 'parameters': {}, 'output': 'Successful response.', 'format': 'application/json', 'path': '/list-entities', 'method': 'get'}]"
        # },
        # {
        #     "role": "assistant",
        #     "content": "Workflow: Search for indicators related to the average years of schooling globally. -> Fetch data for the indicator 'SE.SEC.ENRL.TC.ZS' related to Finland's education system between 1995 and 2005."
        # }
    ],
    "toolalpaca":[
        {
            "role": "user",
            "content": "Here are an example.\nTask: I'm planning to visit the Vancouver Aquarium tomorrow, and I'd like to take public transportation. Can you find the nearest station or stop to the aquarium and tell me the details about it, such as amenities and wait times?\nThe api list you can use: [{'name': 'getRealTimeTransitData', 'description': 'Retrieve real-time and historical data on buses, trains, ferries, and other transit options.', 'parameters': {'routeNumber': 'Required. string. The route number of the transit option.', 'dateTime': 'string. The date and time for historical data in YYYY-MM-DDTHH:MM:SS format.'}, 'output': 'Returns an array of transit data objects containing information such as route number, vehicle type, direction, arrival time, departure time, and location coordinates.', 'format': 'application/json', 'path': '/real-time-transit-data', 'method': 'get'}, {'name': 'findStationsOrStops', 'description': 'Locate stations or stops near a specific address or GPS coordinate.', 'parameters': {'address': 'string. The address to search for nearby stations or stops.', 'latitude': 'number. The latitude coordinate to search for nearby stations or stops.', 'longitude': 'number. The longitude coordinate to search for nearby stations or stops.', 'radius': 'integer. The search radius in meters.'}, 'output': 'Returns an array of station or stop objects containing information such as station or stop ID, name, address, latitude, longitude, and distance from the search location.', 'format': 'application/json', 'path': '/stations-or-stops', 'method': 'get'}, {'name': 'getStationOrStopDetails', 'description': 'Retrieve detailed information about a specific station or stop including amenities, accessibility, and estimated wait times.', 'parameters': {'stationOrStopId': 'Required. integer. The ID of the station or stop to retrieve information for.'}, 'output': 'Returns a station or stop object containing information such as station or stop ID, name, address, latitude, longitude, amenities, accessibility features, and estimated wait times for transit options.', 'format': 'application/json', 'path': '/stations-or-stops/{stationOrStopId}', 'method': 'get'}, {'name': 'searchStationsOrStops', 'description': 'Search for stations or stops by name or partial name.', 'parameters': {'query': 'Required. string. The name or partial name of the station or stop to search for.'}, 'output': 'Returns an array of station or stop objects containing information such as station or stop ID, name, address, latitude, and longitude.', 'format': 'application/json', 'path': '/stations-or-stops/search', 'method': 'get'}, {'name': 'getTransitRouteDetails', 'description': 'Retrieve detailed information about a specific transit route, including stops, schedules, and route map.', 'parameters': {'routeNumber': 'Required. string. The route number of the transit option.'}, 'output': 'Returns a transit route object containing information such as route number, vehicle type, direction, stops, schedules, and route map URL.', 'format': 'application/json', 'path': '/transit-route-details', 'method': 'get'}]"
        },
        {
            "role": "assistant",
            "content": "Node:\n1: Identify the nearest public transportation station or stop to a specific address and gather detailed information about it, such as amenities and wait times.\n2: Retrieve detailed information about a specific public transportation station or stop including amenities and wait times.\nEdge: (START,1) (1,2) (2,END)"
        },
        # {
        #     "role": "user",
        #     "content": "I recently started using a one-time-use plastic water bottle daily instead of my reusable one, totaling 30 bottles last month. Could you tell me the carbon emissions produced during the manufacturing of these bottles in the plastic industry? Once you have that information, suggest some relevant carbon offset projects for me to support.\nthe tools you can use are [{'name': 'calculateCarbonEmissions', 'description': 'Calculate the carbon emissions of any product or activity across various industries.', 'parameters': {'activity': 'Required. string. The name of the product or activity for which carbon emissions are to be calculated.', 'industry': 'Required. string. The industry to which the product or activity belongs.', 'quantity': 'number. The quantity of the product or activity. Default is 1.'}, 'output': 'Successful response containing calculated carbon emissions.', 'format': 'application/json', 'path': '/calculateCarbonEmissions', 'method': 'post'}, {'name': 'listOffsetProjects', 'description': 'Retrieve a list of carbon offset projects available to offset emissions.', 'parameters': {'industry': 'string. Filter projects by industry.', 'location': 'string. Filter projects by location.', 'pageSize': 'integer. The number of projects to return per page.', 'pageNumber': 'integer. The page number to return.'}, 'output': 'Successful response containing a list of carbon offset projects.', 'format': 'application/json', 'path': '/listOffsetProjects', 'method': 'get'}, {'name': 'purchaseCarbonOffsets', 'description': 'Purchase carbon offsets based on calculations from other functions.', 'parameters': {'projectId': 'Required. integer. The ID of the carbon offset project to purchase offsets from.', 'emissions': 'Required. number. The amount of carbon emissions to offset in metric tons CO2 equivalent.', 'userId': 'Required. integer. The ID of the user making the purchase.'}, 'output': 'Successful response containing purchase details.', 'format': 'application/json', 'path': '/purchaseCarbonOffsets', 'method': 'post'}]"
        # },
        # {
        #     "role": "assistant",
        #     "content": "Workflow: Calculate the carbon emissions for 30 plastic water bottles produced in the plastic industry. -> Find some carbon offset projects related to the plastic industry for me to consider supporting."
        # }
    ],
    "lumos":[
        {
            "role": "user",
            "content": "Here are an example.\nTask: Find the sum of all real solutions to the equation \\[\\sqrt{x} + \\sqrt{\\frac{4}{x}} + \\sqrt{x + \\frac{4}{x}} = 6.\\]\nThe api list you can use: {'Calculator(formula)': 'Calculate the input formula', 'SetEquation(equation)': 'Set up an equation to be solved', 'SolveEquation(equation)': 'Solve the previous set equation', 'Count(list)': 'Count the number of elements in the given list', 'SolveInequality(inequality)': 'Solve the previous set inequality', 'Code(pseudo_code)': 'Generate a Python function that corresponds to the pseudo code', 'Define(variable/number)': 'Define a variable or a number for latter usage.'}"
        },
        {
            "role": "assistant",
            "content": "Node:\n1: Move the $\\sqrt{x+\\frac4x}$ term to the right-hand side of the equation.\n2: Square both sides of the equation to eliminate the radicals.\n3: Simplify the equation to isolate the remaining radical.\n4: Square both sides of the equation again to eliminate the remaining radical.\n5: Solve the quadratic equation to find the roots.\n6: Check that both roots satisfy the original equation.\n7: Calculate the sum of the roots if they both satisfy the original equation.\nEdge: (START,1) (1,2) (2,3) (3,4) (4,5) (5,6) (6,7) (7,END)"
        },
        # {
        #     "role": "user",
        #     "content": "What county is the town where KNFM is licensed the capital of?\nthe tools you can use are {'Calculator(formula)': 'Calculate the input formula', 'KnowledgeQuery(query)': 'Capture the relevant webpages according to the query', 'ParagraphRetrieve(context, query)': 'Given a query, retrieve the most relevant paragraphs from the given context', 'QA(context, query)': 'Given context, answer the given query', 'Code(pseudo_code)': 'Generate a Python function that corresponds to the pseudo code.'}"
        # },
        # {
        #     "role": "assistant",
        #     "content": "Workflow: Identify the town where KNFM is licensed. -> Determine the county of which the identified town is the capital. The grounding could be based on the following results: The execution result for Subgoal 1 is Midland–Odessa metropolitan area."
        # }
    ],
    "wikihow":[
        {
            "role": "user",
            "content": "Here are an example.\nTask: How to Soothe Your Child's Allergies\nThe action list you can take: [\"Put together any documents that explain your child's allergy - what symptoms he/she has, past reactions, health records, etc.\\n\", 'Warn others about your child’s allergies.\\n', 'Get your child decongestants.', 'Treat the allergies medically.\\n', 'Avoid exposing the child to allergens or irritants.\\n', 'Speak to your child’s pediatrician if their symptoms do not improve.', 'Manage allergy symptoms with medication.\\n', 'Get your child tested by an allergist.\\n', \"Get a written diagnosis of your child's allergy from your pediatrician or allergist.\\n\", 'Give your child an over-the-counter antihistamine.\\n', 'Buy over-the-counter antihistamines.', 'Teach your child how to talk about their allergy.\\n', 'Get your child decongestants.\\n']"
        },
        {
            "role": "assistant",
            "content": "Node:\n1: Buy over-the-counter antihistamines.\n2: Get your child decongestants.\n3: Speak to your child’s pediatrician if their symptoms do not improve.\nEdge: (START,1) (START,2) (1,3) (2,3) (3,END)"
        }
        # {
        #     "role": "user",
        #     "content": "How to Be Ladylike the Sex and the City Way\nThe actions you can perform are ['Dress in the style.\\n', 'Act your age.', 'Dress timelessly and modestly.\\n', 'Avoid feminine body language and habits.\\n', 'Accessorize like Barbie.\\n', 'Respect others.', 'How to Pull Off Her Fashion Sense.\\n', 'Get some Flapper Glamour.\\n', 'Be well dressed and well groomed when it matters.', 'Have stylish clothes.\\n', 'When you are dressing formally, wear long pretty dresses with elegant jewellery.\\n', 'Be loyal.', 'Dress like a Flapper.So, you know how to act, how to talk... but how do you dress?.\\n', 'Dress fashionably and girlishly, but with a conservative slant.\\n']"
        # },
        # {
        #     "role": "assistant",
        #     "content": "Workflow: Act your age. -> Be loyal. -> Respect others. -> Be well dressed and well groomed when it matters."
        # }
    ]
}

two_shot_example = {
        "alfworld":[
        {
            "role": "user",
            "content": "Here are two examples. You need to strictly follow the format provided in the examples.\nYou are in the middle of a room. Looking quickly around you, you see a bathtubbasin 1, a cabinet 2, a cabinet 1, a countertop 1, a garbagecan 1, a handtowelholder 1, a sinkbasin 1, a toilet 1, a toiletpaperhanger 1, and a towelholder 1.\n\nYour task is to: put a toiletpaper in toiletpaperhanger. The action list you can take: 1. go to {recep}\n2. task {obj} from {recep}\n3. put {obj} in/on {recep}\n4. open {recep}\n5. close {recep}\n6. toggle {obj} {recep}\n7. clean {obj} with {recep}\n8. heat {obj} with {recep}\n9. cool {obj} with {recep}\nwhere {obj} and {recep} correspond to objects and receptacles."
        },
        {
            "role": "assistant",
            "content": "Node:\n1: go to where the toiletpaper may be placed\n2: take toiletpaper from where you found it\n3: go to where the toiletpaperhanger is in\n4: put toiletpaper in/on toiletpaperhanger.\nEdge: (START,1) (1,2) (2,3) (3,4) (4,END)"
        },
        {
            "role": "user",
            "content": "You are in the middle of a room. Looking quickly around you, you see a cabinet 21, a cabinet 20, a cabinet 19, a cabinet 18, a cabinet 17, a cabinet 16, a cabinet 15, a cabinet 14, a cabinet 13, a cabinet 12, a cabinet 11, a cabinet 10, a cabinet 9, a cabinet 8, a cabinet 7, a cabinet 6, a cabinet 5, a cabinet 4, a cabinet 3, a cabinet 2, a cabinet 1, a coffeemachine 1, a countertop 2, a countertop 1, a diningtable 1, a drawer 5, a drawer 4, a drawer 3, a drawer 2, a drawer 1, a fridge 1, a garbagecan 1, a microwave 1, a sinkbasin 1, a stoveburner 4, a stoveburner 3, a stoveburner 2, a stoveburner 1, and a toaster 1.\n\nYour task is to: find two tomato and put them in countertop.The action list you can take: 1. go to {recep}\n2. task {obj} from {recep}\n3. put {obj} in/on {recep}\n4. open {recep}\n5. close {recep}\n6. toggle {obj} {recep}\n7. clean {obj} with {recep}\n8. heat {obj} with {recep}\n9. cool {obj} with {recep}\nwhere {obj} and {recep} correspond to objects and receptacles."
        },
        {
            "role": "assistant",
            "content": "Node:\n1: go to sinkbasin\n2: take tomato from sinkbasin\n3: go to countertop\n4: put tomato in/on countertop\n5: go to garbagecan\n6: take tomato from garbagecan\n7: go to countertop\n8: put tomato in/on countertop.\nEdge: (START,1) (1,2) (2,3) (3,4) (4,5) (5,6) (6,7) (7,8) (8,END)" 
        }
    ],
    "webshop":[
        {
            "role": "user",
            "content": "Here are two examples. You need to strictly follow the format provided in the examples.\nWebShop [SEP] Instruction: [SEP] find me a motion detection high definition outdoor camera with 1080p hd definition, and price lower than 170.00 dollars [SEP] SearchThe action list you can take: search[keywords]\nclick[value]."
        },
        {
            "role": "assistant",
            "content": "Node:\n1: search[motion detection high definition outdoor camera with 1080p hd definition]\n2: click on the camera that meets the requirements of motion detection, high definition with 1080p, suitable for outdoor use and priced below $170\n3: click[buy now] to purchase the camera.\nEdge: (START,1) (1,2) (2,3) (3,END)"
        },
        {
            "role": "user",
            "content": "WebShop [SEP] Instruction: [SEP] i'm looking for a plant based protein drink that should be free from soy, gluten and dairy, and price lower than 130.00 dollars [SEP] SearchThe action list you can take: search[keywords]\nclick[value]."
        },
        {
            "role": "assistant",
            "content": "Node:\n1: search[plant-based protein drink soy gluten dairy free]\n2: click on the appropriate product that is free from soy, gluten, and dairy, and is priced under $130\n3: click[buy now] to purchase the drink.\nEdge: (START,1) (1,2) (2,3) (3,END)"
        }
    ],
    "os": [
        {
            "role": "user",
            "content": "Here are two examples. You need to strictly follow the format provided in the examples.\nNow, I will start a new problem in a new OS. My problem is:\n\nIn the current directory, there is a hidden log file named \".log_file\". Your task is to count the number of distinct words in this file. A word is defined as having one or more non-space characters, separated by spaces. The computation should ignore case for all words.The action list you can take: bash: execute some bash code, take bash action\nfinish: have finished the task, take finish action\nanswer: have got the answer to the question, take answer action"
        },
        {
            "role": "assistant",
            "content": "Node:\n1: execute bash code to transform all letters to lower case and replace spaces with newlines\n2: execute bash code to sort the transformed words\n3: execute bash code to filter out unique words\n4: execute bash code to count the number of distinct words\n5: take answer action\nEdge: (START,1) (1,2) (2,3) (3,4) (4,5) (5,END)"
        },
        {
            "role": "user",
            "content": "Now, I will start a new problem in a new OS. My problem is:\n\nTrack down all the files in your home folder that were last modified more than 7 days ago and are greater than 1M in size. Your answer should display the counts of such files.The action list you can take: bash: execute some bash code, take bash action\nfinish: have finished the task, take finish action\nanswer: have got the answer to the question, take answer action"
        },
        {
            "role": "assistant",
            "content": "Node:\n1: execute bash code to list files in the home directory\n2: execute bash code to find files last modified more than 7 days ago\n3: execute bash code to further filter files larger than 1M in size\n4: execute bash code to count such files\n5: take answer action\nEdge: (START,1) (1,2) (2,3) (3,4) (4,5) (5,END)"
        }
    ],
    "toolbench":[    
        {
            "role": "user",
            "content": "Here are two examples. You need to strictly follow the format provided in the examples.\nTask: I am planning a trip to a foreign country and I need to detect the offensive language in travel reviews. Can you provide me with the hate score for the comment 'This hotel is a dump!'? Additionally, fetch the synonyms for the word 'delicious' in the local language.\nThe api list you can use: [{'name': 'detect_offensive_message_for_offensive_user_comment_detection', 'description': 'This is the subfunction for tool \"offensive_user_comment_detection\", you can use this tool.The description of this function is: \"This endpoint will detect hate speech, offensive language in a message\"', 'parameters': {'type': 'object', 'properties': {'q': {'type': 'string', 'description': '', 'example_value': 'son of a ***'}}, 'required': ['q'], 'optional': []}}, {'name': 'get_call_for_philippines_social_security_ocr', 'description': 'This is the subfunction for tool \"philippines_social_security_ocr\", you can use this tool.The description of this function is: \"Used to fetch api result using the request id received in responses.\"', 'parameters': {'type': 'object', 'properties': {'request_id': {'type': 'string', 'description': '', 'example_value': '68bbb910-da9b-4d8a-9a1d-4bd878b19846'}}, 'required': ['request_id'], 'optional': []}}, {'name': 'detect_language_for_quick_language_detector', 'description': 'This is the subfunction for tool \"quick_language_detector\", you can use this tool.The description of this function is: \"Feed this API a few sentences and have it determine what language it is with a confidence score\"', 'parameters': {'type': 'object', 'properties': {'text': {'type': 'string', 'description': '', 'example_value': \"Cela peut identifier 52 langues humaines à partir d'échantillons de texte et renvoyer des scores de confiance pour chaque\"}, 'detectedcount': {'type': 'integer', 'description': '', 'example_value': '5'}}, 'required': ['text'], 'optional': ['detectedcount']}}, {'name': 'get_for_synonyms_words', 'description': 'This is the subfunction for tool \"synonyms_words\", you can use this tool.The description of this function is: \"We use the get_for_synonyms_words method in our api, which returns a string with the synonyms separated by commas, which can later be treated in the programming language you use.\\n\\nWord - here you put the word you want to know the synonym of.\\nLanguage - EN,\"', 'parameters': {'type': 'object', 'properties': {'language': {'type': 'string', 'description': '', 'example_value': 'EN'}, 'word': {'type': 'string', 'description': '', 'example_value': 'car'}}, 'required': ['language', 'word'], 'optional': []}}, {'name': 'Finish', 'description': 'If you believe that you have obtained a result that can answer the task, please call this function to provide the final answer. Alternatively, if you recognize that you are unable to proceed with the task in the current state, call this function to restart. Remember: you must ALWAYS call this function at the end of your attempt, and the only part that will be shown to the user is the final answer, so it should contain sufficient information.', 'parameters': {'type': 'object', 'properties': {'return_type': {'type': 'string', 'enum': ['give_answer', 'give_up_and_restart']}, 'final_answer': {'type': 'string', 'description': 'The final answer you want to give the user. You should have this field if \"return_type\"==\"give_answer\"'}}, 'required': ['return_type']}}]",
        },
        {
            "role": "assistant",
            "content": "Node:\n1: Retrieve the hate score for the comment \"This hotel is a dump!\" to assess its level of offensiveness.\n2: Fetch synonyms for the word \"delicious\" in the local language to enhance understanding of culinary vocabulary.\n3: Rerun the synonyms retrieval for the word \"delicious\" using a supported language parameter, specifically English, to successfully obtain synonyms.\nEdge: (START,1) (START,2) (2,3) (1,END) (3,END)"
        },
        {
            "role": "user",
            "content": "Task: My friends and I are planning a roulette night, and we want to review the recent trends. Could you please share the statistics of the wheel results for the past 3 hours? We are particularly interested in the count, percentage, last occurrence, and hot frequency percentage. It would be fantastic if you could also provide the latest spin data and the history of the last 20 spins.\nThe api list you can use: [{'name': 'statistics_for_immersiverouletteapi', 'description': 'This is the subfunction for tool \"immersiverouletteapi\", you can use this tool.The description of this function is: \"Stats of wheelResults:\\n\"count\"\\n\"percentage\"\\n\"lastOccurredAt\"\\n\"lastSeenBefore\"\\n\"hotFrequencyPercentage\": 1.11\\nNote that Duration is Hourly basis 1,2,3,.......72 ect\"', 'parameters': {'type': 'object', 'properties': {}, 'required': [], 'optional': []}}, {'name': 'latest_outcome_for_immersiverouletteapi', 'description': 'This is the subfunction for tool \"immersiverouletteapi\", you can use this tool.The description of this function is: \"Last Spin data\"', 'parameters': {'type': 'object', 'properties': {}, 'required': [], 'optional': []}}, {'name': 'history_for_immersiverouletteapi', 'description': 'This is the subfunction for tool \"immersiverouletteapi\", you can use this tool.The description of this function is: \"the last 20 spins histories includes all details about the game\"', 'parameters': {'type': 'object', 'properties': {}, 'required': [], 'optional': []}}, {'name': 'Finish', 'description': 'If you believe that you have obtained a result that can answer the task, please call this function to provide the final answer. Alternatively, if you recognize that you are unable to proceed with the task in the current state, call this function to restart. Remember: you must ALWAYS call this function at the end of your attempt, and the only part that will be shown to the user is the final answer, so it should contain sufficient information.', 'parameters': {'type': 'object', 'properties': {'return_type': {'type': 'string', 'enum': ['give_answer', 'give_up_and_restart']}, 'final_answer': {'type': 'string', 'description': 'The final answer you want to give the user. You should have this field if \"return_type\"==\"give_answer\"'}}, 'required': ['return_type']}}]",
        },
        {
            "role": "assistant",
            "content": "Node:\n1: Retrieve the statistics of the wheel results for the past 3 hours from the roulette game, including the count, percentage, last occurrence, and hot frequency percentage, as well as the latest spin data and history of the last 20 spins.\n2: Retrieve the latest spin results from the roulette game for the ongoing event planning.\n3: Obtain the history of the last 20 spins from the roulette game to provide a comprehensive overview for the event planning.\nEdge: (START,1) (1,2) (1,3) (2,END) (3,END)"
        },
    ],
    "toolalpaca":[
        {
            "role": "user",
            "content": "Here are two examples. You need to strictly follow the format provided in the examples.\nTask: I'm planning to visit the Vancouver Aquarium tomorrow, and I'd like to take public transportation. Can you find the nearest station or stop to the aquarium and tell me the details about it, such as amenities and wait times?\nThe api list you can use: [{'name': 'getRealTimeTransitData', 'description': 'Retrieve real-time and historical data on buses, trains, ferries, and other transit options.', 'parameters': {'routeNumber': 'Required. string. The route number of the transit option.', 'dateTime': 'string. The date and time for historical data in YYYY-MM-DDTHH:MM:SS format.'}, 'output': 'Returns an array of transit data objects containing information such as route number, vehicle type, direction, arrival time, departure time, and location coordinates.', 'format': 'application/json', 'path': '/real-time-transit-data', 'method': 'get'}, {'name': 'findStationsOrStops', 'description': 'Locate stations or stops near a specific address or GPS coordinate.', 'parameters': {'address': 'string. The address to search for nearby stations or stops.', 'latitude': 'number. The latitude coordinate to search for nearby stations or stops.', 'longitude': 'number. The longitude coordinate to search for nearby stations or stops.', 'radius': 'integer. The search radius in meters.'}, 'output': 'Returns an array of station or stop objects containing information such as station or stop ID, name, address, latitude, longitude, and distance from the search location.', 'format': 'application/json', 'path': '/stations-or-stops', 'method': 'get'}, {'name': 'getStationOrStopDetails', 'description': 'Retrieve detailed information about a specific station or stop including amenities, accessibility, and estimated wait times.', 'parameters': {'stationOrStopId': 'Required. integer. The ID of the station or stop to retrieve information for.'}, 'output': 'Returns a station or stop object containing information such as station or stop ID, name, address, latitude, longitude, amenities, accessibility features, and estimated wait times for transit options.', 'format': 'application/json', 'path': '/stations-or-stops/{stationOrStopId}', 'method': 'get'}, {'name': 'searchStationsOrStops', 'description': 'Search for stations or stops by name or partial name.', 'parameters': {'query': 'Required. string. The name or partial name of the station or stop to search for.'}, 'output': 'Returns an array of station or stop objects containing information such as station or stop ID, name, address, latitude, and longitude.', 'format': 'application/json', 'path': '/stations-or-stops/search', 'method': 'get'}, {'name': 'getTransitRouteDetails', 'description': 'Retrieve detailed information about a specific transit route, including stops, schedules, and route map.', 'parameters': {'routeNumber': 'Required. string. The route number of the transit option.'}, 'output': 'Returns a transit route object containing information such as route number, vehicle type, direction, stops, schedules, and route map URL.', 'format': 'application/json', 'path': '/transit-route-details', 'method': 'get'}]"
        },
        {
            "role": "assistant",
            "content": "Node:\n1: Identify the nearest public transportation station or stop to a specific address and gather detailed information about it, such as amenities and wait times.\n2: Retrieve detailed information about a specific public transportation station or stop including amenities and wait times.\nEdge: (START,1) (1,2) (2,END)"
        },
        {
            "role": "user",
            "content": "Task: I'm interested in learning more about the nutrients in different types of fish. Can you find some fish varieties and give me detailed information about their nutrients? Also, please make sure they're wild-caught and not farm-raised.\nThe api list you can use: [{'name': 'searchFoods', 'description': 'Search for foods by name, keyword, or nutrient content.', 'parameters': {'query': 'Required. string. The search query (name, keyword, or nutrient content).', 'pageSize': 'integer. The number of results per page.', 'pageNumber': 'integer. The page number to retrieve.'}, 'output': 'A list of foods matching the search query, including their IDs, names, and a brief description. Pagination information will also be provided.', 'format': 'application/json', 'path': '/searchFoods', 'method': 'get'}, {'name': 'getFoodDetails', 'description': 'Retrieve nutrient and ingredient information for a specific food.', 'parameters': {'foodId': 'Required. integer. The ID of the food for which details are to be retrieved.'}, 'output': 'Detailed information about the specified food, including its name, description, nutrient content, and ingredient list.', 'format': 'application/json', 'path': '/getFoodDetails', 'method': 'get'}, {'name': 'filterFoods', 'description': 'Filter food data based on specific criteria like allergens or organic production.', 'parameters': {'allergens': 'Array[string].  A list of allergens to filter foods by.', 'organic': 'boolean. Whether to filter foods based on organic production.', 'pageSize': 'integer. The number of results per page.', 'pageNumber': 'integer. The page number to retrieve.'}, 'output': 'A list of foods matching the specified filter criteria, including their IDs, names, and a brief description. Pagination information will also be provided.', 'format': 'application/json', 'path': '/filterFoods', 'method': 'get'}, {'name': 'listAllergens', 'description': 'Retrieve a list of all available allergens for filtering purposes.', 'parameters': {}, 'output': 'A list of all available allergens, including their names and descriptions.', 'format': 'application/json', 'path': '/listAllergens', 'method': 'get'}, {'name': 'getNutrientDetails', 'description': 'Retrieve detailed information about a specific nutrient.', 'parameters': {'nutrientId': 'Required. integer. The ID of the nutrient for which details are to be retrieved.'}, 'output': 'Detailed information about the specified nutrient, including its name, description, and units of measurement.', 'format': 'application/json', 'path': '/getNutrientDetails', 'method': 'get'}]"
        },
        {
            "role": "assistant",
            "content": "Node:\n1: Search for wild-caught fish and retrieve detailed nutrient information for each type.\n2: Filter the search results to include only wild-caught fish varieties, excluding any that are farm-raised or do not meet the specified criteria.\n3: Retrieve detailed nutrient information for each identified wild-caught fish variety based on their food ID.\n4: Retrieve detailed nutrient information for the next wild-caught fish variety using its specific food ID.\n5: Continue retrieving detailed nutrient information for the remaining wild-caught fish variety using its specific food ID.\nEdge: (START,1) (1,2) (2,3) (2,4) (2,5) (3,END) (4,END) (5,END)"
        },
    ],
    "lumos":[
        {
            "role": "user",
            "content": "Here are two examples. You need to strictly follow the format provided in the examples.\nTask: What part of the world is this river likely found in? There're some choices: A. asia, B. australia, C. europe, D. america. The image shows the following contents: The image features a serene scene of a river with several boats floating on the water. There are at least five boats visible in the scene, with some closer to the foreground and others further away. The boats vary in size and shape, adding to the diversity of the scene.  In addition to the boats, there are two people present in the image. One person is located near the center of the scene, while the other is situated closer to the right side. The combination of the boats and people creates a peaceful atmosphere in the image.\nThe api list you can use: {'VQA(image_context, query)': 'Given image context, answer the given query', 'QA(context, query)': 'Given context, answer the given query.'}"
        },
        {
            "role": "assistant",
            "content": "Node:\n1: Describe the boats and the surrounding environment.\n2: Answer which part of the world the river is likely found in based on the boats and environment.\n3: Select the choice from the four answer choices.\nEdge: (START,1) (1,2) (2,3) (3,END)"
        },
        {
            "role": "user",
            "content": "Task: How many of the people who started the great migration of the Slavs live in the country where the football tournament is held?\nThe api list you can use: {'Calculator(formula)': 'Calculate the input formula', 'KnowledgeQuery(query)': 'Capture the relevant webpages according to the query', 'ParagraphRetrieve(context, query)': 'Given a query, retrieve the most relevant paragraphs from the given context', 'QA(context, query)': 'Given context, answer the given query', 'Code(pseudo_code)': 'Generate a Python function that corresponds to the pseudo code.'}"
        },
        {
            "role": "assistant",
            "content": "Node:\n1: Determine where the football tournament was held.\n2: Identify who started the great migration of the Slavs. The grounding could be based on the following results: The execution result for Subgoal 1 is Brazil.\n3: Determine how many of the people who started the great migration of the Slavs live in the country where the football tournament was held. The grounding could be based on the following results: The execution result for Subgoal 1 is Brazil. The execution result for Subgoal 2 is Slavic-speaking tribes.\nEdge: (START,1) (1,2) (1,3) (2,3) (3,END)"
        },
    ],
    "wikihow":[
        {
            "role": "user",
            "content": "Here are two examples. You need to strictly follow the format provided in the examples.\nTask: How to Make Deux Foie Gras\nThe action list you can take: ['Sautee on both sides until brown while keeping it soft in the middle.', 'Broil the filets for 2 minutes and then spoon on the mustard sauce.\\n', 'Prepare the frittata mixture.\\n', 'Dust the foie gras lightly with the flour.', 'Dust the foie gras lightly with the flour.\\n', 'Combine the turkey and gravy in a bowl.\\n', 'Make the roux (the broth-like aspect of the dish).\\n', 'Add the chicken pieces to the gravy.\\n', 'Finally, arrange foie gras on plates and serve.', 'Put in the cooked black chickpeas into the gravy and stir well.\\n', 'Season the eggs and add the crème fraîche right as they finish cooking.\\n', 'Finally, arrange foie gras on plates and serve.\\n', 'Add the butter and the wine to the Dutch oven and mix the ingredients together with the vegetable puree.\\n']"
        },
        {
            "role": "assistant",
            "content": "Node:\n1: Dust the foie gras lightly with the flour.\n2: Sautee on both sides until brown while keeping it soft in the middle.\n3: Finally, arrange foie gras on plates and serve.\nEdge: (START,1) (1,2) (2,3) (3,END)"
        },
        {
            "role": "user",
            "content": "Task: How to Soothe Your Child's Allergies\nThe action list you can take: [\"Put together any documents that explain your child's allergy - what symptoms he/she has, past reactions, health records, etc.\\n\", 'Warn others about your child’s allergies.\\n', 'Get your child decongestants.', 'Treat the allergies medically.\\n', 'Avoid exposing the child to allergens or irritants.\\n', 'Speak to your child’s pediatrician if their symptoms do not improve.', 'Manage allergy symptoms with medication.\\n', 'Get your child tested by an allergist.\\n', \"Get a written diagnosis of your child's allergy from your pediatrician or allergist.\\n\", 'Give your child an over-the-counter antihistamine.\\n', 'Buy over-the-counter antihistamines.', 'Teach your child how to talk about their allergy.\\n', 'Get your child decongestants.\\n']"
        },
        {
            "role": "assistant",
            "content": "Node:\n1: Buy over-the-counter antihistamines.\n2: Get your child decongestants.\n3: Speak to your child’s pediatrician if their symptoms do not improve.\nEdge: (START,1) (START,2) (1,3) (2,3) (3,END)"
        }
    ]
}

add_prompt = {
    "alfworld": "Remember that you should follow the expression sytle of examples and not explicitly mention specific items. Your response should use the following format:\n\nNode:\n1.<subtask 1>\n2.<subtask 2>\n...\nEdges:(START,1) ... (n,END)\n\nNow it's your turn.\n\n",
    "webshop": "Remember that you should follow the expression sytle of examples and use <search> and <click>. Your response should use the following format:\n\nNode:\n1.<subtask 1>\n2.<subtask 2>\n...\nEdges:(START,1) ... (n,END)\n\nNow it's your turn.\n\n",
    "os": "Remember that you should follow the expression sytle of examples, use <execute bash code to do> to conduct subtasks and use <take answer action> to finish your task. Your response should use the following format:\n\nNode:\n1.<subtask 1>\n2.<subtask 2>\n...\nEdges:(START,1) ... (n,END)\n\nNow it's your turn.\n\n",
    "toolalpaca": "Remember that you should avoid explicitly specifying apis and follow the expression sytle of examples. Your response should use the following format:\n\nNode:\n1.<subtask 1>\n2.<subtask 2>\n...\nEdges:(START,1) ... (n,END)\n\nNow it's your turn.\n\n",
    "toolbench": "Remember that you should avoid explicitly specifying apis and follow the expression sytle of examples. Your response should use the following format:\n\nNode:\n1.<subtask 1>\n2.<subtask 2>\n...\nEdges:(START,1) ... (n,END)\n\nNow it's your turn.\n\n",
    "lumos": "Remember that you should avoid explicitly specifying apis and follow the expression sytle of examples. Your response should use the following format:\n\nNode:\n1.<subtask 1>\n2.<subtask 2>\n...\nEdges:(START,1) ... (n,END)\n\nNow it's your turn.\n\n",
    "wikihow": "Remember that you do not need to use all the actions provided. You only need to use the required actions from the list, unless all actions are necessary. Your response should use the following format:\n\nNode:\n1.<subtask 1>\n2.<subtask 2>\n...\nEdges:(START,1) ... (n,END)\n\nNow it's your turn.\n\n",
}