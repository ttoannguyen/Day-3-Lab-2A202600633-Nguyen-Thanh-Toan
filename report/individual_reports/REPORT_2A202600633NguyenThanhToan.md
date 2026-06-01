# Individual Report: Lab 3 - Chatbot vs ReAct Agent

- **Student Name**: Nguyễn Thanh Toàn
- **Student ID**: 2A202600633
- **Date**: 01/06/2026

---

## I. Technical Contribution (15 Points)

In this lab, I implemented several core parts of the agentic framework to bridge the gap between a standard conversational chatbot and a fully functioning ReAct (Reasoning and Acting) agent. My specific contributions are detailed below:

- **Modules Implemented**:
  - `src/agent/agent.py`: Wrote the entire core reasoning loop, unified provider compatibility layer (handling both raw string and dictionary LLM outputs), and a highly robust regular-expression-based parsing mechanism.
  - `app.py` **[NEW]**: Implemented a visually premium Streamlit web application with custom glassmorphism styling, a live step-by-step reasoning timeline (collapsible cards for Thought, Action, Observation, and Final Answer), and an interactive Telemetry Analytics Dashboard (aggregating prompt/completion tokens, latency, cost, and loop metrics) using `tracker.session_metrics`.
  - `src/tools/tools.py`: Created the simulated warehouse database and custom tool wrappers (`check_stock`, `get_discount`, `calc_shipping`). Also designed and integrated **2 new advanced tools**:
    - `get_product_price(item_name: str)`: Fetches exact base pricing of goods (e.g. iPhone = $999.00), preventing hallucinated prices.
    - `calculate_tax(subtotal: float, destination: str)`: Dynamically computes local sales tax (VAT) based on subtotal and destination (e.g., Hanoi = 10%).
  - `src/core/gemini_provider.py`: Modernized the Gemini LLM integration to use the official, next-generation Google GenAI (`google-genai`) SDK.
  - `src/core/llm_provider.py`: Created the custom provider (`LLMProvider`) that utilizes a custom API endpoint (MIMO) with OpenAI's Client schema, securing advanced provider-switching capabilities.
  - `src/telemetry/metrics.py`: Implemented realistic token-based pricing calculations for Google Gemini 1.5 Flash, OpenAI GPT-4o/GPT-4o-mini, custom MiMo models, and local offline models.

- **Code Highlights**:
  1. *Non-Greedy ReAct Parser & Loop* (`src/agent/agent.py`):
     ```python
     # Parsing Action using non-greedy regex to prevent swallowing simulated observations
     match = re.search(r"Action:\s*(\w+)\(([^)]*)\)", text)
     if not match:
         match = re.search(r"Action\s*:\s*(\w+)\(([^)]*)\)", text)
     ```
     This non-greedy matching ensures that variations in spacing or trailing characters in multi-line LLM outputs are handled perfectly without parsing failures.
     
  2. *Dynamic Argument Parser & Robust Dispatcher* (`src/agent/agent.py`):
     ```python
     # Dynamic argument parsing using standard AST literal evaluation and keyword mapping
     try:
         parsed = ast.literal_eval(f"({raw_args})")
         if isinstance(parsed, tuple):
             return tool_name, {"_args": list(parsed)}
         else:
             return tool_name, {"_args": [parsed]}
     except Exception:
         pass
     ```
     This safely parses arguments of varying Python types (`float`, `int`, `str`) dynamically, matching them seamlessly to positional or keyword arguments during function dispatch.

  3. *Telemetry Metrics Accumulation* (`app.py`):
     ```python
     # Clear previous metrics before running and sum up dynamically
     from src.telemetry.metrics import tracker
     tracker.session_metrics = []
     ...
     if tracker.session_metrics:
         prompt_tokens = sum(m.get("prompt_tokens", 0) for m in tracker.session_metrics)
         completion_tokens = sum(m.get("completion_tokens", 0) for m in tracker.session_metrics)
         cost_est = sum(m.get("cost_estimate", 0.0) for m in tracker.session_metrics)
     ```
     This ensures highly accurate session-specific telemetry metrics are dynamically aggregated and displayed in the frontend dashboard.

- **Documentation**:
  The `ReActAgent` class maintains an execution loop up to `max_steps`. For each step, it feeds the current `history_str` (containing all previous thoughts, actions, and observations) alongside the dynamic system prompt to the LLM. If the LLM generates an `Action`, the agent pauses generation, extracts the parameters, executes the local Python function from `src/tools/tools.py`, appends the result labeled as `Observation:`, and resumes the loop. If a `Final Answer` is parsed, the loop breaks early and returns the result. My architecture allows seamless swapping between **OpenAI**, **Gemini**, and **MiMo** dynamically via `.env` adjustments, proving the robustness of the Abstract `LLMProvider` interface.

---

## II. Debugging Case Study (10 Points)

During initial testing, the agent frequently got stuck in infinite loops or hallucinated tool names when using Google Gemini 1.5 Flash.

### Case Study 1: Gemini System Instruction Prefix Issue
- **Problem Description**: 
  The LLM kept repeating conversation dialogue or gave direct conversational answers (e.g., *"I can calculate the price for you!"*) instead of selecting the structured `Action: check_stock("iphone")` tool block. This resulted in the agent exceeding the `max_steps` limit and failing to resolve the query.
  
- **Log Source**: 
  An excerpt from the structured JSON logs in `logs/2026-06-01.log`:
  ```json
  {"timestamp": "2026-06-01T07:40:02.123456", "event": "PARSER_ERROR", "data": {"content": "Hello! I can certainly help you with your order. Let me know what you want to buy..."}}
  {"timestamp": "2026-06-01T07:40:15.654321", "event": "AGENT_END", "data": {"steps": 5, "status": "timeout"}}
  ```
  
- **Diagnosis**: 
  In the legacy template, `gemini_provider.py` used the deprecated `google-generativeai` package. Since it did not support structured API-level system instructions out-of-the-box, it prepended the system rules as a string prefix: `System: <rules> \n\n User: <prompt>`. Under this format, the LLM treated the rules as normal user conversation and "forgot" to follow the strict ReAct output formatting, leading to parsing failures.

- **Solution**: 
  I refactored the provider to use the new modern `google-genai` SDK and passed the system instruction directly to the API level via the `GenerateContentConfig` configuration object:
  ```python
  config = types.GenerateContentConfig(system_instruction=system_prompt)
  response = self.client.models.generate_content(model=self.model_name, contents=prompt, config=config)
  ```
  This immediately fixed the adherence rate, guiding the LLM back to the correct track on the next iteration.

### Case Study 2: Tool Argument Mismatch and Greedy Regex Parsing
- **Problem Description**:
  The agent failed with `TypeError: get_discount() got an unexpected keyword argument 'query'` when the LLM returned multiple simulated steps in a single response block.
  
- **Log Source**:
  ```json
  {"timestamp": "2026-06-01T08:40:13.701523", "event": "TOOL_OBSERVATION", "data": {"tool": "get_discount", "observation": "Lỗi khi thực thi công cụ 'get_discount': get_discount() got an unexpected keyword argument 'query'"}}
  ```
  
- **Diagnosis**:
  The previous regex pattern used greedy matching with `re.DOTALL`: `Action:\s*(\w+)\((.*)\)`. This matched from the first opening parenthesis `(` to the absolute last closing parenthesis `)` in the entire response block (which contained simulated observations and final answers). This bloated the arguments string, causing it to fall back to a raw string format and throw a keyword arguments error.
  
- **Solution**:
  1. I changed the regex pattern in `parse_action` to be non-greedy: `Action:\s*(\w+)\(([^)]*)\)`. This matches only the content within the immediate tool call parenthesis.
  2. I refactored `_execute_tool` to dynamically inspect and resolve keyword arguments `_args` and raw fallback formats, providing a robust execution wrapper that safely catches errors and returns friendly string messages.

---

## III. Personal Insights: Chatbot vs ReAct (10 Points)

Reflecting on my findings during the development and evaluation of both systems, I have gained the following core insights:

1. **Reasoning**: 
   The `Thought` block acts as a "System 2" cognitive scratchpad. Unlike a standard chatbot that performs "next-token prediction" in a single feed-forward pass, the ReAct loop allows the model to pause, decompose a query into logical steps, evaluate what it knows, and decide what it needs to find out. This planning drastically reduces logical errors in multi-step mathematics and search workflows.

2. **Reliability**: 
   A ReAct agent performs *worse* than a standard chatbot in simple, open-ended conversational tasks (e.g. Q&A like *"Who are you?"* or *"Write a poem"*). For simple tasks, the overhead of the ReAct format adds significant latency, increases token usage, and introduces a risk of parser failures for queries that do not require external tools in the first place.

3. **Observation**: 
   Environment feedback (`Observations`) acts as a ground-truth anchor. In a standard chatbot, if a user inputs an invalid coupon code, the chatbot will either hallucinate a discount or fail. In the ReAct loop, the observation (e.g., `{"status": "invalid"}`) forces the agent to update its belief state and choose an alternative path (e.g. continuing without a discount and informing the user) rather than hallucinating success.

---

## IV. Future Improvements (5 Points)

To scale this prototype into a production-level, resilient AI system, I propose the following architectural enhancements:

- **Scalability**: 
  Implement asynchronous execution using Python's `asyncio` or Celery task workers. If the agent needs to call multiple tools that do not depend on one another (e.g., checking stock for multiple unrelated items), they can be triggered concurrently to reduce total user latency.
  
- **Safety**: 
  Integrate an LLM-based supervisor guardrail (such as *Llama Guard* or *NeMo Guardrails*). This layer would intercept the output of the agent before executing a tool to verify that the tool arguments do not violate safety policies, preventing prompt-injection attacks from triggering malicious commands.
  
- **Performance**: 
  Implement a dynamic tool retriever using a Vector Database (like ChromaDB). Instead of packing all tool descriptions into the system prompt (which degrades context-window efficiency and confuses the LLM), the system would perform a similarity search on the user's query and inject only the top-K relevant tool definitions into the prompt.
