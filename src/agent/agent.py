import os
import re
import json
import sys
import subprocess
import ast
from typing import List, Dict, Any, Optional

from src.core.llm_provider import LLMProvider
from src.telemetry.logger import logger

# Dynamically import core tools if available to prevent ModuleNotFoundError on standard Lab 3 runs
try:
    from src.tools.web_search import web_search
except ImportError:
    web_search = None

try:
    from src.tools.fetch_web_content import fetch_web_content
except ImportError:
    fetch_web_content = None

try:
    from src.tools.file_io import read_file as fs_read_file, write_file as fs_write_file
except ImportError:
    fs_read_file = None
    fs_write_file = None


class ReActAgent:
    """
    A unified, professional ReAct Agent supporting both:
    1. Static E-commerce Tools (check_stock, get_discount, calc_shipping, etc.) for Lab 3 evaluation.
    2. Anthropic Agent Skills dynamic filesystem progressive disclosure architecture.
    """

    def __init__(
        self,
        llm: LLMProvider,
        tools: Optional[List[Dict[str, Any]]] = None,
        max_steps: int = 8,
        persist_history: bool = True
    ):
        self.llm = llm
        self.tools = tools or []
        self.max_steps = max_steps
        self.persist_history = persist_history
        self.history = []
        self.conversation_history: List[Dict[str, str]] = []
        self.skills_dir = os.path.join(
            os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), 
            "skills"
        )

    def _build_conversation_preamble(self) -> str:
        if not self.conversation_history:
            return ""
        lines = ["Conversation History:"]
        for msg in self.conversation_history:
            role = "User" if msg.get("role") == "user" else "Assistant"
            lines.append(f"{role}: {msg.get('content', '')}")
        lines.append("")
        return "\n".join(lines)

    def _store_conversation_turn(self, user_input: str, response: str) -> str:
        if self.persist_history:
            self.conversation_history.append({"role": "user", "content": user_input})
            self.conversation_history.append({"role": "assistant", "content": response})
        return response

    def parse_frontmatter(self, content: str) -> Dict[str, str]:
        """
        Lightweight YAML frontmatter parser (no PyYAML dependency)
        to extract skill metadata (name, description) from SKILL.md.
        """
        metadata = {}
        content = content.strip()
        if not content.startswith("---"):
            return metadata
            
        lines = content.splitlines()
        yaml_lines = []
        for line in lines[1:]:
            if line.strip() == "---":
                break
            yaml_lines.append(line)
            
        for line in yaml_lines:
            if ":" in line:
                key, val = line.split(":", 1)
                metadata[key.strip()] = val.strip()
        return metadata

    def load_available_skills_metadata(self) -> List[Dict[str, str]]:
        """
        Scan the skills/ directory on the filesystem to retrieve 
        Level 1 Metadata of all currently available skills.
        """
        skills_metadata = []
        if not os.path.exists(self.skills_dir):
            return skills_metadata
            
        try:
            # Scan all subdirectories in skills/
            for entry in os.scandir(self.skills_dir):
                if entry.is_dir():
                    skill_md_path = os.path.join(entry.path, "SKILL.md")
                    if os.path.exists(skill_md_path):
                        with open(skill_md_path, "r", encoding="utf-8") as f:
                            content = f.read()
                        meta = self.parse_frontmatter(content)
                        if "name" in meta and "description" in meta:
                            meta["path"] = f"skills/{entry.name}/SKILL.md"
                            skills_metadata.append(meta)
        except Exception as e:
            logger.log_event("SKILLS_METADATA_READ_ERROR", {"error": str(e)})
            
        return skills_metadata

    def get_system_prompt(self) -> str:
        """
        Dynamically build the System Prompt combining Lab 3 static tools 
        and available dynamic Agent Skills from the filesystem.
        """
        prompt_parts = []
        prompt_parts.append(
            "You are an advanced intelligent ReAct Agent capable of solving complex user requests.\n"
            "You must strictly follow the ReAct reasoning process (Thought -> Action -> Observation).\n"
        )

        # 1. Register static core e-commerce tools
        if self.tools:
            prompt_parts.append("### 🛠️ CORE E-COMMERCE TOOLS:")
            for t in self.tools:
                prompt_parts.append(f"- `{t['name']}`: {t['description']}")
            prompt_parts.append("")

        # 2. Register dynamic filesystem skills (Level 1 Skills)
        skills = self.load_available_skills_metadata()
        if skills:
            prompt_parts.append("### 🧠 AVAILABLE DYNAMIC AGENT SKILLS:")
            for skill in skills:
                prompt_parts.append(f"- **{skill['name']}** (Path: `{skill['path']}`): {skill['description']}")
            
            prompt_parts.append(
                "\n### 📖 PROGRESSIVE DISCLOSURE PROTOCOL:\n"
                "1. **Level 1 (Awareness)**: You are aware of the available dynamic skills listed above.\n"
                "2. **Level 2 (Read Detailed Guide)**: When a request matches a dynamic skill above, your VERY FIRST Action MUST be to call `read_file` to load the corresponding `SKILL.md` (e.g. `read_file(path=\"skills/create-new-skill/SKILL.md\")`).\n"
                "3. **Level 3 (Execution of scripts)**: Follow the instructions inside `SKILL.md` to execute python scripts under `scripts/` using `run_command` if needed."
            )
            prompt_parts.append("")

        prompt_parts.append(
            "### 📝 RESPONSE FORMATTING RULES (STRICTLY ENFORCED):\n"
            "For every conversation turn, you MUST strictly output your reasoning in the following formatted blocks:\n\n"
            "Thought: Your logical analysis about the next step, which tool to call, and what you expect to get from it.\n"
            "Action: tool_name(arguments)\n"
            "Observation: The result returned by executing the tool (This will be supplied to you by the system, DO NOT simulate this block yourself).\n\n"
            "... (Repeat the Thought -> Action -> Observation loop as many times as necessary to solve the request)\n\n"
            "Thought: I have gathered all necessary information to answer the user.\n"
            "Final Answer: Your final complete, clear, and direct response to the user's query.\n"
        )

        prompt_parts.append(
            "### ⚠️ CRITICAL RULES:\n"
            "1. You must call EXACTLY ONE tool per turn (i.e. output only a single `Action: ...` line per response).\n"
            "2. The Action block must be formatted precisely. You can pass arguments as Python positional or keyword arguments (e.g., `check_stock(\"iphone\")` or `calc_shipping(0.5, \"Hanoi\")`).\n"
            "3. Do not make up the `Observation:` block yourself. Always output the Action block and wait for the system response."
        )

        return "\n".join(prompt_parts)

    def parse_action(self, text: str) -> Optional[tuple]:
        """
        Smart argument parser to extract Tool Name and Arguments from LLM responses.
        Supports JSON, Python keyword arguments, and standard positional arguments.
        """
        # Match standard Action: tool_name(arguments) using non-greedy parsing to avoid swallowing multi-line blocks
        match = re.search(r"Action:\s*(\w+)\(([^)]*)\)", text)
        if not match:
            # Fallback regex for flexible spacing
            match = re.search(r"Action\s*:\s*(\w+)\(([^)]*)\)", text)
            if not match:
                return None
            
        tool_name = match.group(1).strip()
        raw_args = match.group(2).strip()
        
        if not raw_args:
            return tool_name, {}

        # 1. Attempt parsing as a valid JSON or dictionary string
        try:
            cleaned_json = raw_args.replace("'", '"')
            parsed_args = json.loads(cleaned_json)
            if isinstance(parsed_args, dict):
                return tool_name, parsed_args
            else:
                return tool_name, {"_raw": parsed_args}
        except json.JSONDecodeError:
            pass

        # 2. Attempt parsing as standard Python keyword arguments (e.g., key="value" or key=123)
        parsed_args = {}
        kw_matches = list(re.finditer(r"(\w+)\s*=\s*(?:\"([^\"]*)\"|'([^']*)'|({.*?})|(\[.*?\])|(\w+))", raw_args, re.DOTALL))
        
        if kw_matches:
            for kw in kw_matches:
                key = kw.group(1)
                val = next(v for v in kw.groups()[1:] if v is not None)
                val = val.strip()
                
                if (val.startswith("{") and val.endswith("}")) or (val.startswith("[") and val.endswith("]")):
                    try:
                        val = json.loads(val.replace("'", '"'))
                    except Exception:
                        pass
                elif val.lower() == "true":
                    val = True
                elif val.lower() == "false":
                    val = False
                elif val.isdigit():
                    val = int(val)
                elif val.replace('.', '', 1).isdigit() and val.count('.') < 2:
                    val = float(val)
                    
                parsed_args[key] = val
            return tool_name, parsed_args

        # 3. Fallback using AST parser to analyze standard Python positional arguments
        try:
            parsed = ast.literal_eval(f"({raw_args})")
            if isinstance(parsed, tuple):
                return tool_name, {"_args": list(parsed)}
            else:
                return tool_name, {"_args": [parsed]}
        except Exception:
            pass

        # 4. Final Fallback: Treat as raw string
        cleaned_str = re.sub(r"^[\"']|[\"']$", "", raw_args).strip()
        return tool_name, {"query": cleaned_str, "url": cleaned_str, "path": cleaned_str, "content": cleaned_str, "_raw": cleaned_str}

    def run(self, user_input: str) -> str:
        """
        Run the ReAct Reasoning and Acting loop (Thought-Action-Observation) with telemetry integration.
        """
        logger.log_event("AGENT_START", {"input": user_input, "model": self.llm.model_name})
        print(f"\n🤖 [AGENT] Starting ReAct reasoning loop for query: '{user_input}'...")
        
        # Initialize conversation history
        history_str = self._build_conversation_preamble()
        history_str += f"Question: {user_input}\n"
        self.history = [
            {"role": "user", "content": user_input}
        ]
        
        steps = 0
        while steps < self.max_steps:
            steps += 1
            print(f"\n💭 [AGENT] Reasoning Turn {steps}/{self.max_steps}...")
            
            # Dynamically fetch the system prompt
            system_prompt = self.get_system_prompt()
            
            try:
                # 1. Generate LLM completion
                response = self.llm.generate(history_str, system_prompt=system_prompt)
                
                # Handle provider response variations (MIMO returns str, OpenAI/Gemini return dict)
                if isinstance(response, dict):
                    content = response.get("content", "").strip()
                    usage = response.get("usage", {})
                    latency_ms = response.get("latency_ms", 0)
                    provider = response.get("provider", "")
                else:
                    content = str(response).strip()
                    usage = {
                        "prompt_tokens": len(history_str) // 4,
                        "completion_tokens": len(content) // 4,
                        "total_tokens": (len(history_str) + len(content)) // 4
                    }
                    latency_ms = 0
                    provider = "mimo"

                # Log metrics to Performance Tracker
                from src.telemetry.metrics import tracker
                tracker.track_request(
                    provider=provider,
                    model=self.llm.model_name,
                    usage=usage,
                    latency_ms=latency_ms
                )

                print(f"\n{content}\n")
                logger.log_event("AGENT_THOUGHT", {"step": steps, "response": content})
                
                # Add LLM output to conversation history
                self.history.append({"role": "assistant", "content": content})
                history_str += content + "\n"
                
                # 2. Check if the LLM provided the Final Answer
                final_match = re.search(r"Final\s*Answer\s*:\s*(.*)", content, re.DOTALL | re.IGNORECASE)
                if final_match and "Action:" not in content:
                    final_answer = final_match.group(1).strip()
                    logger.log_event("AGENT_END", {"steps": steps, "status": "completed"})
                    return self._store_conversation_turn(user_input, final_answer)
                
                # 3. Parse Action (Tool Call)
                parsed = self.parse_action(content)
                if not parsed:
                    # Fallback: if Final Answer is mixed in, extract directly
                    if final_match:
                        final_answer = final_match.group(1).strip()
                        logger.log_event("AGENT_END", {"steps": steps, "status": "completed"})
                        return self._store_conversation_turn(user_input, final_answer)
                        
                    warning_msg = "Error: You did not output a Final Answer or call a valid Action format (e.g. Action: tool_name(arguments)). Please check formatting rules."
                    print(f"⚠️ [PARSER ERROR] {warning_msg}")
                    self.history.append({"role": "user", "content": warning_msg})
                    history_str += f"Observation: {warning_msg}\n"
                    continue
                    
                tool_name, tool_args = parsed
                print(f"⚙️ [EXECUTE TOOL] Executing tool '{tool_name}' with arguments: {tool_args}...")
                logger.log_event("TOOL_CALL", {"tool": tool_name, "args": str(tool_args)})
                
                # 4. Execute the matched tool
                observation = self._execute_tool(tool_name, tool_args)
                
                obs_str = str(observation)
                print(f"🔍 [OBSERVATION] Result received:\n{obs_str[:800]}..." if len(obs_str) > 800 else f"🔍 [OBSERVATION] Result received:\n{obs_str}")
                logger.log_event("TOOL_OBSERVATION", {"tool": tool_name, "observation": obs_str})
                
                # Append Observation back into history for next step
                self.history.append({"role": "user", "content": f"Observation: {obs_str}"})
                history_str += f"Observation: {obs_str}\n"
                
            except Exception as e:
                error_msg = f"System error during ReAct loop: {str(e)}"
                print(f"🚨 [SYSTEM ERROR] {error_msg}")
                logger.log_event("AGENT_ERROR", {"error": str(e)})
                return self._store_conversation_turn(user_input, error_msg)
                
        timeout_msg = "Failed to reach a Final Answer within the maximum steps limit."
        logger.log_event("AGENT_END", {"steps": steps, "status": "timeout"})

        if self.max_steps > 3:
            limit_prompt = (
                "Observation: You have reached the maximum number of reasoning steps "
                f"({self.max_steps}). You must provide a Final Answer now and explicitly "
                "state that the limit was reached."
            )
            self.history.append({"role": "user", "content": limit_prompt})
            history_str += f"{limit_prompt}\n"

            try:
                response = self.llm.generate(history_str, system_prompt=self.get_system_prompt())

                if isinstance(response, dict):
                    content = response.get("content", "").strip()
                    usage = response.get("usage", {})
                    latency_ms = response.get("latency_ms", 0)
                    provider = response.get("provider", "")
                else:
                    content = str(response).strip()
                    usage = {
                        "prompt_tokens": len(history_str) // 4,
                        "completion_tokens": len(content) // 4,
                        "total_tokens": (len(history_str) + len(content)) // 4
                    }
                    latency_ms = 0
                    provider = "mimo"

                from src.telemetry.metrics import tracker
                tracker.track_request(
                    provider=provider,
                    model=self.llm.model_name,
                    usage=usage,
                    latency_ms=latency_ms
                )

                self.history.append({"role": "assistant", "content": content})
                history_str += content + "\n"

                final_match = re.search(r"Final\s*Answer\s*:\s*(.*)", content, re.DOTALL | re.IGNORECASE)
                if final_match:
                    final_answer = final_match.group(1).strip()
                    logger.log_event("AGENT_END", {"steps": steps, "status": "forced_completion"})
                    return self._store_conversation_turn(user_input, final_answer)

                forced_reply = f"{timeout_msg} Limit reached ({self.max_steps}). {content}"
                logger.log_event("AGENT_END", {"steps": steps, "status": "forced_completion"})
                return self._store_conversation_turn(user_input, forced_reply)
            except Exception as e:
                error_msg = (
                    f"{timeout_msg} Limit reached ({self.max_steps}). "
                    f"System error during final forced response: {str(e)}"
                )
                logger.log_event("AGENT_ERROR", {"error": str(e)})
                return self._store_conversation_turn(user_input, error_msg)

        # Attempt to rescue the run by extracting any final answer found in history
        rescue_match = re.search(r"Final\s*Answer\s*:\s*(.*)", history_str, re.DOTALL | re.IGNORECASE)
        if rescue_match:
            return self._store_conversation_turn(user_input, rescue_match.group(1).strip())

        return self._store_conversation_turn(user_input, timeout_msg)

    def _execute_tool(self, tool_name: str, args: Any) -> Any:
        """
        Coordinate the execution of static tools and core system tools.
        """
        try:
            # Handle list-based or dict-based arguments dynamically
            args_dict = args if isinstance(args, dict) else {}
            args_list = args.get("_args", []) if isinstance(args, dict) else (args if isinstance(args, list) else [])

            # --- SEGMENT 1: E-commerce Static Tools (Lab 3) ---
            for tool in self.tools:
                if tool['name'] == tool_name:
                    func = tool.get('func')
                    if not func:
                        return f"Error: Tool '{tool_name}' does not have an executable function."
                    
                    # Call function with parsed arguments
                    if args_list:
                        return func(*args_list)
                    elif args_dict:
                        # Filter helper keys starting with underscore
                        clean_dict = {k: v for k, v in args_dict.items() if not k.startswith("_")}
                        if clean_dict:
                            return func(**clean_dict)
                        elif "_raw" in args_dict:
                            return func(args_dict["_raw"])
                    
                    # Fallback as raw string
                    return func(str(args))

            # --- SEGMENT 2: Dynamic Agent Skills Core Tools ---
            if tool_name == "web_search":
                if web_search:
                    q = args_dict.get("query") or args_dict.get("q") or (args_list[0] if args_list else None) or args_dict.get("_raw") or str(args)
                    return web_search(query=q)
                return "Error: web_search tool is not integrated or library is missing."

            elif tool_name == "fetch_web_content":
                if fetch_web_content:
                    u = args_dict.get("url") or (args_list[0] if args_list else None) or args_dict.get("_raw") or str(args)
                    return fetch_web_content(url=u)
                return "Error: fetch_web_content tool is not integrated or library is missing."

            elif tool_name == "read_file":
                path = args_dict.get("path") or args_dict.get("filepath") or (args_list[0] if args_list else None) or args_dict.get("_raw") or str(args)
                path = str(path).strip().strip("'\"")
                
                # Utilize lab read file tool if registered
                if fs_read_file:
                    return fs_read_file(path=path)
                
                # Fallback to standard Python file reading
                with open(path, "r", encoding="utf-8") as f:
                    return f.read()

            elif tool_name == "write_file":
                path = args_dict.get("path") or args_dict.get("filepath")
                content = args_dict.get("content") or args_dict.get("data")
                
                if not path or not content:
                    if args_list and len(args_list) >= 2:
                        path, content = args_list[0], args_list[1]
                    elif isinstance(args_dict, dict):
                        keys = [k for k in args_dict.keys() if not k.startswith("_")]
                        if len(keys) >= 2:
                            path = args_dict[keys[0]]
                            content = args_dict[keys[1]]
                
                path = str(path).strip().strip("'\"")
                
                # Utilize lab write file tool if registered
                if fs_write_file:
                    return fs_write_file(path=path, content=str(content))
                
                # Fallback to standard Python file writing
                with open(path, "w", encoding="utf-8") as f:
                    f.write(str(content))
                return f"Successfully wrote to file '{path}'."

            elif tool_name == "run_command":
                cmd = args_dict.get("cmd") or args_dict.get("command") or (args_list[0] if args_list else None) or args_dict.get("_raw") or str(args)
                cmd = str(cmd).strip().strip("'\"")
                
                logger.log_event("TOOL_USE_START", {"tool": "run_command", "command": cmd})
                result = subprocess.run(
                    cmd,
                    shell=True,
                    capture_output=True,
                    text=True,
                    timeout=30
                )
                output = result.stdout or ""
                error = result.stderr or ""
                full_output = f"stdout:\n{output}\nstderr:\n{error}" if error else output
                logger.log_event("TOOL_USE_END", {"tool": "run_command", "status": "success" if result.returncode == 0 else "failed"})
                return full_output

            else:
                return f"Error: Tool '{tool_name}' not found in registry. Please verify the tool name."

        except Exception as e:
            return f"Error executing tool '{tool_name}': {str(e)}"
