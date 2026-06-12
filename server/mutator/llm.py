"""LLM-assisted mutations — used when rule-based mutations plateau."""
from __future__ import annotations

import json
from typing import Optional


async def llm_mutate(
    prompt: dict,
    score,
    diagnosis: list[str],
    llm_url: str = "",
    llm_model: str = "",
) -> Optional[dict]:
    """Ask an LLM to suggest prompt improvements based on scoring failures.
    Returns mutated prompt or None if LLM unavailable.
    """
    if not llm_url or not llm_model:
        return None
    
    try:
        import aiohttp
        
        system_msg = (
            "You are a prompt engineering assistant. Given a structured image prompt "
            "and scoring feedback, suggest specific improvements to fix failing regions. "
            "Return ONLY valid JSON with the same structure as the input prompt. "
            "Focus on the elements listed in the diagnosis. Be specific and concise."
        )
        
        user_msg = json.dumps({
            "current_prompt": prompt,
            "score": score.to_dict() if hasattr(score, 'to_dict') else str(score),
            "diagnosis": diagnosis,
            "instruction": "Improve the prompt to fix these issues. Return the full updated prompt as JSON."
        })
        
        async with aiohttp.ClientSession() as session:
            async with session.post(
                f"{llm_url}/chat/completions",
                json={
                    "model": llm_model,
                    "messages": [
                        {"role": "system", "content": system_msg},
                        {"role": "user", "content": user_msg},
                    ],
                    "temperature": 0.4,
                    "max_tokens": 2000,
                },
                timeout=aiohttp.ClientTimeout(total=30),
            ) as resp:
                if resp.status != 200:
                    return None
                data = await resp.json()
                content = data["choices"][0]["message"]["content"]
                
                # Extract JSON from response
                content = content.strip()
                if content.startswith("```"):
                    content = content.split("\n", 1)[1].rsplit("```", 1)[0]
                return json.loads(content)
    
    except Exception:
        return None
