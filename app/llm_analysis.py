"""
LLM Analysis Module — uses Groq (Llama 3.3 70B) to analyze insurance claims.

What you'll learn:
  1. Prompt engineering: how to structure prompts for structured output
  2. RAG-augmented prompts: injecting retrieved policy context into the LLM call
  3. Structured JSON output: parsing LLM responses into usable data
  4. Streaming: receiving tokens as they are generated (real-time feel)
"""
import json
import re
from groq import Groq
from app.config import GROQ_API_KEY, GROQ_MODEL
from app.rag_pipeline import retrieve_relevant_policies


def get_groq_client() -> Groq:
    return Groq(api_key=GROQ_API_KEY)


def build_inpatient_query(claim: dict) -> str:
    """Convert a claim record into a natural language query for RAG retrieval."""
    return (
        f"DRG {claim.get('drg_code')} {claim.get('drg_description')} "
        f"charge-to-payment ratio {claim.get('charge_to_payment_ratio')} "
        f"inpatient hospital billing fraud detection rules"
    )


def build_physician_query(claim: dict) -> str:
    return (
        f"HCPCS {claim.get('hcpcs_code')} {claim.get('hcpcs_description')} "
        f"physician {claim.get('provider_type')} billing compliance "
        f"submitted charge {claim.get('avg_submitted_charge')} fraud indicators"
    )


def analyze_inpatient_claim(claim: dict) -> dict:
    """
    Full RAG + LLM pipeline for an inpatient claim:
      1. Build a retrieval query from claim data
      2. Retrieve relevant policy chunks from ChromaDB
      3. Build a prompt with claim details + retrieved policies
      4. Send to Groq LLM
      5. Parse structured response
    """
    # Step 1 & 2: Retrieve relevant policies
    query = build_inpatient_query(claim)
    retrieved = retrieve_relevant_policies(query, n_results=3)
    policy_context = _format_context(retrieved)

    # Step 3: Build the RAG-augmented prompt
    prompt = f"""You are a Medicare claims intelligence analyst. Analyze the following inpatient hospital claim for potential fraud, waste, or abuse.

## RETRIEVED POLICY CONTEXT (from knowledge base)
{policy_context}

## CLAIM DETAILS
- Hospital: {claim.get('provider_name')} ({claim.get('city')}, {claim.get('state')})
- DRG Code: {claim.get('drg_code')} — {claim.get('drg_description')}
- Total Discharges: {claim.get('total_discharges')}
- Average Submitted Charge: ${claim.get('avg_submitted_charge', 0):,.2f}
- Average Total Payment: ${claim.get('avg_total_payment', 0):,.2f}
- Average Medicare Payment: ${claim.get('avg_medicare_payment', 0):,.2f}
- Charge-to-Payment Ratio: {claim.get('charge_to_payment_ratio', 0):.1f}x

## TASK
Based on the policy context above and the claim details, provide:
1. A risk score (0-100) for this claim
2. Key findings (2-3 bullet points)
3. Specific policy violations or concerns identified
4. Recommended action

Respond ONLY in this exact JSON format:
{{
  "risk_score": <integer 0-100>,
  "risk_label": "<Low|Moderate|High|Critical>",
  "key_findings": ["<finding 1>", "<finding 2>", "<finding 3>"],
  "policy_concerns": "<specific policy or rule that may be violated>",
  "recommended_action": "<action to take>",
  "explanation": "<2-3 sentence summary>"
}}"""

    return _call_llm(prompt, retrieved)


def analyze_physician_claim(claim: dict) -> dict:
    """RAG + LLM pipeline for a physician claim."""
    query = build_physician_query(claim)
    retrieved = retrieve_relevant_policies(query, n_results=3)
    policy_context = _format_context(retrieved)

    allowed = claim.get('avg_medicare_allowed', 0) or 0
    charge = claim.get('avg_submitted_charge', 0) or 0
    ratio = round(charge / allowed, 1) if allowed > 0 else 0

    prompt = f"""You are a Medicare claims intelligence analyst. Analyze the following physician claim for potential fraud, waste, or abuse.

## RETRIEVED POLICY CONTEXT (from knowledge base)
{policy_context}

## CLAIM DETAILS
- Provider: {claim.get('provider_name')} ({claim.get('provider_type')})
- Location: {claim.get('city')}, {claim.get('state')}
- HCPCS Code: {claim.get('hcpcs_code')} — {claim.get('hcpcs_description')}
- Total Beneficiaries: {claim.get('total_beneficiaries')}
- Total Services: {claim.get('total_services')}
- Average Submitted Charge: ${charge:,.2f}
- Average Medicare Allowed: ${allowed:,.2f}
- Average Medicare Payment: ${claim.get('avg_medicare_payment', 0):,.2f}
- Charge-to-Allowed Ratio: {ratio}x

## TASK
Based on the policy context above and claim details, provide your analysis.

Respond ONLY in this exact JSON format:
{{
  "risk_score": <integer 0-100>,
  "risk_label": "<Low|Moderate|High|Critical>",
  "key_findings": ["<finding 1>", "<finding 2>", "<finding 3>"],
  "policy_concerns": "<specific policy or rule that may be violated>",
  "recommended_action": "<action to take>",
  "explanation": "<2-3 sentence summary>"
}}"""

    return _call_llm(prompt, retrieved)


def analyze_outpatient_claim(claim: dict) -> dict:
    """RAG + LLM pipeline for an outpatient hospital claim."""
    query = (
        f"APC {claim.get('apc_code')} {claim.get('apc_description')} "
        f"outpatient hospital billing compliance charge ratio "
        f"{claim.get('charge_to_allowed_ratio')} fraud indicators"
    )
    retrieved = retrieve_relevant_policies(query, n_results=3)
    policy_context = _format_context(retrieved)

    charge = claim.get('avg_submitted_charge', 0) or 0
    allowed = claim.get('avg_medicare_allowed', 0) or 0
    ratio = claim.get('charge_to_allowed_ratio', 0) or 0

    prompt = f"""You are a Medicare claims intelligence analyst. Analyze the following outpatient hospital claim for potential fraud, waste, or abuse.

## RETRIEVED POLICY CONTEXT (from knowledge base)
{policy_context}

## CLAIM DETAILS
- Hospital: {claim.get('provider_name')} ({claim.get('city')}, {claim.get('state')})
- APC Code: {claim.get('apc_code')} — {claim.get('apc_description')}
- Beneficiary Count: {claim.get('beneficiary_count')}
- Total Services: {claim.get('total_services')}
- Average Submitted Charge: ${charge:,.2f}
- Average Medicare Allowed: ${allowed:,.2f}
- Average Medicare Payment: ${claim.get('avg_medicare_payment', 0):,.2f}
- Outlier Services: {claim.get('outlier_services')}
- Charge-to-Allowed Ratio: {ratio}x

## TASK
Analyze this outpatient claim for fraud risk based on the policy context above.

Respond ONLY in this exact JSON format:
{{
  "risk_score": <integer 0-100>,
  "risk_label": "<Low|Moderate|High|Critical>",
  "key_findings": ["<finding 1>", "<finding 2>", "<finding 3>"],
  "policy_concerns": "<specific policy or rule that may be violated>",
  "recommended_action": "<action to take>",
  "explanation": "<2-3 sentence summary>"
}}"""

    return _call_llm(prompt, retrieved)


def ask_question_about_claim(claim: dict, question: str, claim_type: str = "inpatient",
                              chat_history: list = None) -> str:
    """
    Free-form Q&A about a specific claim — demonstrates conversational RAG.
    User can ask follow-up questions like:
      "Why is this DRG code suspicious?"
      "What documentation is required for this procedure?"
    """
    if chat_history is None:
        chat_history = []

    if claim_type == "inpatient":
        context_query = f"{question} DRG {claim.get('drg_code')} inpatient hospital"
    elif claim_type == "outpatient":
        context_query = f"{question} APC {claim.get('apc_code')} outpatient hospital"
    else:
        context_query = f"{question} HCPCS {claim.get('hcpcs_code')} physician billing"

    retrieved = retrieve_relevant_policies(context_query, n_results=3)
    policy_context = _format_context(retrieved)

    # System message with claim context + retrieved policies
    system_msg = f"""You are a Medicare claims expert. Answer questions about the following claim using the policy context provided. Be concise (3-5 sentences).

## RETRIEVED POLICY CONTEXT
{policy_context}

## CLAIM
{json.dumps({k: v for k, v in claim.items() if k not in ['id', 'ingested_at']}, indent=2)}"""

    # Build messages: system + full chat history + new question
    messages = [{"role": "system", "content": system_msg}]
    for msg in chat_history:
        messages.append({"role": msg["role"], "content": msg["content"]})
    messages.append({"role": "user", "content": question})

    try:
        client = get_groq_client()
        response = client.chat.completions.create(
            model=GROQ_MODEL,
            messages=messages,
            temperature=0.3,
            max_tokens=400,
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"Error: {e}"


def batch_analyze_claims(claims: list, claim_type: str, progress_callback=None) -> list:
    """
    Analyze a list of claims in batch.
    progress_callback(i, total) is called after each claim so the UI can show a progress bar.
    Returns list of dicts: original claim + analysis fields merged together.
    """
    results = []
    total = len(claims)
    for i, claim in enumerate(claims):
        try:
            if claim_type == "inpatient":
                analysis = analyze_inpatient_claim(claim)
            elif claim_type == "outpatient":
                analysis = analyze_outpatient_claim(claim)
            else:
                analysis = analyze_physician_claim(claim)

            merged = {**claim}
            merged["risk_score"] = analysis.get("risk_score", 0)
            merged["risk_label"] = analysis.get("risk_label", "Unknown")
            merged["key_findings"] = " | ".join(analysis.get("key_findings", []))
            merged["policy_concerns"] = analysis.get("policy_concerns", "")
            merged["recommended_action"] = analysis.get("recommended_action", "")
            merged["explanation"] = analysis.get("explanation", "")
            results.append(merged)
        except Exception as e:
            merged = {**claim, "risk_score": 0, "risk_label": "Error",
                      "key_findings": str(e), "policy_concerns": "",
                      "recommended_action": "Retry", "explanation": str(e)}
            results.append(merged)

        if progress_callback:
            progress_callback(i + 1, total)

    return results


def _format_context(retrieved: list[dict]) -> str:
    """Format retrieved policy chunks for inclusion in the prompt."""
    if not retrieved:
        return "No policy context retrieved."
    parts = []
    for i, r in enumerate(retrieved, 1):
        parts.append(
            f"[Policy {i} | Source: {r['source']} | Relevance: {r['similarity_score']}]\n{r['text']}"
        )
    return "\n\n---\n\n".join(parts)


def _call_llm(prompt: str, retrieved: list) -> dict:
    """Call Groq LLM and parse the JSON response."""
    try:
        client = get_groq_client()
        response = client.chat.completions.create(
            model=GROQ_MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2,
            max_tokens=600,
        )
        content = response.choices[0].message.content

        # Extract JSON from response (LLM sometimes adds extra text)
        json_match = re.search(r"\{.*\}", content, re.DOTALL)
        if json_match:
            result = json.loads(json_match.group())
        else:
            result = json.loads(content)

        result["retrieved_policies"] = retrieved
        return result

    except json.JSONDecodeError as e:
        return {
            "risk_score": 50,
            "risk_label": "Unknown",
            "key_findings": ["Could not parse LLM response"],
            "policy_concerns": "Parse error",
            "recommended_action": "Manual review",
            "explanation": f"LLM returned non-JSON response: {str(e)[:100]}",
            "retrieved_policies": retrieved,
        }
    except Exception as e:
        return {
            "risk_score": 0,
            "risk_label": "Error",
            "key_findings": [str(e)],
            "policy_concerns": "",
            "recommended_action": "Check API key and connection",
            "explanation": f"API error: {str(e)}",
            "retrieved_policies": retrieved,
        }
