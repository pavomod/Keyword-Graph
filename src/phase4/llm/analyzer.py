"""
Phase 4: LLM-based Inconsistency Detection and Analysis using Gemini 2.5 Pro.

This module analyzes clusters of requirements to detect:
- Semantic duplicates or overlaps
- Logical inconsistencies or contradictions
- Ambiguities or underspecified requirements
- Missing requirements (gaps)
- Dependencies between requirements
"""

import json
import os
from pathlib import Path
from typing import Any

import google.generativeai as genai


def load_system_prompt(prompt_path: str) -> str:
    """Load the system prompt from file."""
    path = Path(prompt_path)
    if not path.exists():
        raise FileNotFoundError(f"System prompt not found: {prompt_path}")
    return path.read_text(encoding="utf-8")


def load_extraction_data(extraction_path: str) -> dict[str, str]:
    """Load extraction data and return mapping of requirement ID to text."""
    path = Path(extraction_path)
    if not path.exists():
        raise FileNotFoundError(f"Extraction file not found: {extraction_path}")
    
    data = json.loads(path.read_text(encoding="utf-8"))
    requirement_texts = {}
    for record in data.get("records", []):
        req_id = str(record["source_id"])
        requirement_texts[req_id] = record["input"]
    
    return requirement_texts


def load_clustering_report(clustering_report_path: str) -> tuple[dict[str, list[str]], dict[str, int]]:
    """Load clustering report and return (cluster_requirements_mapping, requirement_to_cluster_mapping)."""
    path = Path(clustering_report_path)
    if not path.exists():
        raise FileNotFoundError(f"Clustering report not found: {clustering_report_path}")

    data = json.loads(path.read_text(encoding="utf-8"))
    cluster_mapping = data.get("cluster_requirements_mapping", {})
    req_to_cluster_raw = data.get("requirement_to_cluster_mapping", {})
    # Flatten list values to single int (majority-vote produces [cluster_id])
    req_to_cluster: dict[str, int] = {
        req_id: (v[0] if isinstance(v, list) else int(v))
        for req_id, v in req_to_cluster_raw.items()
    }
    return cluster_mapping, req_to_cluster


def build_cluster_input(
    cluster_mapping: dict[str, list[str]],
    requirement_texts: dict[str, str],
) -> list[dict[str, Any]]:
    """
    Build the input format expected by the LLM based on the system prompt.
    
    Returns a list of cluster objects with cluster_id and requirements.
    """
    clusters = []
    
    for cluster_id, requirement_ids in cluster_mapping.items():
        requirements = []
        for req_id in requirement_ids:
            text = requirement_texts.get(req_id, "")
            if text:
                requirements.append({
                    "id": req_id,
                    "text": text
                })
        
        if requirements:
            clusters.append({
                "cluster_id": cluster_id,
                "requirements": requirements
            })
    
    return clusters


def preview_analysis_input(
    clustering_report_path: str,
    extraction_path: str,
    system_prompt_path: str,
    output_path: str = "results/phase4_preview.txt",
) -> None:
    """
    Preview the input that will be sent to Gemini without actually sending it.
    
    This allows you to verify the system prompt and cluster data before
    making an API call.
    
    Args:
        clustering_report_path: Path to clustering report JSON
        extraction_path: Path to extraction JSON
        system_prompt_path: Path to system prompt file
        output_path: Path where to save the preview
    """
    print("[Phase 4 Preview] Loading data...")
    
    # Load system prompt
    system_prompt = load_system_prompt(system_prompt_path)
    print(f"✓ System prompt loaded from {system_prompt_path}")
    
    # Load extraction data
    requirement_texts = load_extraction_data(extraction_path)
    print(f"✓ Loaded {len(requirement_texts)} requirement texts from {extraction_path}")
    
    # Load clustering report
    cluster_mapping, _ = load_clustering_report(clustering_report_path)
    print(f"✓ Loaded {len(cluster_mapping)} clusters from {clustering_report_path}")

    # Build input for LLM
    print("[Phase 4 Preview] Building cluster input for LLM...")
    clusters = build_cluster_input(cluster_mapping, requirement_texts)
    total_requirements = sum(len(c["requirements"]) for c in clusters)
    print(f"✓ Built input with {len(clusters)} clusters ({total_requirements} requirements)")
    
    # Build the complete message
    cluster_json = json.dumps(clusters, indent=2, ensure_ascii=False)
    
    # Create preview content
    preview_content = []
    preview_content.append("=" * 80)
    preview_content.append("PHASE 4 - GEMINI 2.5 PRO ANALYSIS PREVIEW")
    preview_content.append("=" * 80)
    
    preview_content.append("\n📋 SYSTEM PROMPT:\n")
    preview_content.append("-" * 80)
    preview_content.append(system_prompt)
    preview_content.append("-" * 80)
    
    preview_content.append("\n\n📊 INPUT CLUSTERS (JSON):\n")
    preview_content.append("-" * 80)
    preview_content.append(cluster_json)
    preview_content.append("-" * 80)
    
    preview_content.append("\n\n📈 STATISTICS:\n")
    preview_content.append(f"  • Total clusters: {len(clusters)}")
    preview_content.append(f"  • Total requirements: {total_requirements}")
    preview_content.append(f"  • System prompt length: {len(system_prompt)} chars")
    preview_content.append(f"  • Input data length: {len(cluster_json)} chars")
    
    preview_text = "\n".join(preview_content)
    
    # Print to console
    print("\n" + preview_text)
    
    # Save to file
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    output_file.write_text(preview_text, encoding="utf-8")
    print(f"\n✓ Preview saved to {output_path}")


def analyze_clusters_with_gemini(
    clusters: list[dict[str, Any]],
    system_prompt: str,
    api_key: str | None = None,
    model_name: str = "gemini-2.5-pro",
    temperature: float = 0.2,
    top_p: float = 0.9,
    max_output_tokens: int = 8000,
) -> dict[str, Any]:
    """
    Send clusters to Gemini for analysis.
    
    Args:
        clusters: List of cluster objects with requirements
        system_prompt: System prompt for the model
        api_key: Google API key (if None, uses GOOGLE_API_KEY env var)
        model_name: Gemini model to use (default: "gemini-2.5-pro")
        temperature: Model temperature (default: 0.2)
        top_p: Model top_p (default: 0.9)
        max_output_tokens: Max tokens (default: 8000)
    
    Returns:
        Parsed JSON response from the model
    """
    if api_key is None:
        api_key = os.getenv("MODEL_API_KEY")
    
    if not api_key:
        raise ValueError("API key not provided and MODEL_API_KEY env var not set")
    
    genai.configure(api_key=api_key)
    model = genai.GenerativeModel(model_name)
    
    # Build the user message with clusters
    user_message = json.dumps(clusters, indent=2, ensure_ascii=False)
    
    # Create the message with system prompt
    response = model.generate_content(
        [
            {
                "role": "user",
                "parts": [
                    {"text": system_prompt},
                    {"text": f"\n\n## INPUT CLUSTERS\n\n{user_message}"}
                ]
            }
        ],
        generation_config=genai.types.GenerationConfig(
            temperature=temperature,
            top_p=top_p,
            max_output_tokens=max_output_tokens,
        )
    )
    
    response_text = response.text
    
    # Extract JSON from the response
    # The model should output valid JSON only
    try:
        # Try to parse the entire response as JSON
        result = json.loads(response_text)
        return result
    except json.JSONDecodeError as e:
        # If that fails, try to extract JSON from the response
        # Look for JSON blocks between ``` markers or at the start
        import re
        
        print(f"[Debug] JSON parse failed: {e}")
        print(f"[Debug] Response length: {len(response_text)} chars")
        
        # Try to find JSON block with code fences (improved regex)
        # This matches ``` followed by optional 'json', then captures everything until the closing ```
        json_match = re.search(r'```(?:json)?\s*(.*?)```', response_text, re.DOTALL)
        if json_match:
            json_str = json_match.group(1).strip()
            print(f"[Debug] Extracted JSON from code fences: {len(json_str)} chars")
            try:
                return json.loads(json_str)
            except json.JSONDecodeError as e2:
                print(f"[Debug] Failed to parse extracted JSON: {e2}")
                pass
        
        # If no code fences, try to find JSON object directly (greedy match)
        json_match = re.search(r'\{[\s\S]*\}', response_text)
        if json_match:
            json_str = json_match.group(0)
            print(f"[Debug] Extracted JSON from greedy match: {len(json_str)} chars")
            try:
                return json.loads(json_str)
            except json.JSONDecodeError as e3:
                print(f"[Debug] Failed to parse greedy JSON: {e3}")
                pass
        
        # If all parsing fails, return the raw response wrapped in error info
        print("[Debug] All JSON parsing attempts failed")
        return {
            "error": "Failed to parse JSON from response",
            "raw_response": response_text
        }


def run_analysis(
    clustering_report_path: str,
    extraction_path: str,
    system_prompt_path: str,
    output_path: str,
    api_key: str | None = None,
    gemini_model: str = "gemini-2.5-pro",
    gemini_temperature: float = 0.2,
    gemini_top_p: float = 0.9,
    gemini_max_tokens: int = 8000,
    dry_run: bool = False,
) -> dict[str, Any]:
    """
    Run the complete analysis pipeline for Phase 4.
    
    Args:
        clustering_report_path: Path to clustering report JSON
        extraction_path: Path to extraction JSON
        system_prompt_path: Path to system prompt file
        output_path: Path where to save the analysis results
        api_key: Google API key (optional, uses env var if not provided)
        gemini_model: Gemini model to use (default: "gemini-2.5-pro")
        gemini_temperature: Model temperature (default: 0.2)
        gemini_top_p: Model top_p (default: 0.9)
        gemini_max_tokens: Max output tokens (default: 8000)
        dry_run: If True, show preview without sending to API
    
    Returns:
        Analysis result dictionary (or preview info if dry_run=True)
    """
    print("[Phase 4] Loading data...")
    
    # Load system prompt
    system_prompt = load_system_prompt(system_prompt_path)
    print(f"✓ System prompt loaded from {system_prompt_path}")
    
    # Load extraction data
    requirement_texts = load_extraction_data(extraction_path)
    print(f"✓ Loaded {len(requirement_texts)} requirement texts from {extraction_path}")
    
    # Load clustering report
    cluster_mapping, req_to_cluster = load_clustering_report(clustering_report_path)
    print(f"✓ Loaded {len(cluster_mapping)} clusters from {clustering_report_path}")

    # Build input for LLM
    print("[Phase 4] Building cluster input for LLM...")
    clusters = build_cluster_input(cluster_mapping, requirement_texts)
    total_requirements = sum(len(c["requirements"]) for c in clusters)
    print(f"✓ Built input with {len(clusters)} clusters ({total_requirements} requirements)")
    
    # If dry_run, show preview and return
    if dry_run:
        print("\n[Phase 4] DRY RUN MODE - Showing preview without sending to API\n")
        cluster_json = json.dumps(clusters, indent=2, ensure_ascii=False)
        
        preview = {
            "mode": "dry_run",
            "status": "preview_only",
            "system_prompt_length": len(system_prompt),
            "input_clusters": clusters,
            "total_clusters": len(clusters),
            "total_requirements": total_requirements,
        }
        
        # Save preview
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        output_file.write_text(
            json.dumps(preview, indent=2, ensure_ascii=False),
            encoding="utf-8"
        )
        print(f"✓ Preview saved to {output_path}")
        return preview
    
    # Send to Gemini
    print("[Phase 4] Sending to Gemini 2.5 Pro for analysis...")
    analysis_result = analyze_clusters_with_gemini(
        clusters,
        system_prompt,
        api_key=api_key,
        model_name=gemini_model,
        temperature=gemini_temperature,
        top_p=gemini_top_p,
        max_output_tokens=gemini_max_tokens,
    )
    print("✓ Analysis complete")
    
    # Inject requirement_to_cluster_mapping so the visualizer can show all requirements
    if isinstance(analysis_result, dict) and "error" not in analysis_result:
        analysis_result["requirement_to_cluster_mapping"] = req_to_cluster

    # Save results
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    output_file.write_text(
        json.dumps(analysis_result, indent=2, ensure_ascii=False),
        encoding="utf-8"
    )
    print(f"✓ Analysis saved to {output_path}")

    return analysis_result
