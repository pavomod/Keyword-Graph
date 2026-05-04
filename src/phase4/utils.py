"""
Phase 4 Analysis Utilities

Utility functions for post-processing and analyzing Phase 4 results.
"""

import json
from pathlib import Path
from typing import Any, List, Dict


def load_analysis(analysis_path: str) -> dict[str, Any]:
    """Load Phase 4 analysis results from JSON file."""
    path = Path(analysis_path)
    if not path.exists():
        raise FileNotFoundError(f"Analysis file not found: {analysis_path}")
    
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)


def save_analysis(analysis: dict[str, Any], output_path: str) -> None:
    """Save analysis results to JSON file."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(analysis, f, indent=2, ensure_ascii=False)


def get_cluster_summary(cluster: dict[str, Any]) -> dict[str, int]:
    """Get summary statistics for a cluster."""
    return {
        "cluster_id": cluster.get("cluster_id"),
        "num_requirements": len(cluster.get("requirements", [])) if "requirements" in cluster else 0,
        "duplicates": len(cluster.get("duplicates", [])),
        "inconsistencies": len(cluster.get("inconsistencies", [])),
        "ambiguities": len(cluster.get("ambiguities", [])),
        "gaps": len(cluster.get("gaps", [])),
        "dependencies": len(cluster.get("dependencies", [])),
    }


def get_overall_statistics(analysis: dict[str, Any]) -> dict[str, int]:
    """Get overall statistics for the entire analysis."""
    clusters = analysis.get("clusters", [])
    
    total_requirements = 0
    total_duplicates = 0
    total_inconsistencies = 0
    total_ambiguities = 0
    total_gaps = 0
    total_dependencies = 0
    
    for cluster in clusters:
        # Count requirements in the first cluster mapping if available
        total_duplicates += len(cluster.get("duplicates", []))
        total_inconsistencies += len(cluster.get("inconsistencies", []))
        total_ambiguities += len(cluster.get("ambiguities", []))
        total_gaps += len(cluster.get("gaps", []))
        total_dependencies += len(cluster.get("dependencies", []))
    
    return {
        "num_clusters": len(clusters),
        "total_duplicates": total_duplicates,
        "total_inconsistencies": total_inconsistencies,
        "total_ambiguities": total_ambiguities,
        "total_gaps": total_gaps,
        "total_dependencies": total_dependencies,
        "total_issues": total_duplicates + total_inconsistencies + total_ambiguities + total_gaps + total_dependencies,
    }


def generate_summary_report(analysis: dict[str, Any]) -> str:
    """Generate a human-readable summary report of the analysis."""
    stats = get_overall_statistics(analysis)
    
    report = []
    report.append("=" * 70)
    report.append("PHASE 4 ANALYSIS SUMMARY REPORT")
    report.append("=" * 70)
    
    report.append(f"\n📊 Overall Statistics:")
    report.append(f"   • Clusters analyzed: {stats['num_clusters']}")
    report.append(f"   • Total issues found: {stats['total_issues']}")
    
    report.append(f"\n🔍 Issue Breakdown:")
    report.append(f"   • Semantic Duplicates: {stats['total_duplicates']}")
    report.append(f"   • Logical Inconsistencies: {stats['total_inconsistencies']}")
    report.append(f"   • Ambiguities: {stats['total_ambiguities']}")
    report.append(f"   • Missing Requirements (Gaps): {stats['total_gaps']}")
    report.append(f"   • Dependencies: {stats['total_dependencies']}")
    
    # Cluster breakdown
    report.append(f"\n📈 Cluster Breakdown:")
    for cluster in analysis.get("clusters", []):
        summary = get_cluster_summary(cluster)
        report.append(f"\n   Cluster {summary['cluster_id']}:")
        report.append(f"      - Duplicates: {summary['duplicates']}")
        report.append(f"      - Inconsistencies: {summary['inconsistencies']}")
        report.append(f"      - Ambiguities: {summary['ambiguities']}")
        report.append(f"      - Gaps: {summary['gaps']}")
        report.append(f"      - Dependencies: {summary['dependencies']}")
    
    report.append("\n" + "=" * 70)
    
    return "\n".join(report)


def filter_by_issue_type(analysis: dict[str, Any], issue_type: str) -> dict[str, Any]:
    """
    Filter analysis to show only specific issue type.
    
    issue_type: 'duplicates', 'inconsistencies', 'ambiguities', 'gaps', 'dependencies'
    """
    valid_types = ['duplicates', 'inconsistencies', 'ambiguities', 'gaps', 'dependencies']
    if issue_type not in valid_types:
        raise ValueError(f"Invalid issue type. Must be one of: {valid_types}")
    
    filtered_clusters = []
    
    for cluster in analysis.get("clusters", []):
        filtered_cluster = {
            "cluster_id": cluster.get("cluster_id"),
            issue_type: cluster.get(issue_type, [])
        }
        filtered_clusters.append(filtered_cluster)
    
    return {"clusters": filtered_clusters, "issue_type": issue_type}


def export_csv_report(analysis: dict[str, Any], output_path: str) -> None:
    """Export analysis results to CSV format."""
    import csv
    
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(path, 'w', newline='', encoding='utf-8') as csvfile:
        fieldnames = [
            'cluster_id',
            'issue_type',
            'requirement_ids',
            'description',
            'type/severity',
            'related_to'
        ]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        
        for cluster in analysis.get("clusters", []):
            cluster_id = cluster.get("cluster_id")
            
            # Write duplicates
            for dup in cluster.get("duplicates", []):
                writer.writerow({
                    'cluster_id': cluster_id,
                    'issue_type': 'duplicate',
                    'requirement_ids': ', '.join(dup.get('requirements', [])),
                    'description': dup.get('reason', ''),
                    'type/severity': dup.get('type', 'unknown'),
                    'related_to': ''
                })
            
            # Write inconsistencies
            for incon in cluster.get("inconsistencies", []):
                writer.writerow({
                    'cluster_id': cluster_id,
                    'issue_type': 'inconsistency',
                    'requirement_ids': ', '.join(incon.get('requirements', [])),
                    'description': incon.get('reason', ''),
                    'type/severity': incon.get('type', 'unknown'),
                    'related_to': ''
                })
            
            # Write ambiguities
            for amb in cluster.get("ambiguities", []):
                writer.writerow({
                    'cluster_id': cluster_id,
                    'issue_type': 'ambiguity',
                    'requirement_ids': amb.get('requirement', ''),
                    'description': amb.get('issue', ''),
                    'type/severity': 'clarification needed',
                    'related_to': amb.get('suggestion', '')
                })
            
            # Write gaps
            for gap in cluster.get("gaps", []):
                writer.writerow({
                    'cluster_id': cluster_id,
                    'issue_type': 'gap',
                    'requirement_ids': '',
                    'description': gap.get('description', ''),
                    'type/severity': 'missing',
                    'related_to': ', '.join(gap.get('related_to', []))
                })
            
            # Write dependencies
            for dep in cluster.get("dependencies", []):
                writer.writerow({
                    'cluster_id': cluster_id,
                    'issue_type': 'dependency',
                    'requirement_ids': f"{dep.get('from')} -> {dep.get('to')}",
                    'description': dep.get('reason', ''),
                    'type/severity': dep.get('type', 'unknown'),
                    'related_to': ''
                })


def get_critical_issues(analysis: dict[str, Any]) -> dict[str, List[Dict[str, Any]]]:
    """Extract potentially critical issues (inconsistencies, conflicts)."""
    critical = {
        "inconsistencies": [],
        "ambiguities": [],
        "gaps": []
    }
    
    for cluster in analysis.get("clusters", []):
        for incon in cluster.get("inconsistencies", []):
            critical["inconsistencies"].append({
                "cluster_id": cluster.get("cluster_id"),
                "requirements": incon.get("requirements", []),
                "reason": incon.get("reason", ""),
                "type": incon.get("type", "")
            })
        
        for amb in cluster.get("ambiguities", []):
            critical["ambiguities"].append({
                "cluster_id": cluster.get("cluster_id"),
                "requirement": amb.get("requirement", ""),
                "issue": amb.get("issue", ""),
                "suggestion": amb.get("suggestion", "")
            })
        
        for gap in cluster.get("gaps", []):
            critical["gaps"].append({
                "cluster_id": cluster.get("cluster_id"),
                "description": gap.get("description", ""),
                "related_to": gap.get("related_to", [])
            })
    
    return critical


if __name__ == "__main__":
    # Example usage
    import sys
    
    if len(sys.argv) > 1:
        analysis_file = sys.argv[1]
        
        print(f"\n📂 Loading analysis from: {analysis_file}")
        analysis = load_analysis(analysis_file)
        
        # Print summary
        print(generate_summary_report(analysis))
        
        # Extract critical issues
        critical = get_critical_issues(analysis)
        if critical["inconsistencies"]:
            print(f"\n⚠️  Found {len(critical['inconsistencies'])} inconsistencies that need attention")
        
        if critical["ambiguities"]:
            print(f"❓ Found {len(critical['ambiguities'])} ambiguities that need clarification")
        
        if critical["gaps"]:
            print(f"📋 Found {len(critical['gaps'])} missing requirements")
    
    else:
        print("Usage: python phase4_utils.py <analysis_file>")
        print("       python phase4_utils.py results/phase4_analysis.json")
