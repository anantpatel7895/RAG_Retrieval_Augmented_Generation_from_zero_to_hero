#!/usr/bin/env python3
"""
main_orchestrator.py
Master Pipeline Driver and Orchestrator for Agentic SDLC Suite.

What it does:
  - Classifies target repositories into domain categories (backend, mfe, docs_only).
  - Orchestrates sequential execution across all lifecycle stages: Knowledge Base Indexing, 
    Planning (Macro Approval Gate), Developer Implementation, Documentation Audit, 
    SAST Security, Architecture Audit, PR Review, and QA Testing.
  - Enforces budget caps, process safety, two-tier Human-in-the-Loop governance, and structured run manifest recording.

How it does it:
  - Auto-detects domain type via file presence (pom.xml/build.gradle -> backend; package.json -> mfe; else -> docs_only).
  - Cross-checks domain findings against config/repo_manifest.yaml and logs warnings on mismatch.
  - Sequentially invokes domain-specific agent submodules, handling macro plan rejection and BudgetExceededError gracefully.

Prerequisites:
  - Local repository workspace populated under WORKSPACE_ROOT.
  - Environment variables set for Anthropic API, Jira (if ticket flags used), and optional Neo4j credentials.
  - Aider CLI executable on system PATH.

Dependencies:
  - config.settings
  - utils.token_tracker (TokenTrackerV2, BudgetExceededError)
  - utils.approval_gate (CLIApprovalGate)
  - utils.run_manifest (RunManifestTracker)
  - agents.* (all 10 agent modules)
  - knowledge_base_builder (FRAMLKnowledgeBaseBuilder)
  - mfe_knowledge_base_builder (MFEKnowledgeBaseBuilder)

Pipeline Placement:
  - Top-level CLI entry point driving the entire Agentic SDLC pipeline sequence.
"""

import os
import sys
import yaml
import argparse
import logging
from typing import List, Dict, Any, Optional
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent))

from config import settings
from utils.token_tracker import TokenTrackerV2, BudgetExceededError
from utils.approval_gate import CLIApprovalGate
from utils.run_manifest import RunManifestTracker

from knowledge_base_builder import FRAMLKnowledgeBaseBuilder
from mfe_knowledge_base_builder import MFEKnowledgeBaseBuilder

from agents.planning_agent import PlanningAgent
from agents.estimating_agent import EstimatingAgent
from agents.developer_agent import DeveloperAgent
from agents.qa_agent import QAAgentSuite
from agents.security_agent import SecurityAgent
from agents.architect_agent import ArchitectAgent
from agents.pr_review_agent import PRReviewAgent
from agents.repo_doc_auditor_agent import RepoDocAuditorAgent
from agents.mfe_auditor_agent import MFEAuditorAgent

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("main_orchestrator")


def classify_repository_domain(repo_path: str) -> str:
    """
    Classifies repository domain based on root file detection rules:
      - pom.xml or build.gradle present -> backend
      - package.json present (no pom/gradle) -> mfe
      - Otherwise -> docs_only
    """
    path = Path(repo_path)
    if not path.exists():
        logger.warning(f"Repository directory does not exist: {repo_path}. Defaulting to docs_only.")
        return "docs_only"

    has_pom = (path / "pom.xml").exists()
    has_gradle = (path / "build.gradle").exists() or (path / "build.gradle.kts").exists()
    has_package_json = (path / "package.json").exists()

    if has_pom or has_gradle:
        return "backend"
    elif has_package_json:
        return "mfe"
    else:
        logger.info(f"Repo {path.name} matches neither JVM nor Node indicators. Defaulting to docs_only.")
        return "docs_only"


def validate_domain_against_manifest(repo_name: str, detected_domain: str):
    """Cross-checks auto-detected domain against repo_manifest.yaml if present[cite: 1]."""
    manifest_path = Path(settings.REPO_MANIFEST_PATH)
    if not manifest_path.exists():
        return

    try:
        with open(manifest_path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
            repos = data.get("repos", [])
            for r in repos:
                if r.get("name") == repo_name:
                    declared_domain = r.get("domain")
                    if declared_domain and declared_domain != detected_domain:
                        logger.warning(
                            f"[MANIFEST MISMATCH] Repo '{repo_name}': Manifest declares '{declared_domain}', "
                            f"but auto-detector found '{detected_domain}'. Using auto-detected '{detected_domain}'."
                        )
                    break
    except Exception as e:
        logger.warning(f"Could not parse manifest for domain cross-check: {e}")


class MainOrchestrator:
    def __init__(self, max_budget_usd: Optional[float] = None):
        if max_budget_usd is not None:
            settings.MAX_BUDGET_USD = max_budget_usd

        self.token_tracker = TokenTrackerV2()
        self.approval_gate = CLIApprovalGate()
        self.manifest_tracker = RunManifestTracker()

        # Instantiate agents
        self.planning_agent = PlanningAgent(self.token_tracker, self.approval_gate)
        self.estimating_agent = EstimatingAgent(self.token_tracker)
        self.developer_agent = DeveloperAgent(self.token_tracker, self.approval_gate)
        self.doc_auditor_agent = RepoDocAuditorAgent(self.token_tracker, self.approval_gate)
        self.security_agent = SecurityAgent(self.token_tracker, self.approval_gate)
        self.architect_agent = ArchitectAgent(self.token_tracker, self.approval_gate)
        self.pr_review_agent = PRReviewAgent(self.token_tracker, self.approval_gate)
        self.qa_agent = QAAgentSuite(self.token_tracker, self.approval_gate)
        self.mfe_auditor_agent = MFEAuditorAgent(self.token_tracker, self.approval_gate)

        # Knowledge Base builders
        self.backend_kb_builder = FRAMLKnowledgeBaseBuilder(self.token_tracker)
        self.mfe_kb_builder = MFEKnowledgeBaseBuilder(self.token_tracker)

    def determine_scheduled_stages(self, domain: str, has_ticket_or_task: bool, skip_qa: bool) -> List[str]:
        """Determines ordered stage list based on repo domain classification[cite: 1]."""
        if domain == "docs_only":
            return ["Doc Audit"]

        stages = ["Knowledge Base", "Planning"]
        if has_ticket_or_task:
            stages.append("Developer")
        
        stages.extend(["Doc Audit", "Security"])
        
        if domain == "backend":
            stages.append("Architect")
        elif domain == "mfe":
            stages.append("MFE Governance Audit")

        stages.append("PR Review")
        
        if not skip_qa:
            stages.append("QA")

        return stages

    def execute_pipeline(
        self,
        repo_paths: List[str],
        fix_mode: bool = False,
        refactor_mode: bool = False,
        skip_qa: bool = False,
        base_branch: str = "Developer",
        dry_run: bool = False,
        discover_related_repos: bool = False,
        jira_id: Optional[str] = None,
        jira_ids: Optional[List[str]] = None,
        task: Optional[str] = None,
        bug: Optional[str] = None,
        stack_trace_file: Optional[str] = None,
        jira_target_repo: Optional[str] = None,
        jira_related_repos: Optional[List[str]] = None,
    ):
        logger.info("=== Starting Agentic SDLC Orchestration Pass ===")

        # Resolve primary target repo for feature/bug flags if multiple repos provided
        has_ticket_or_task = bool(jira_id or jira_ids or task or bug)
        primary_dev_target_path = None

        if has_ticket_or_task:
            if len(repo_paths) == 1:
                primary_dev_target_path = os.path.abspath(repo_paths[0])
            elif jira_target_repo:
                primary_dev_target_path = os.path.abspath(jira_target_repo)
            else:
                logger.error(
                    "Multiple repositories provided with ticket/task flags, but --jira-target-repo was not specified. "
                    "Please specify --jira-target-repo to disambiguate the Developer stage target."
                )
                sys.exit(1)

        # Build repository execution configurations
        repo_configs = []
        for path_str in repo_paths:
            abs_path = os.path.abspath(path_str)
            repo_name = os.path.basename(abs_path)
            domain = classify_repository_domain(abs_path)
            validate_domain_against_manifest(repo_name, domain)

            is_dev_target = (abs_path == primary_dev_target_path) if has_ticket_or_task else False
            scheduled_stages = self.determine_scheduled_stages(
                domain=domain,
                has_ticket_or_task=is_dev_target,
                skip_qa=skip_qa
            )

            repo_configs.append({
                "name": repo_name,
                "path": abs_path,
                "domain": domain,
                "is_dev_target": is_dev_target,
                "stages": scheduled_stages
            })

            self.manifest_tracker.repos_processed.append({
                "name": repo_name,
                "path": abs_path,
                "domain": domain
            })

        # --- Stage 1: Knowledge Base Indexing ---
        logger.info("\n>>> [STAGE 1/8] Knowledge Base Indexing <<<")
        for r in repo_configs:
            if r["domain"] == "backend":
                self.backend_kb_builder.scan_repo(r["path"], r["name"])
            elif r["domain"] == "mfe":
                self.mfe_kb_builder.scan_repo(r["path"], r["name"])
            else:
                logger.info(f"Skipping Knowledge Base indexing for docs_only repo {r['name']}.")

        # --- Stage 2: Planning Stage & Macro Approval Gate ---
        logger.info("\n>>> [STAGE 2/8] Planning Stage (Macro Gate) <<<")
        try:
            task_summary_text = task or bug or jira_id or (", ".join(jira_ids) if jira_ids else None)
            plan_file = self.planning_agent.generate_plan(
                repo_configs=repo_configs,
                jira_id=jira_id,
                task=task_summary_text,
                discover_related=discover_related_repos
            )
            self.manifest_tracker.macro_plan_path = plan_file
            self.manifest_tracker.record_artifact(plan_file)
        except RuntimeError as e:
            logger.error(f"Pipeline execution halted at Planning Stage: {e}")
            self.manifest_tracker.finalize(self.token_tracker.total_cost())
            return

        # --- Stages 3 to 8: Sequential Execution per Repository ---
        try:
            for r in repo_configs:
                repo_path = r["path"]
                repo_name = r["name"]
                domain = r["domain"]
                stages = r["stages"]

                logger.info(f"\n==================================================")
                logger.info(f" Processing Repository: {repo_name} [{domain.upper()}]")
                logger.info(f" Scheduled Stages: {', '.join(stages)}")
                logger.info(f"==================================================")

                # --- Stage 3: Developer Stage ---
                if "Developer" in stages and r["is_dev_target"]:
                    logger.info(f"\n>>> [STAGE 3] Developer Implementation on {repo_name} <<<")
                    dev_success = self.developer_agent.run(
                        repo_path=repo_path,
                        jira_id=jira_id,
                        task=task,
                        bug=bug,
                        stack_trace_file=stack_trace_file,
                        related_repos=jira_related_repos,
                        dry_run=dry_run
                    )
                    self.manifest_tracker.stage_outcomes.append({
                        "repo": repo_name,
                        "stage": "Developer",
                        "success": dev_success
                    })

                # --- Stage 4: Doc Audit Stage ---
                if "Doc Audit" in stages:
                    logger.info(f"\n>>> [STAGE 4] Documentation Audit on {repo_name} <<<")
                    doc_success = self.doc_auditor_agent.run(repo_path, domain=domain, dry_run=dry_run)
                    self.manifest_tracker.stage_outcomes.append({
                        "repo": repo_name,
                        "stage": "Doc Audit",
                        "success": doc_success
                    })

                # --- MFE Governance Audit (MFE Domain specific) ---
                if "MFE Governance Audit" in stages:
                    logger.info(f"\n>>> [STAGE 4.5] MFE Governance Audit on {repo_name} <<<")
                    mfe_success = self.mfe_auditor_agent.run(repo_path, dry_run=dry_run)
                    self.manifest_tracker.stage_outcomes.append({
                        "repo": repo_name,
                        "stage": "MFE Governance Audit",
                        "success": mfe_success
                    })

                # --- Stage 5: Security Stage ---
                if "Security" in stages:
                    logger.info(f"\n>>> [STAGE 5] Security Audit & Patching on {repo_name} <<<")
                    sec_success = self.security_agent.run(repo_path, fix_mode=fix_mode, dry_run=dry_run)
                    self.manifest_tracker.stage_outcomes.append({
                        "repo": repo_name,
                        "stage": "Security",
                        "success": sec_success
                    })

                # --- Stage 6: Architect Stage ---
                if "Architect" in stages:
                    logger.info(f"\n>>> [STAGE 6] Architecture Audit on {repo_name} <<<")
                    arch_success = self.architect_agent.run(repo_path, refactor_mode=refactor_mode, dry_run=dry_run)
                    self.manifest_tracker.stage_outcomes.append({
                        "repo": repo_name,
                        "stage": "Architect",
                        "success": arch_success
                    })

                # --- Stage 7: PR Review Stage ---
                if "PR Review" in stages:
                    logger.info(f"\n>>> [STAGE 7] PR Review & Contract Audit on {repo_name} <<<")
                    pr_success = self.pr_review_agent.run(repo_path, base_branch=base_branch, dry_run=dry_run)
                    self.manifest_tracker.stage_outcomes.append({
                        "repo": repo_name,
                        "stage": "PR Review",
                        "success": pr_success
                    })

                # --- Stage 8: QA Stage ---
                if "QA" in stages:
                    logger.info(f"\n>>> [STAGE 8] QA & Component Testing on {repo_name} <<<")
                    qa_success = self.qa_agent.run(repo_path, domain=domain, dry_run=dry_run)
                    self.manifest_tracker.stage_outcomes.append({
                        "repo": repo_name,
                        "stage": "QA",
                        "success": qa_success
                    })

        except BudgetExceededError as be:
            logger.error(f"\n[BUDGET CAP EXCEEDED] Pipeline halted: {be}")
        except Exception as ex:
            logger.error(f"\n[UNHANDLED ERROR] Pipeline execution encountered an error: {ex}", exc_info=True)
        finally:
            self.token_tracker.print_cost_report()
            manifest_file = self.manifest_tracker.finalize(self.token_tracker.total_cost())
            logger.info(f"Orchestrator run complete. Manifest saved to {manifest_file}")


def main():
    parser = argparse.ArgumentParser(description="Master Orchestrator for Agentic SDLC Suite")
    parser.add_argument("--repos", nargs="+", required=True, help="Target repository directory paths")
    parser.add_argument("--fix", action="store_true", help="Enable SecurityAgent --fix mode")
    parser.add_argument("--refactor", action="store_true", help="Enable ArchitectAgent --refactor mode")
    parser.add_argument("--skip-qa", action="store_true", help="Skip QA stage executions")
    parser.add_argument("--base-branch", default="Developer", help="Base git branch for PR review diffs")
    parser.add_argument("--dry-run", action="store_true", help="Generate prompts/diffs without disk writes")
    parser.add_argument("--max-budget-usd", type=float, help="Override system MAX_BUDGET_USD budget cap")
    parser.add_argument("--discover-related-repos", action="store_true", help="Opt-in knowledge-graph candidate discovery")

    # Ticket & Task flags (mutually exclusive group)
    task_group = parser.add_mutually_exclusive_group()
    task_group.add_argument("--jira-id", type=str, help="Single Jira ticket key (e.g., FRAML-1234)")
    task_group.add_argument("--jira-ids", nargs="+", help="Multiple Jira ticket keys")
    task_group.add_argument("--task", type=str, help="Inline feature requirement description")
    task_group.add_argument("--bug", type=str, help="Inline bug description")

    parser.add_argument("--stack-trace-file", type=str, help="Path to stack trace file for bug fixes")
    parser.add_argument("--jira-target-repo", type=str, help="Path to explicit target repo for ticket/task")
    parser.add_argument("--jira-related-repos", nargs="+", help="Informational list of related repo names")

    args = parser.parse_args()

    orchestrator = MainOrchestrator(max_budget_usd=args.max_budget_usd)
    orchestrator.execute_pipeline(
        repo_paths=args.repos,
        fix_mode=args.fix,
        refactor_mode=args.refactor,
        skip_qa=args.skip_qa,
        base_branch=args.base_branch,
        dry_run=args.dry_run,
        discover_related_repos=args.discover_related_repos,
        jira_id=args.jira_id,
        jira_ids=args.jira_ids,
        task=args.task,
        bug=args.bug,
        stack_trace_file=args.stack_trace_file,
        jira_target_repo=args.jira_target_repo,
        jira_related_repos=args.jira_related_repos,
    )


if __name__ == "__main__":
    main()