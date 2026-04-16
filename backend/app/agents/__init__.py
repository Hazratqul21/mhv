from __future__ import annotations

from .base_agent import AgentResult, BaseAgent

# ── Core Agents ──────────────────────────────────────────────────────────
from .chat_agent import ChatAgent
from .code_agent import CodeAgent
from .vision_agent import VisionAgent
from .voice_agent import VoiceAgent
from .search_agent import SearchAgent
from .tool_agent import ToolAgent

# ── Knowledge & Research ─────────────────────────────────────────────────
from .rag_agent import RAGAgent
from .research_agent import ResearchAgent
from .summarizer_agent import SummarizerAgent

# ── Language & Communication ─────────────────────────────────────────────
from .translator_agent import TranslatorAgent
from .email_agent import EmailAgent
from .creative_agent import CreativeAgent

# ── Technical ────────────────────────────────────────────────────────────
from .math_agent import MathAgent
from .data_agent import DataAgent
from .sql_agent import SQLAgent
from .shell_agent import ShellAgent
from .security_agent import SecurityAgent
from .devops_agent import DevOpsAgent
from .planner_agent import PlannerAgent

# ── Autonomous / Meta ────────────────────────────────────────────────────
from .meta_agent import MetaAgent
from .reflection_agent import ReflectionAgent
from .memory_manager_agent import MemoryManagerAgent
from .workflow_agent import WorkflowAgent
from .scheduler_agent import SchedulerAgent
from .monitor_agent import MonitorAgent
from .learning_agent import LearningAgent
from .collaboration_agent import CollaborationAgent

# ── Media ─────────────────────────────────────────────────────────────────
from .video_agent import VideoAgent
from .music_agent import MusicAgent
from .voice_clone_agent import VoiceCloneAgent
from .art_director_agent import ArtDirectorAgent

# ── PC Host / System ─────────────────────────────────────────────────────
from .system_admin_agent import SystemAdminAgent
from .gpu_manager_agent import GPUManagerAgent

# ── App Builder ──────────────────────────────────────────────────────────
from .app_builder_agent import AppBuilderAgent
from .frontend_agent import FrontendAgent
from .backend_agent import BackendAgent

# ── Fine-Tuning ──────────────────────────────────────────────────────────
from .finetune_agent import FineTuneAgent

# ── Specialized ──────────────────────────────────────────────────────────
from .api_agent import APIAgent
from .scraping_agent import ScrapingAgent
from .notification_agent import NotificationAgent
from .file_manager_agent import FileManagerAgent
from .image_gen_agent import ImageGenAgent
from .testing_agent import TestingAgent
from .document_agent import DocumentAgent
from .code_generation_agent import CodeGenerationAgent
from .code_generator_agent import CodeGeneratorAgent
from .code_debugger_agent import CodeDebuggerAgent
from .code_debugging_agent import CodeDebuggingAgent
from .code_generation_agent_agent import CodeGenerationAgentAgent
from .code_prompt_agent_agent import CodePromptAgentAgent
from .code_for_text_generation_agent import CodeForTextGenerationAgent
from .code_writing_agent_agent import CodeWritingAgentAgent
from .code_tutorial_agent import CodeTutorialAgent
from .code_optimization_agent import CodeOptimizationAgent
from .code_execution_agent import CodeExecutionAgent
from .code_generation_for_simple_tasks_agent import CodeGenerationForSimpleTasksAgent
from .code_explainer_agent import CodeExplainerAgent
from .naroscom_agent_agent import NaroscomAgentAgent

__all__ = [
    "AgentResult",
    "BaseAgent",
    # Core
    "ChatAgent",
    "CodeAgent",
    "VisionAgent",
    "VoiceAgent",
    "SearchAgent",
    "ToolAgent",
    # Knowledge & Research
    "RAGAgent",
    "ResearchAgent",
    "SummarizerAgent",
    # Language & Communication
    "TranslatorAgent",
    "EmailAgent",
    "CreativeAgent",
    # Technical
    "MathAgent",
    "DataAgent",
    "SQLAgent",
    "ShellAgent",
    "SecurityAgent",
    "DevOpsAgent",
    "PlannerAgent",
    # Autonomous / Meta
    "MetaAgent",
    "ReflectionAgent",
    "MemoryManagerAgent",
    "WorkflowAgent",
    "SchedulerAgent",
    "MonitorAgent",
    "LearningAgent",
    "CollaborationAgent",
    # Media
    "VideoAgent",
    "MusicAgent",
    "VoiceCloneAgent",
    "ArtDirectorAgent",
    # PC Host / System
    "SystemAdminAgent",
    "GPUManagerAgent",
    # App Builder
    "AppBuilderAgent",
    "FrontendAgent",
    "BackendAgent",
    # Fine-Tuning
    "FineTuneAgent",
    # Specialized
    "APIAgent",
    "ScrapingAgent",
    "NotificationAgent",
    "FileManagerAgent",
    "ImageGenAgent",
    "TestingAgent",
    "DocumentAgent",
    "CodeGenerationAgent",
    "CodeGeneratorAgent",
    "CodeDebuggerAgent",
    "CodeDebuggingAgent",
    "CodeGenerationAgentAgent",
    "CodePromptAgentAgent",
    "CodeForTextGenerationAgent",
    "CodeWritingAgentAgent",
    "CodeTutorialAgent",
    "CodeOptimizationAgent",
    "CodeExecutionAgent",
    "CodeGenerationForSimpleTasksAgent",
    "CodeExplainerAgent",
    "NaroscomAgentAgent",
]
