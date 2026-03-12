from __future__ import annotations

from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, Field

SessionStatus = Literal["running", "paused", "finished", "failed"]


class ConversationMessage(BaseModel):
    role: str
    content: Any
    source: str = "conversation"
    trainable: bool = False
    metadata: Dict[str, Any] = Field(default_factory=dict)


class ModelTurn(BaseModel):
    session_id: str
    step_index: int
    query_messages: List[ConversationMessage]
    response_message: ConversationMessage
    reward: Optional[float] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)


class RolloutEvent(BaseModel):
    event_id: str
    session_id: str
    step_index: int
    kind: str
    ts: float
    payload: Dict[str, Any] = Field(default_factory=dict)
    provenance: Dict[str, Any] = Field(default_factory=dict)


class SessionComponentState(BaseModel):
    type_path: str
    config: Dict[str, Any] = Field(default_factory=dict)
    state: Dict[str, Any] = Field(default_factory=dict)


class RolloutSessionSpec(BaseModel):
    session_id: str
    task: str
    task_id: Optional[str] = None
    group_id: Optional[str] = None
    sample_index: int = 0
    policy_ref: str = "policy"
    policy_version: Optional[str] = None
    dataset_name: Optional[Any] = None
    ground_truth: Optional[Any] = None
    raw_user_query: Optional[str] = None
    limits: Dict[str, Any] = Field(default_factory=dict)
    pause_points: List[str] = Field(default_factory=list)
    metadata: Dict[str, Any] = Field(default_factory=dict)


class RolloutSnapshot(BaseModel):
    session_id: str
    status: SessionStatus
    spec: RolloutSessionSpec
    agent: SessionComponentState
    model: SessionComponentState
    environment: SessionComponentState
    memory: Optional[SessionComponentState] = None
    last_step_index: int = -1
    last_event_id: Optional[str] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)


class RolloutResult(BaseModel):
    session_id: str
    status: SessionStatus
    spec: RolloutSessionSpec
    events: List[RolloutEvent] = Field(default_factory=list)
    model_turns: List[ModelTurn] = Field(default_factory=list)
    final_messages: List[ConversationMessage] = Field(default_factory=list)
    exit_status: str = ""
    submission: str = ""
    metadata: Dict[str, Any] = Field(default_factory=dict)


class TrainingProjection(BaseModel):
    session_id: str
    group_id: Optional[str] = None
    policy_ref: str = "policy"
    policy_version: Optional[str] = None
    prompt_token_ids: List[int]
    continuation_token_ids: List[int]
    trainable_mask: List[int]
    finish_reason: str
    dataset_name: Optional[Any] = None
    ground_truth: Optional[Any] = None
    raw_user_query: Optional[str] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)
