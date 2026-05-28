from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from swe_agent.parallel_utils import _normalize_messages, gap_corr

from .contracts import ExportSample, SFTExportBundle


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


class _RunArtifacts:
    def __init__(self, run_dir: Path) -> None:
        self.run_dir = run_dir
        self.manifest = _load_json(run_dir / "run_manifest.json")
        self.instance_id = str(self.manifest["instance_id"])
        self.system_prompt = str(self.manifest.get("system_prompt"))
        self.user_prompt = str(self.manifest.get("user_prompt"))
        self.nodes: dict[str, dict[str, Any]] = {}
        self.node_messages: dict[str, list[dict[str, str]]] = {}
        self.rounds: dict[int, dict[str, Any]] = {}
        self.rubric_messages: dict[str, list[dict[str, str]]] = {}

        node_index_path = run_dir / "node_index.jsonl"
        for raw_line in node_index_path.read_text(encoding="utf-8").splitlines():
            line = raw_line.strip()
            payload = json.loads(line)
            self.nodes[str(payload["node_id"])] = payload

        for node_id in sorted(self.nodes):
            node_dir = run_dir / "nodes" / node_id
            messages_path = node_dir / "messages.json"
            payload = _load_json(messages_path)
            self.node_messages[node_id] = _normalize_messages(payload)
            judge_path = node_dir / "judge.json"
            payload = _load_json(judge_path)
            self.nodes[node_id]["ground_truth_reward"] = payload.get("ground_truth_reward", 0.0)

        rubrics_dir = run_dir / "rubrics"
        if rubrics_dir.exists():
            for round_path in sorted(rubrics_dir.glob("round_*.json")):
                payload = _load_json(round_path)
                self.rounds[int(payload["round_index"])] = {rubric["rubric_list_id"]: rubric for rubric in payload["rubric_samples"]}

            for rubric_dir in sorted(path for path in rubrics_dir.iterdir() if path.is_dir()):
                rubric_id = rubric_dir.name
                messages_path = rubric_dir / "messages.json"
                payload = _load_json(messages_path)
                self.rubric_messages[rubric_id] = _normalize_messages(payload)

    def build_prefix_messages(self, parent_node_id: str | None) -> list[dict[str, str]]:
        messages: list[dict[str, str]] = []
        if self.system_prompt:
            messages.append({"role": "system", "content": self.system_prompt})
        if self.user_prompt:
            messages.append({"role": "user", "content": self.user_prompt})

        lineage: list[str] = []
        current = parent_node_id
        while current and current != "root":
            lineage.append(current)
            current = self.nodes[current].get("parent_id")
        for node_id in reversed(lineage):
            messages.extend(self.node_messages.get(node_id))
        return messages

    def _is_teacher_gt_node(self, node_id: str) -> bool:
        node = self.nodes.get(node_id)
        if not node or str(node.get("policy_source")) != "teacher":
            return False
        return float(node.get("ground_truth_reward")) >= 1.0

    def build_policy_sft_sample(self, round_index: int) -> tuple[str, list[str], ExportSample | None]:
        valid_rubrics = [rubric for rubric in self.rounds[round_index].values() if rubric.get("is_valid") is not False]
        if not valid_rubrics:
            return f"round_{round_index}", [], None
        first_rubric = valid_rubrics[0]
        group_id = first_rubric["parent_node_id"] if round_index in self.rounds else f"round_{round_index}"
        node_ids = list(first_rubric["child_rewards"].keys())
        teacher_nodes = [node_id for node_id in node_ids if self._is_teacher_gt_node(node_id)]
        if not teacher_nodes:
            return group_id, node_ids, None
        node_id = teacher_nodes[0]
        node = self.nodes[node_id]
        return group_id, node_ids, ExportSample(
            sample_id=node_id,
            group_id=group_id,
            prompt=self.build_prefix_messages(node.get("parent_id")),
            turns=self.node_messages[node_id],
        )

    def _rubic_judge_corr_gt(self, node_ids: list[str], rubric: dict[str, Any]) -> bool:
        rewards = [rubric.get("child_rewards").get(node_id) for node_id in node_ids]
        gts = [self.nodes[node_id].get("ground_truth_reward") for node_id in node_ids]
        return gap_corr(rewards, gts) > 0.8

    def build_rubric_sft_sample(self, round_index: int, group_id: str, node_ids: list[str]) -> ExportSample | None:
        teacher_nodes = [node_id for node_id in node_ids if self._is_teacher_gt_node(node_id)]
        if not teacher_nodes:
            return None

        for rubric in self.rounds[round_index].values():
            if rubric.get("is_valid") is False:
                continue
            if rubric.get("format_errors") or rubric.get("terminal_error"):
                continue
            if not self._rubic_judge_corr_gt(node_ids, rubric):
                continue
            rubric_list_id = str(rubric.get("rubric_list_id") or "")
            return ExportSample(
                sample_id=rubric_list_id,
                group_id=group_id,
                prompt=self.rubric_messages[rubric_list_id][:1],
                turns=self.rubric_messages[rubric_list_id][1:],
            )
        return None


class SFTDataExporter:
    def __init__(self, *, run_dir: Path) -> None:
        self.artifacts = _RunArtifacts(run_dir)

    def export_bundle(self) -> SFTExportBundle:
        accepted_group_ids: list[str] = []
        policy_samples: list[ExportSample] = []
        rubric_samples: list[ExportSample] = []

        for round_index in range(1, len(self.artifacts.rounds) + 1):
            group_id, node_ids, policy_sample = self.artifacts.build_policy_sft_sample(round_index)
            if policy_sample is None:
                continue
            rubric_sample = self.artifacts.build_rubric_sft_sample(round_index, group_id, node_ids)
            if rubric_sample is None:
                continue
            accepted_group_ids.append(group_id)
            rubric_samples.append(rubric_sample)
            policy_samples.append(policy_sample)

        return SFTExportBundle(
            instance_id=self.artifacts.instance_id,
            run_dir=str(self.artifacts.run_dir),
            accepted_group_ids=accepted_group_ids,
            policy_samples=policy_samples,
            rubric_samples=rubric_samples,
        )


# ---------------------------------------------------------------------------
# Parallel-runner SFT exporter (task 5)
# ---------------------------------------------------------------------------


class _ParallelRunArtifacts:
    """Reads trajectory_search_parallel artifacts off disk.

    Disk layout (per instance run_dir):
      instance_record.json
      config.json
      groups/group_NNN/
        members.json              -- branch index + parent MidCp
        shared_parent_raw.json    -- Lane A messages up to MidCp (with token fields)
        rubric_samples.json       -- full-parity rubric pipeline output (task 2)
        judge_response.json       -- {rubric_key: {branch_key: {score_normalized}}}
        branches/branch_BB/
          continuation_raw.json   -- Lane B's new messages (with token fields)
          gt.json                 -- {gt_score, gt_payload}
          summary.json
          terminal_patch.txt
    """

    def __init__(self, run_dir: Path) -> None:
        self.run_dir = run_dir
        record_path = run_dir / "instance_record.json"
        if not record_path.exists():
            raise FileNotFoundError(f"{record_path}: parallel runner did not finish")
        self.instance_record = _load_json(record_path)
        self.instance_id = str(self.instance_record["instance_id"])
        # Per-group payloads keyed by group_index
        self.groups: dict[int, dict[str, Any]] = {}
        groups_dir = run_dir / "groups"
        if groups_dir.exists():
            for gdir in sorted(groups_dir.iterdir()):
                if not gdir.is_dir():
                    continue
                try:
                    gid = int(gdir.name.removeprefix("group_"))
                except ValueError:
                    continue
                payload: dict[str, Any] = {"group_index": gid, "group_dir": gdir}
                members_path = gdir / "members.json"
                if members_path.exists():
                    payload["members"] = _load_json(members_path)
                parent_path = gdir / "shared_parent_raw.json"
                if parent_path.exists():
                    payload["parent_messages"] = json.loads(
                        parent_path.read_text(encoding="utf-8")
                    )
                rubric_samples_path = gdir / "rubric_samples.json"
                if rubric_samples_path.exists():
                    payload["rubric_samples_blob"] = _load_json(rubric_samples_path)
                judge_path = gdir / "judge_response.json"
                if judge_path.exists():
                    payload["judge_response"] = _load_json(judge_path)
                branches_dir = gdir / "branches"
                branches: list[dict[str, Any]] = []
                if branches_dir.exists():
                    for bdir in sorted(branches_dir.iterdir()):
                        if not bdir.is_dir():
                            continue
                        try:
                            bi = int(bdir.name.removeprefix("branch_"))
                        except ValueError:
                            continue
                        cont_path = bdir / "continuation_raw.json"
                        gt_path = bdir / "gt.json"
                        summary_path = bdir / "summary.json"
                        branch = {
                            "branch_index": bi,
                            "messages": (
                                json.loads(cont_path.read_text(encoding="utf-8"))
                                if cont_path.exists() else []
                            ),
                            "gt": (
                                _load_json(gt_path) if gt_path.exists() else {}
                            ),
                            "summary": (
                                _load_json(summary_path) if summary_path.exists() else {}
                            ),
                        }
                        branches.append(branch)
                payload["branches"] = branches
                self.groups[gid] = payload


class ParallelSFTDataExporter:
    """SFT data exporter for the v1-lanes parallel search.

    Mirrors SFTDataExporter's two-stage filter for the parallel topology:
      * policy sample = the branch in the group whose GT reward >= 1.0
        (i.e. the patch solved the problem). One per fork-group, picked
        deterministically (lowest branch_index among passing branches).
      * rubric sample = the rubric_samples entry whose rubric judge scores
        correlate with GT (gap_corr > 0.8) across the group's branches.
        Mirrors data_export._rubic_judge_corr_gt.

    A fork-group is accepted only when BOTH a passing policy sample and a
    well-correlated rubric sample exist. Otherwise the group is dropped —
    same fail-loud discipline as the sequential exporter.
    """

    GT_PASS_THRESHOLD = 1.0
    RUBRIC_CORR_THRESHOLD = 0.8

    def __init__(self, *, run_dir: Path) -> None:
        self.artifacts = _ParallelRunArtifacts(run_dir)

    def _rubric_judge_corr_gt(
        self,
        sample_payload: dict[str, Any],
        branches: list[dict[str, Any]],
    ) -> bool:
        # sample_payload["child_rewards"] is keyed by LaneBBranch.node_id
        # (not branch_index) — Lane C drops errored / round1-empty branches
        # from `valid_branches` before populating child_rewards, so its
        # keys are a SUBSET of the on-disk `branches`. Join on node_id
        # via each branch's summary.json (which records branch.node_id at
        # dump time). Falls back to skipping the sample if we can't form
        # at least 2 (rubric_score, gt) pairs.
        child_rewards = sample_payload.get("child_rewards") or {}
        if not child_rewards:
            return False
        per_branch_rewards: list[float] = []
        gts: list[float] = []
        for branch in branches:
            node_id = (branch.get("summary") or {}).get("node_id")
            if not node_id or node_id not in child_rewards:
                continue
            gt = (branch.get("gt") or {}).get("gt_score")
            if gt is None:
                continue
            try:
                per_branch_rewards.append(float(child_rewards[node_id]))
                gts.append(float(gt))
            except (TypeError, ValueError):
                continue
        if len(per_branch_rewards) < 2:
            return False
        # Constant-GT special case: on easy instances DSv4 reliably solves
        # all branches → gts=[1,1,1,...,1]. Pearson correlation is
        # undefined here (zero variance in denominator → NaN). On 30 easy
        # SWE-rebench instances we hit this on copier-org__copier-1998
        # (8/8 branches GT=1.0) and lost the entire group to filtering.
        # Accept the rubric IF (a) it's well-formed (caller already
        # gated on no format_errors / terminal_error) AND (b) rubric
        # rewards are themselves not perfectly constant — perfectly-flat
        # rubrics give no learning signal so we still drop those.
        if len(set(gts)) <= 1:
            return len(set(per_branch_rewards)) > 1
        return gap_corr(per_branch_rewards, gts) > self.RUBRIC_CORR_THRESHOLD

    def _build_policy_samples(
        self,
        group_id: str,
        group_index: int,
        parent_messages: list[dict[str, Any]],
        branches: list[dict[str, Any]],
    ) -> list[ExportSample]:
        """One ExportSample per GT-passing branch in the group.

        Earlier this kept only `passing[0]`, throwing away up to m-1=7
        equally-passing branches per group. On easy instances where all
        8 branches GT-pass that's an 8× reduction in policy samples.
        Each passing branch gives a distinct SFT example (same prompt,
        different terminal turns) so we keep them all.
        """
        out: list[ExportSample] = []
        for branch in branches:
            gt = (branch.get("gt") or {}).get("gt_score")
            if gt is None or float(gt) < self.GT_PASS_THRESHOLD:
                continue
            if not branch.get("messages"):
                continue
            out.append(ExportSample(
                sample_id=f"{self.artifacts.instance_id}-g{group_index:03d}-b{branch['branch_index']:02d}",
                group_id=group_id,
                prompt=_normalize_messages(parent_messages),
                turns=_normalize_messages(branch["messages"]),
            ))
        return out

    def _build_rubric_samples(
        self,
        group_id: str,
        rubric_samples_blob: dict[str, Any],
        branches: list[dict[str, Any]],
    ) -> list[ExportSample]:
        """One ExportSample per valid rubric draft in the group.

        Earlier this returned on the FIRST draft that passed the filter,
        throwing away the other N-1 valid drafts. Each draft is an
        independent rubric-model generation conditioned on the same
        parent state — distinct training examples. Keep all that pass
        the well-formedness + correlation gate.
        """
        out: list[ExportSample] = []
        samples = (rubric_samples_blob or {}).get("rubric_samples") or []
        for sample in samples:
            if sample.get("format_errors") or sample.get("terminal_error"):
                continue
            if not self._rubric_judge_corr_gt(sample, branches):
                continue
            messages = sample.get("messages") or []
            if len(messages) < 2:
                continue
            out.append(ExportSample(
                sample_id=str(sample.get("rubric_list_id") or f"rubric-{group_id}"),
                group_id=group_id,
                prompt=messages[:1],
                turns=messages[1:],
            ))
        return out

    def export_bundle(self) -> SFTExportBundle:
        """Decoupled policy + rubric SFT capture (per user 2026-05-21).

        Policy and rubric sample acceptance are now INDEPENDENT:
          * policy_samples: one per GT-passing (gt>=1.0) branch.
            Strict gate — only truly-solved problems become policy SFT.
          * rubric_samples: one per validity-passing rubric draft (no
            format/terminal err + judge scores correlate with GT, or
            have variance in the constant-GT case). NOT gated on the
            group having any GT-pass branch — even groups where ALL
            branches failed contribute rubric data if the rubric model
            differentiated the failing branches meaningfully.

        accepted_group_ids: groups that contributed AT LEAST one sample
        (policy OR rubric). Per-bundle metadata records both counts.
        """
        accepted_group_ids: list[str] = []
        policy_samples: list[ExportSample] = []
        rubric_samples: list[ExportSample] = []
        n_groups_with_policy = 0
        n_groups_with_rubric = 0
        for gid in sorted(self.artifacts.groups):
            payload = self.artifacts.groups[gid]
            branches = payload.get("branches") or []
            parent_messages = payload.get("parent_messages") or []
            rubric_samples_blob = payload.get("rubric_samples_blob") or {}
            group_id = f"g{gid:03d}"
            pols = self._build_policy_samples(
                group_id, gid, parent_messages, branches,
            )
            rubs = self._build_rubric_samples(
                group_id, rubric_samples_blob, branches,
            )
            if pols:
                n_groups_with_policy += 1
                policy_samples.extend(pols)
            if rubs:
                n_groups_with_rubric += 1
                rubric_samples.extend(rubs)
            if pols or rubs:
                accepted_group_ids.append(group_id)
        return SFTExportBundle(
            instance_id=self.artifacts.instance_id,
            run_dir=str(self.artifacts.run_dir),
            accepted_group_ids=accepted_group_ids,
            policy_samples=policy_samples,
            rubric_samples=rubric_samples,
            metadata={
                "search_scheme": "lane_v1_parallel",
                "num_groups": len(self.artifacts.groups),
                "num_accepted_groups": len(accepted_group_ids),
                "num_groups_with_policy": n_groups_with_policy,
                "num_groups_with_rubric": n_groups_with_rubric,
                "num_policy_samples": len(policy_samples),
                "num_rubric_samples": len(rubric_samples),
            },
        )
