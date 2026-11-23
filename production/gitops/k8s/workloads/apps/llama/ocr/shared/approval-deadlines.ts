import { z } from "zod";
import type { ActorContext } from "./actor.js";
import type { ApprovalRun } from "./approvals.js";
import type { ApprovalReviewer } from "./approval-substitutions.js";
import {
  approvalConditionsSchema,
  approvalBusinessDay,
} from "./approval-conditions.js";
import { daysBetween } from "./subscriptions.js";

export const reviewDays = z.number().int().min(1).max(90);
const date = approvalConditionsSchema.shape.validFrom.unwrap();
const common = {
  runId: z.string().uuid(),
  stageIndex: z.number().int().min(0),
  stageVersion: z.number().int().min(0),
  asOf: date,
  sourceFingerprint: z.string().regex(/^[a-f0-9]{64}$/),
  dueOn: date,
  reason: z.string().trim().min(5).max(1500),
  requestKey: z.string().uuid(),
};
export const stageFollowUpCommand = z.discriminatedUnion("action", [
  z.object({ ...common, action: z.literal("declare") }).strict(),
  z
    .object({
      ...common,
      action: z.literal("handoff"),
      reviewerId: z.string().regex(/^[a-zA-Z0-9-]{1,80}$/),
      reviewerVersion: z.number().int().positive(),
    })
    .strict(),
]);
export interface StageAssignment {
  reviewer: ApprovalReviewer;
  dueOn: string;
}
export interface StageFollowUpEvent {
  id: string;
  action: "declare" | "handoff";
  before: StageAssignment | null;
  after: StageAssignment;
  version: number;
  asOf: string;
  overdueDays: number;
  sourceStale: boolean;
  actor: ActorContext;
  reason: string;
  at: string;
  notificationSent: false;
}
export interface StageReview {
  version: number;
  readyAt: string;
  assignee: ApprovalReviewer;
  dueOn: string;
  basis: "rule" | "declared";
  history: StageFollowUpEvent[];
  completion?: { at: string; overdueDays: number };
}
export const addReviewDays = (day: string, days: number) => {
  const d = new Date(day + "T12:00:00Z");
  d.setUTCDate(d.getUTCDate() + days);
  return d.toISOString().slice(0, 10);
};
export const currentStageIndex = (run: ApprovalRun) =>
  run.state === "pending"
    ? run.steps.findIndex((s) => s.state === "waiting")
    : -1;
export const stageAssigneeId = (run: ApprovalRun) => {
  const step = run.steps[currentStageIndex(run)];
  return step?.review?.assignee.memberId ?? step?.approver;
};
export interface StageDeadlineView {
  asOf: string;
  stageIndex: number;
  stageVersion: number;
  assigneeId: string | null;
  dueOn: string | null;
  state: "none" | "open" | "due_today" | "overdue" | "closed";
  overdueDays: number;
}
export function stageDeadline(
  run: ApprovalRun,
  asOf: string,
): StageDeadlineView {
  const stageIndex = currentStageIndex(run),
    step = run.steps[stageIndex];
  const dueOn = step?.review?.dueOn ?? null;
  const state =
    run.state !== "pending"
      ? "closed"
      : !dueOn
        ? "none"
        : dueOn < asOf
          ? "overdue"
          : dueOn === asOf
            ? "due_today"
            : "open";
  return {
    asOf,
    stageIndex,
    stageVersion: step?.review?.version ?? 0,
    assigneeId: stageAssigneeId(run) ?? null,
    dueOn,
    state,
    overdueDays: state === "overdue" ? daysBetween(dueOn!, asOf) : 0,
  };
}
export function beginStageReview(run: ApprovalRun, readyAt: Date) {
  const index = currentStageIndex(run),
    step = run.steps[index];
  if (!step || !run.reviewWithinDays || step.review) return;
  if (!step.reviewer)
    throw Error(
      "Uma etapa com prazo exige a identidade preservada do aprovador.",
    );
  step.review = {
    version: 1,
    readyAt: readyAt.toISOString(),
    assignee: step.reviewer,
    dueOn: addReviewDays(approvalBusinessDay(readyAt), run.reviewWithinDays),
    basis: "rule",
    history: [],
  };
}
export interface StageFollowUpContext {
  run: ApprovalRun;
  deadline: StageDeadlineView;
  reviewers: (ApprovalReviewer & { version: number })[];
  sourceStale: boolean;
  sourceFingerprint: string;
}
