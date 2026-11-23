import { z } from "zod";
import type { ActorContext } from "./actor.js";
import type { CardTransaction } from "./cards.js";
export type CardCaptureFacts = Pick<
  CardTransaction,
  | "id"
  | "cardId"
  | "owner"
  | "ownerMemberId"
  | "merchant"
  | "category"
  | "amount"
  | "date"
  | "origin"
  | "sourceRef"
  | "controls"
>;
export interface CardCapturePortion {
  id: string;
  statementId: string;
  authorizationId: string;
  amount: number;
  statement: CardCaptureFacts;
  authorization: CardCaptureFacts;
  statementVersion: number;
  authorizationVersion: number;
  actor: ActorContext;
  reason: string;
  createdAt: string;
  state: "active" | "released";
  releasedAt?: string;
  releasedBy?: ActorContext;
  releaseReason?: string;
}
export interface CardCaptureCoverage {
  reserved: number;
  matched: number;
  remaining: number;
  available: number;
  stale: number;
  overReserved: boolean;
  active: number;
  legacy: boolean;
}
export interface CardCaptureReview extends CardCaptureCoverage {
  transaction: CardTransaction;
  basisToken: string;
  links: {
    portion: CardCapturePortion;
    current: boolean;
    issues: string[];
    counterpart: CardTransaction | null;
  }[];
  candidates: {
    transaction: CardTransaction;
    coverage: CardCaptureCoverage;
    maximum: number;
  }[];
  issues: string[];
  bankConfirmed: false;
}
const common = {
  version: z.number().int().positive(),
  basisToken: z.string().regex(/^[a-f0-9]{64}$/),
  reason: z.string().trim().min(10).max(1500),
  requestKey: z.string().uuid(),
};
export const cardCaptureCommandSchema = z.discriminatedUnion("action", [
  z
    .object({
      ...common,
      action: z.literal("link"),
      counterpartId: z.string().uuid(),
      counterpartVersion: z.number().int().positive(),
      amount: z.number().int().positive().max(100_000_000),
      understood: z.literal(true),
    })
    .strict(),
  z
    .object({
      ...common,
      action: z.literal("release"),
      portionId: z.string().uuid(),
    })
    .strict(),
]);
export type CardCaptureCommand = z.infer<typeof cardCaptureCommandSchema>;
