import type { ActorContext } from "./actor.js";
import type {
  StatementAccount,
  StatementEntry,
  StatementMapping,
  StatementRow,
  ReconciliationTarget,
  ReconciliationMatch,
} from "./reconciliation.js";

export interface StatementImportStage {
  actor: ActorContext;
  at: string;
  version: number;
  financialToken: string;
}
export interface StatementOriginalEvidence {
  importId: string;
  accountId: string;
  filename: string;
  sourceHash: string;
  mapping: StatementMapping;
  rowIndex: number;
  row: StatementRow;
  preparation: StatementImportStage | null;
  confirmation: StatementImportStage | null;
}
export interface StatementMatchReview {
  match: ReconciliationMatch;
  currentTarget: ReconciliationTarget | null;
  originalDebit: {
    entryId: string;
    matchId: string;
    sourceHash: string;
    original: StatementOriginalEvidence;
    match: ReconciliationMatch;
  } | null;
  reviewers: ActorContext[];
  issues: string[];
}
/** Local correspondence only; neither a bank balance nor bank confirmation. */
export interface StatementEntryReview {
  entry: StatementEntry;
  account: StatementAccount;
  original: StatementOriginalEvidence | null;
  matches: StatementMatchReview[];
  issues: string[];
  matched: number;
  remaining: number;
  status: "matched" | "partial" | "unmatched" | "excluded" | "source-issue";
  fingerprint: string;
  integrityToken: string;
  bankConfirmed: false;
}
