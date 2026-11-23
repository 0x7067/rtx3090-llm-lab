import type { AmendmentBlocker } from "./purchase-amendments.js";
import type { PricedItem } from "./item-quantities.js";

export type AmendmentDependencyKind = AmendmentBlocker["kind"];
export interface AmendmentCompatibilityIssue {
  code:
    | "scope_changed"
    | "line_changed"
    | "quantity_floor"
    | "financial_floor"
    | "stale_evidence"
    | "invalid_evidence";
  message: string;
  itemIndex?: number;
  kind?: AmendmentDependencyKind;
  id?: string;
}
/** A read-only assessment; this never authorizes applying a change. */
export interface PurchaseAmendmentCompatibility {
  scopeCompatible: boolean;
  dependencyToken: string;
  originalAmount: number;
  proposedAmount: number;
  monetaryFloor: number;
  lines: {
    itemIndex: number;
    source: PricedItem;
    proposed: PricedItem | null;
    pinned: boolean;
    receivedFloor: number;
    invoicedFloor: number;
    minimumQuantity: number;
  }[];
  retained: {
    kind: AmendmentDependencyKind;
    id: string;
    state: string;
    token: string;
  }[];
  issues: AmendmentCompatibilityIssue[];
}
