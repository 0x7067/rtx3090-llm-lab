import { z } from "zod";
import type { ActorContext } from "./actor.js";
import type { Document } from "./types.js";

// Basis points in the rotated image frame; original bytes are never replaced.
export const readingRecipeSchema = z.object({
  rotation: z.union([z.literal(0), z.literal(90), z.literal(180), z.literal(270)]),
  x: z.number().int().min(0).max(9900),
  y: z.number().int().min(0).max(9900),
  width: z.number().int().min(100).max(10000),
  height: z.number().int().min(100).max(10000),
}).strict().refine(v => v.x + v.width <= 10000 && v.y + v.height <= 10000,
  "O recorte deve permanecer dentro da imagem girada.");
export type ReadingRecipe = z.infer<typeof readingRecipeSchema>;
export const fullReadingRecipe: ReadingRecipe = { rotation: 0, x: 0, y: 0, width: 10000, height: 10000 };
export const readingCommandSchema = z.object({
  version: z.number().int().positive(),
  documentVersion: z.number().int().min(0),
  originalSha256: z.string().min(1).max(128),
  recipe: readingRecipeSchema.nullable(),
  reason: z.string().trim().min(5).max(1500),
  requestKey: z.uuid(),
}).strict();
export interface DocumentReadingView {
  version: number;
  recipe: ReadingRecipe | null;
  originalSha256: string;
  actor: ActorContext;
  reason: string;
  at: string;
}
export interface ReadingRevision {
  id: number;
  documentId: string;
  before: DocumentReadingView | null;
  after: DocumentReadingView;
}
export interface DocumentReadingWorkspace {
  document: Document;
  expenseVersion: number;
  editable: boolean;
  history: ReadingRevision[];
}
export function readingRasterPlan(width: number, height: number, raw: ReadingRecipe) {
  const recipe = readingRecipeSchema.parse(raw);
  if (!Number.isSafeInteger(width) || !Number.isSafeInteger(height) || width < 1 || height < 1 ||
      width > 16000 || height > 16000 || width * height > 40_000_000)
    throw Error("A imagem excede o limite de leitura: até 16000 px por lado e 40 milhões de pixels. O original continua disponível.");
  const rotatedWidth = recipe.rotation % 180 ? height : width;
  const rotatedHeight = recipe.rotation % 180 ? width : height;
  const x = Math.floor(rotatedWidth * recipe.x / 10000);
  const y = Math.floor(rotatedHeight * recipe.y / 10000);
  const cropWidth = Math.max(1, Math.ceil(rotatedWidth * (recipe.x + recipe.width) / 10000) - x);
  const cropHeight = Math.max(1, Math.ceil(rotatedHeight * (recipe.y + recipe.height) / 10000) - y);
  const scale = Math.min(1, 4096 / cropWidth, 4096 / cropHeight, Math.sqrt(6_000_000 / (cropWidth * cropHeight)));
  return { x, y, cropWidth, cropHeight, outputWidth: Math.max(1, Math.floor(cropWidth * scale)), outputHeight: Math.max(1, Math.floor(cropHeight * scale)) };
}
export function readingDrag(start: { x: number; y: number }, end: { x: number; y: number }) {
  const bound = (v: number) => Math.max(0, Math.min(10000, Math.round(v)));
  const x = Math.min(bound(start.x), bound(end.x)), y = Math.min(bound(start.y), bound(end.y));
  const width = Math.abs(bound(end.x) - bound(start.x)), height = Math.abs(bound(end.y) - bound(start.y));
  return width < 100 || height < 100 ? null : { x, y, width, height };
}
