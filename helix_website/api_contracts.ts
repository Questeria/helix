/**
 * Helix Website — TypeScript API Contracts
 *
 * Use these typed interfaces in the website's frontend. Stub the implementations
 * with mock data initially; swap in real backend calls later without changing
 * any UI code.
 *
 * The user (developer) will wire these up to a real backend (Python FastAPI
 * calling helixc) when ready. Until then, the UI consumes typed stubs.
 */

// ============================================================================
// Compile API — for /playground and /learn
// ============================================================================

export interface Token {
  /** Numeric token tag — see lexer.hx */
  tag: number;
  /** Token kind name, e.g. "INT", "IDENT", "FATARROW" */
  kindName: string;
  /** Source byte index where this token starts */
  srcStart: number;
  /** Source byte length */
  srcLen: number;
  /** Optional decoded payload (e.g. integer value, identifier text) */
  payload?: number | string;
}

export type AstTag =
  | 'INT' | 'VAR' | 'ADD' | 'SUB' | 'MUL' | 'DIV' | 'NEG'
  | 'IF' | 'WHILE' | 'LET' | 'LET_MUT' | 'ASSIGN' | 'SEQ'
  | 'FN_DECL' | 'FN_LIST' | 'CALL' | 'PARAM' | 'ARG'
  | 'EQ' | 'NE' | 'LT' | 'LE' | 'GT' | 'GE'
  | 'BAND' | 'BOR' | 'BXOR' | 'BNOT' | 'NOT' | 'SHL' | 'SHR'
  | 'STR_LIT' | 'INTLIT_I64' | 'INTLIT_U32' | 'INTLIT_U8' | 'INTLIT_U64'
  | 'INTLIT_I8' | 'INTLIT_U16' | 'INTLIT_I16'
  | 'FLOATLIT' | 'FLOATLIT_F64' | 'FLOATLIT_BF16'
  | 'TUPLE_LIT' | 'TUPLE_CONS' | 'TUPLE_FIELD' | 'INDEX'
  | 'STRUCT_DECL' | 'ENUM_DECL' | 'ENUM_CONSTRUCT'
  | 'MATCH' | 'MATCH_ARM' | 'PAT_LIT' | 'PAT_BIND' | 'PAT_WILDCARD'
  | 'PAT_RANGE' | 'PAT_OR' | 'PAT_VARIANT' | 'PAT_TUPLE'
  | 'GENERIC_PARAM' | 'TURBOFISH'
  | 'TRAIT_DECL' | 'IMPL_BLOCK' | 'TRAIT_BOUND'
  | 'CLOSURE_LIT' | 'CLOSURE_CALL'
  | 'MOD_DECL' | 'USE_DECL' | 'PATH_EXPR'
  | 'QUOTE' | 'SPLICE' | 'MODIFY' | 'VERIFIER_BLOCK'
  | 'AD_FORWARD' | 'AD_TANGENT_PAIR' | 'AD_REVERSE' | 'AD_ADJOINT_BUCKET'
  | 'AD_CHECKPOINT'
  | 'TILE_LIT' | 'TENSOR_LIT' | 'TILE_LOAD' | 'TILE_STORE' | 'TILE_MATMUL'
  | 'KERNEL_DECL' | 'PTX_BLOCK' | 'FFI_EXTERN_DECL'
  | 'ERR';

export interface AstNode {
  /** Node tag, e.g. "ADD", "FN_DECL" */
  tag: AstTag;
  /** Child nodes (length depends on tag) */
  children: AstNode[];
  /** Optional payload (integer literal value, identifier text, etc.) */
  payload?: number | string;
  /** Source span for error highlighting */
  src: { start: number; end: number };
}

export interface IrOp {
  /** Op kind, e.g. "ADD", "STORE", "CONST_INT" */
  kind: string;
  /** Operand IDs (indices into the op array) */
  operands: number[];
  /** Optional payload (constant value, etc.) */
  payload?: number | string;
  /** Result type tag */
  resultType?: string;
}

export interface CompileRequest {
  /** Helix source code */
  source: string;
  /** If set, use a canned example by name (faster than recompiling) */
  example?: string;
  /** Stages to include in the response (for selective rendering) */
  emit?: Array<'tokens' | 'ast' | 'ir' | 'bytes' | 'asm'>;
}

export interface CompileError {
  /** Error code, e.g. "E64001" */
  code: string;
  /** Human-readable message */
  message: string;
  /** Source span */
  span: { start: number; end: number };
  /** Optional did-you-mean suggestions */
  suggestions?: string[];
}

export interface CompileResult {
  ok: boolean;
  /** Tokens (if requested) */
  tokens?: Token[];
  /** AST root (if requested) */
  ast?: AstNode;
  /** IR ops (if requested) */
  ir?: IrOp[];
  /** Final x86-64 byte sequence (if requested) */
  bytes?: number[];
  /** Disassembled asm listing (if requested) */
  asm?: string;
  /** Errors (if !ok) */
  errors?: CompileError[];
  /** Compile time in ms */
  durationMs: number;
}

export interface RunRequest {
  /** Helix source */
  source: string;
  /** Stdin to provide */
  stdin?: string;
  /** Timeout in ms (default: 5000) */
  timeoutMs?: number;
}

export interface RunResult {
  ok: boolean;
  exitCode: number;
  stdout: string;
  stderr: string;
  durationMs: number;
}

export interface CompileApi {
  compile(req: CompileRequest): Promise<CompileResult>;
  run(req: RunRequest): Promise<RunResult>;
  /** List available canned examples */
  listExamples(): Promise<{ id: string; name: string; description: string }[]>;
}

// ============================================================================
// Stages API — for /roadmap
// ============================================================================

export interface StageInfo {
  /** Stage ID, e.g. "5", "8.5", "14.5" */
  id: string;
  /** Title */
  title: string;
  /** One-paragraph summary */
  summary: string;
  /** Status */
  status: 'done' | 'in-progress' | 'queued' | 'blocked';
  /** Difficulty (1-10) */
  difficulty: number;
  /** Estimated commits */
  estimatedCommits: number;
  /** Actual commits landed */
  actualCommits: number;
  /** Test count for this stage */
  testCount: number;
  /** Stage dependencies (IDs) */
  dependencies: string[];
  /** Sub-iterations (e.g. 5A, 5B, 5C, 5D) */
  iterations?: SubStage[];
  /** Recent commits */
  recentCommits?: GitCommit[];
}

export interface SubStage {
  id: string;
  title: string;
  status: 'done' | 'in-progress' | 'queued';
  commits: string[];
}

export interface GitCommit {
  hash: string;
  message: string;
  author: string;
  date: string;
  filesChanged: number;
  insertions: number;
  deletions: number;
}

export interface StagesApi {
  /** List all stages */
  listStages(): Promise<StageInfo[]>;
  /** Get a single stage with full detail */
  getStage(id: string): Promise<StageInfo>;
  /** Overall progress: percentage, stages done, in-flight */
  getProgress(): Promise<{
    percentDone: number;
    stagesDone: number;
    stagesTotal: number;
    inFlight: string[];
    blocked: string[];
  }>;
}

// ============================================================================
// Audits API — for /audits
// ============================================================================

export interface AuditFinding {
  /** Finding ID, e.g. 1, 2, 3, ..., 8 */
  id: number;
  /** Severity */
  severity: 'HIGH' | 'MEDIUM' | 'LOW';
  /** Category */
  category: 'silent-corruption' | 'safety' | 'soundness' | 'performance';
  /** Status */
  status: 'resolved' | 'deferred' | 'theoretical-only' | 'open';
  /** Title */
  title: string;
  /** AST tag affected */
  affectedAstTag: string;
  /** Source location (file:line range) */
  location: string;
  /** Description */
  description: string;
  /** Reproducer code */
  reproducer: string;
  /** Recommended fix description */
  recommendedFix: string;
  /** Trap-id reservations */
  trapIds: number[];
  /** Resolution commit hash (if resolved) */
  resolutionCommit?: string;
  /** Resolution date */
  resolutionDate?: string;
}

export interface AuditApi {
  listFindings(): Promise<AuditFinding[]>;
  getFinding(id: number): Promise<AuditFinding>;
  /** Stats: how many resolved, deferred, etc. */
  getStats(): Promise<{
    total: number;
    resolved: number;
    deferred: number;
    theoreticalOnly: number;
    open: number;
    bySeverity: Record<'HIGH' | 'MEDIUM' | 'LOW', number>;
  }>;
}

// ============================================================================
// Bootstrap Chain API — for /bootstrap-chain
// ============================================================================

export interface BootstrapStage {
  /** Stage name */
  name: 'hex0' | 'hex1' | 'M0' | 'M1' | 'M2-Planet' | 'kovc-bootstrap' | 'kovc';
  /** Display title */
  title: string;
  /** Binary size in bytes */
  sizeBytes: number;
  /** Source language (e.g. "x86-64 hex", "M0 macro asm", "ANSI C", "Helix") */
  sourceLanguage: string;
  /** What this stage can do */
  capability: string;
  /** Optional hex dump of the binary */
  hexDump?: string;
  /** Source code (for stages where it's small enough to display) */
  source?: string;
  /** Compiles which next stage */
  compiles: string;
}

export interface BootstrapApi {
  listStages(): Promise<BootstrapStage[]>;
  getStage(name: BootstrapStage['name']): Promise<BootstrapStage>;
}

// ============================================================================
// Examples API — canned snippets for /playground and /learn
// ============================================================================

export interface CodeExample {
  id: string;
  title: string;
  category: 'getting-started' | 'language-features' | 'ml' | 'reflection' | 'ffi';
  /** One-line description */
  description: string;
  /** Source code */
  source: string;
  /** Expected output (rendered alongside) */
  expectedOutput: string;
  /** Optional explanation */
  explanation?: string;
  /** Difficulty (1-5) */
  difficulty: number;
  /** Features demonstrated */
  features: string[];
}

export interface ExamplesApi {
  listExamples(filter?: {
    category?: CodeExample['category'];
    feature?: string;
  }): Promise<CodeExample[]>;
  getExample(id: string): Promise<CodeExample>;
}

// ============================================================================
// Stub implementations (drop into src/api/stubs.ts during MVP)
// ============================================================================

export const STUB_EXAMPLES: CodeExample[] = [
  {
    id: 'hello',
    title: 'Hello, 42',
    category: 'getting-started',
    description: 'The canonical first program.',
    source: 'fn main() -> i32 { 42 }',
    expectedOutput: '42',
    difficulty: 1,
    features: ['fn-decl', 'literal'],
  },
  {
    id: 'fib',
    title: 'Fibonacci',
    category: 'language-features',
    description: 'Recursive function with totality check.',
    source: `fn fib(n: i32) -> i32 {
    if n < 2 { n }
    else { fib(n - 1) + fib(n - 2) }
}

fn main() -> i32 { fib(10) }`,
    expectedOutput: '55',
    difficulty: 2,
    features: ['recursion', 'if-expr', 'totality'],
  },
  {
    id: 'autodiff',
    title: 'Forward-mode autodiff',
    category: 'ml',
    description: 'Symbolic derivative via grad(f)(x).',
    source: `fn loss(x: f64) -> f64 { x * x + 3.0_f64 * x }

fn main() -> f64 {
    grad(loss)(2.0_f64)
}`,
    expectedOutput: '7.0',
    explanation: "f(x) = x² + 3x, so f'(x) = 2x + 3, and f'(2) = 7.",
    difficulty: 3,
    features: ['autodiff', 'f64'],
  },
  // ... (full set in code_samples.md)
];

export const STUB_BOOTSTRAP_CHAIN: BootstrapStage[] = [
  {
    name: 'hex0',
    title: 'hex0 — the trust root',
    sizeBytes: 120,
    sourceLanguage: 'hand-encoded x86-64 hex',
    capability: 'Reads hex digits from stdin, writes byte values to stdout. That\'s it.',
    compiles: 'hex1',
  },
  {
    name: 'hex1',
    title: 'hex1 — labels and references',
    sizeBytes: 700,
    sourceLanguage: 'hex0 (with labels)',
    capability: 'Adds named labels and forward/back references to hex0 syntax.',
    compiles: 'M0',
  },
  {
    name: 'M0',
    title: 'M0 — minimal macro assembler',
    sizeBytes: 3000,
    sourceLanguage: 'hex1',
    capability: 'Macro definitions and substitution.',
    compiles: 'M1',
  },
  {
    name: 'M1',
    title: 'M1 — full macro assembler',
    sizeBytes: 8000,
    sourceLanguage: 'M0',
    capability: 'Directives, conditionals, includes.',
    compiles: 'M2-Planet',
  },
  {
    name: 'M2-Planet',
    title: 'M2-Planet — ANSI C compiler',
    sizeBytes: 30000,
    sourceLanguage: 'M1',
    capability: 'Compiles a useful subset of ANSI C.',
    compiles: 'kovc-bootstrap',
  },
  {
    name: 'kovc-bootstrap',
    title: 'kovc-bootstrap — Helix in C',
    sizeBytes: 80000,
    sourceLanguage: 'ANSI C',
    capability: 'Initial Helix compiler, written in C, compiled by M2-Planet.',
    compiles: 'kovc',
  },
  {
    name: 'kovc',
    title: 'kovc — self-hosted Helix',
    sizeBytes: 50000,
    sourceLanguage: 'Helix',
    capability: 'Self-hosting Helix compiler. Compiles itself byte-identically.',
    compiles: 'itself',
  },
];

export const STUB_AUDIT_FINDINGS: AuditFinding[] = [
  {
    id: 1,
    severity: 'HIGH',
    category: 'silent-corruption',
    status: 'resolved',
    title: 'AST_FN_DECL body-vs-ret-ty trap only checks 8-byte width',
    affectedAstTag: 'FN_DECL',
    location: 'helixc/bootstrap/kovc.hx:4537-4544',
    description: 'The post-body trap fires only when 8b/8b mismatches; narrow-vs-narrow mismatches escape silently.',
    reproducer: `fn f() -> u8 { 257 }
fn main() -> i32 {
    let x: u8 = f();   // 257 truncates to 1 silently
    x as i32
}`,
    recommendedFix: 'Replace 8b/8b check with full expr_type comparison.',
    trapIds: [14002],
    resolutionCommit: '6aaec01',
    resolutionDate: '2026-05-08',
  },
  // ... 7 more findings (full set in docs/audit-stage4-followup.md)
];

// ============================================================================
// Usage example for Claude Design
// ============================================================================

/*
import { CompileApi, STUB_EXAMPLES } from './api/contracts';
import { stubCompileApi } from './api/stubs';

// During MVP — use the stub
const api: CompileApi = stubCompileApi;

// Later, when backend is wired in:
// const api: CompileApi = realCompileApi('https://api.helix.dev');

const result = await api.compile({ source: 'fn main() -> i32 { 42 }' });
console.log(result.exitCode); // 42 (from stub)
*/
