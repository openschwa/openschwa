// Generates src/lib/api/types.gen.ts from the committed contract schemas.
// Run via `npm run gen:types` (wired into `just schema` at the repo root).
//
// All contracts are compiled together rather than one file at a time: several
// of them share definitions (ModelInfo appears in both the health and model
// catalog responses), and compiling separately would emit a duplicate — and
// therefore conflicting — interface for each.
import { mkdirSync, readdirSync, readFileSync, writeFileSync } from 'node:fs';
import { dirname } from 'node:path';
import { fileURLToPath } from 'node:url';
import { compile } from 'json-schema-to-typescript';

const schemaDir = fileURLToPath(new URL('../../schemas/', import.meta.url));
const outPath = fileURLToPath(new URL('../src/lib/api/types.gen.ts', import.meta.url));

const files = readdirSync(schemaDir)
  .filter((name) => name.endsWith('.schema.json'))
  .sort();

/**
 * Pydantic gives every field a `title`, which the compiler turns into a named
 * alias per property — `Confidence1`, `StartS2`, and ninety more that nobody
 * reads. Model and definition titles are kept: those name the interfaces.
 */
function stripPropertyTitles(node) {
  if (Array.isArray(node)) {
    node.forEach(stripPropertyTitles);
    return;
  }
  if (!node || typeof node !== 'object') return;
  for (const property of Object.values(node.properties ?? {})) {
    if (property && typeof property === 'object') delete property.title;
  }
  Object.values(node).forEach(stripPropertyTitles);
}

/** Merged $defs across every contract, keyed by definition name. */
const defs = {};
/** Root model name -> its schema body, with $defs lifted out. */
const roots = {};

for (const file of files) {
  const schema = JSON.parse(readFileSync(schemaDir + file, 'utf8'));
  const { $schema, $id, $defs: fileDefs, ...body } = schema;
  stripPropertyTitles(body);

  for (const [name, def] of Object.entries(fileDefs ?? {})) {
    stripPropertyTitles(def);
    const existing = defs[name];
    if (existing && JSON.stringify(existing) !== JSON.stringify(def)) {
      throw new Error(
        `contract conflict: '${name}' is defined differently in ${file} than in another schema`,
      );
    }
    defs[name] = def;
  }
  roots[body.title] = body;
}

// One wrapper object so every root and every shared definition is emitted once.
// The wrapper interface itself is stripped from the output below.
const WRAPPER = 'OpenSchwaContracts';
const combined = {
  $schema: 'https://json-schema.org/draft/2020-12/schema',
  title: WRAPPER,
  type: 'object',
  additionalProperties: false,
  required: Object.keys(roots),
  properties: roots,
  $defs: defs,
};

const banner = [
  `/* Generated from schemas/*.schema.json — DO NOT EDIT.`,
  ` * Regenerate with \`just schema\` (or \`npm run gen:types\`).`,
  ` * Sources: ${files.join(', ')} */`,
  '',
  '',
].join('\n');

let ts = await compile(combined, WRAPPER, { bannerComment: '', additionalProperties: false });

// json-schema-to-typescript doesn't resolve prefixItems item types; every
// tuple in the v1 contract is a [number, number] interval/range.
ts = ts.replaceAll('[unknown, unknown]', '[number, number]');

// Drop the synthetic wrapper — it exists only to force single emission.
ts = ts.replace(new RegExp(`export interface ${WRAPPER} \\{[^}]*\\}\\n+`, 'm'), '');

mkdirSync(dirname(outPath), { recursive: true });
writeFileSync(outPath, banner + ts.trimStart());
console.log(`wrote ${outPath} from ${files.length} schemas`);
