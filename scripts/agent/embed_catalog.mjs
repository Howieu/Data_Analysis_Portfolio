// Precompute semantic vectors for every product (title, brand, category, stated facts, highlights, key specs).
// node --env-file=.env scripts/agent/embed_catalog.mjs   → data/embeddings.json
import { writeFileSync } from 'node:fs';
import { loadCatalog } from '../../lib/catalog.mjs';
import { productText, embed } from '../../lib/agent.mjs';

const catalog = loadCatalog(new URL('../../data', import.meta.url).pathname);
const ids = catalog.products.map(p => p.id), vectors = {};
for (let i = 0; i < ids.length; i += 32) {
  const batch = ids.slice(i, i + 32);
  const vs = await embed(batch.map(id => productText(catalog.byId.get(id), catalog)));
  if (!vs) throw new Error(`embedding failed at ${i}`);
  batch.forEach((id, k) => { vectors[id] = vs[k].map(x => Math.round(x * 1e4) / 1e4); });
  console.log(`${Math.min(i + 32, ids.length)}/${ids.length}`);
}
writeFileSync(new URL('../../data/embeddings.json', import.meta.url), JSON.stringify({ generatedAt: new Date().toISOString(), dims: 256, vectors }));
