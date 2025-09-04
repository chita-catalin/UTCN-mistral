// ingest.js — Fast CSV -> MongoDB import with tariff join, progress bars, concurrency, and reset-each-run.

import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { parse as parseStream } from 'csv-parse';
import { parse as parseSync } from 'csv-parse/sync';
import fg from 'fast-glob';
import { MongoClient } from 'mongodb';
import cliProgress from 'cli-progress';
import pLimit from 'p-limit';

// -------- Config (override with env if needed) --------
const ROOT         = (process.env.ROOT || process.cwd()).replace(/\\/g, '/');
const READINGS_GLOB= process.env.READINGS_GLOB || `${ROOT}/partitioned-data/LCL-*.csv`;
const TARIFFS_CSV  = process.env.TARIFFS_CSV   || `${ROOT}/tariffs.csv`;

const MONGO_URL    = process.env.MONGO_URL || 'mongodb://127.0.0.1:27017';
const DB_NAME      = process.env.DB_NAME   || 'energy';
const COLL_NAME    = process.env.COLL_NAME || 'readings';

const BATCH_SIZE   = Number(process.env.BATCH_SIZE   || 10000); // bigger batches → fewer roundtrips
const CONCURRENCY  = Number(process.env.CONCURRENCY  || Math.min(6, Math.max(2, Math.floor(os.cpus().length / 2))));
const RESET        = (process.env.RESET ?? 'true').toLowerCase() !== 'false'; // default true

// -------- Time helpers --------
function parseReadingTs(s) {
  if (!s) return null;
  const t = String(s).trim()
    .replace(/\.(\d{3})\d+$/, '.$1')      // keep ms
    .replace(/\.\d{4,}$/, m => m.slice(0, 4));
  const m = t.match(/^(\d{4})-(\d{2})-(\d{2})[ T](\d{2}):(\d{2}):(\d{2})(?:\.(\d{1,3}))?$/);
  if (!m) return null;
  const [_, yy, MM, dd, HH, mm, ss, ms = '0'] = m;
  const d = new Date(Date.UTC(+yy, +MM - 1, +dd, +HH, +mm, +ss, +ms));
  return Number.isNaN(d.getTime()) ? null : d;
}
function parseTariffTs(s) {
  if (s === undefined || s === null) return null;
  const str = String(s).trim();
  let m = str.match(/^(\d{1,2})\/(\d{1,2})\/(\d{4})[ T](\d{1,2}):(\d{2})$/); // M/D or D/M
  if (m) {
    let [_, a, b, yyyy, HH, mm] = m.map(Number);
    let MM = a, dd = b;
    if (dd > 12 && MM <= 12) { MM = b; dd = a; }
    const d = new Date(Date.UTC(yyyy, MM - 1, dd, HH, mm, 0, 0));
    return Number.isNaN(d.getTime()) ? null : d;
  }
  m = str.match(/^(\d{4})-(\d{2})-(\d{2})[ T](\d{2}):(\d{2})(?::(\d{2}))?$/); // ISO-ish
  if (m) {
    const [_, yy, MM, dd, HH, mm, ss = '0'] = m;
    const d = new Date(Date.UTC(+yy, +MM - 1, +dd, +HH, +mm, +ss));
    return Number.isNaN(d.getTime()) ? null : d;
  }
  return null;
}
const ymdUTC = (date) => date.toISOString().slice(0, 10);

// -------- Tariffs (CSV only) --------
function loadTariffsFromCsv(filePath) {
  if (!fs.existsSync(filePath)) {
    console.warn(`No tariffs CSV at: ${filePath}. Proceeding without tariffs.`);
    return { breaks: [], labelAt: new Map() };
  }
  const rows = parseSync(fs.readFileSync(filePath, 'utf8'), {
    columns: true, skip_empty_lines: true, trim: true,
  });
  if (!rows.length) return { breaks: [], labelAt: new Map() };

  const keys = Object.keys(rows[0]);
  const norm = s => String(s).toLowerCase().replace(/[^a-z0-9]/g, '');
  const byNorm = Object.fromEntries(keys.map(k => [norm(k), k]));
  const tsKey = byNorm['tariffdatetime'] || byNorm['datetime'] || byNorm['date'] || byNorm['time'] || keys[0];
  const labelKey = byNorm['tariff'] || byNorm['label'] || byNorm['name'] || keys[1];

  const labelAt = new Map();
  const breaks = [];
  for (const r of rows) {
    const ts = parseTariffTs(r[tsKey]);
    const label = (r[labelKey] ?? '').toString().trim();
    if (!ts || !label) continue;
    const ms = ts.getTime();
    if (!labelAt.has(ms)) { labelAt.set(ms, label); breaks.push(ms); }
  }
  breaks.sort((a, b) => a - b);
  return { breaks, labelAt };
}
function findTariffLabel(ts, { breaks, labelAt }) {
  if (!ts) return null;
  const ms = ts.getTime();
  if (labelAt.has(ms)) return labelAt.get(ms);
  let lo = 0, hi = breaks.length - 1, ans = -1; // last <= ms
  while (lo <= hi) {
    const mid = (lo + hi) >> 1;
    if (breaks[mid] <= ms) { ans = mid; lo = mid + 1; } else { hi = mid - 1; }
  }
  return ans >= 0 ? labelAt.get(breaks[ans]) : null;
}

// -------- Ingest one CSV (streamed) -> batched insertMany --------
async function ingestFile(file, col, tariffs, bars) {
  return new Promise((resolve, reject) => {
    const size = Math.max(1, fs.statSync(file).size);
    const filename = path.posix.basename(file);

    const fileBar = bars.create(size, 0, { filename, rows: 0 }, {
      format: `{bar} {percentage}% | {value}/{total}B | rows: {rows} | {filename}`,
    });

    let seen = 0;
    let inserted = 0;
    let batch = [];

    const rs = fs.createReadStream(file, { highWaterMark: 1 << 20 }); // 1MB chunks
    const stream = rs.pipe(parseStream({
      bom: true, columns: true, trim: true, skip_empty_lines: true, relax_column_count: true
    }));

    const flush = async () => {
      if (!batch.length) return;
      const docs = batch; batch = [];
      try {
        const res = await col.insertMany(docs, { ordered: false });
        inserted += (res.insertedCount ?? Object.keys(res.insertedIds ?? {}).length);
      } catch (e) {
        // If we ever switch back to upserts/unique indexes, ignore dup errors.
        if (e.code !== 11000) return reject(e);
      }
    };

    const tick = setInterval(() => fileBar.update(Math.min(rs.bytesRead, size), { rows: seen }), 100);

    stream.on('data', (rec) => {
      seen++;
      const meter_id    = String(rec['LCLid'] || rec['LCLID'] || rec['lclid'] || '').trim();
      const tariff_type = String(rec['stdorToU'] || rec['StdOrToU'] || '').trim();
      const dt_raw      = rec['DateTime'] || rec['datetime'] || '';
      const kwh_raw     = rec['KWH/hh (per half hour)'] ?? rec['KWH/hh'] ?? rec['kwh'] ?? '';

      const ts  = parseReadingTs(dt_raw);
      const kwh = Number(String(kwh_raw).trim());
      if (!meter_id || !ts || !Number.isFinite(kwh)) return;

      const day = ymdUTC(ts);
      const tariff_label = findTariffLabel(ts, tariffs);

      batch.push({ meter_id, ts, day, kwh, tariff_type, tariff_label });

      if (batch.length >= BATCH_SIZE) {
        stream.pause();
        flush().then(() => {
          fileBar.update(Math.min(rs.bytesRead, size), { rows: seen });
          stream.resume();
        }).catch(err => { clearInterval(tick); fileBar.stop(); reject(err); });
      }
    });

    stream.on('end', () => {
      clearInterval(tick);
      flush().then(() => {
        fileBar.update(size, { rows: seen });
        fileBar.stop();
        resolve({ rows: seen, inserted });
      }).catch(err => { fileBar.stop(); reject(err); });
    });

    stream.on('error', (err) => { clearInterval(tick); fileBar.stop(); reject(err); });
  });
}

// -------- Main --------
async function main() {
  console.log('ROOT:', ROOT);
  console.log('READINGS_GLOB:', READINGS_GLOB);
  console.log('TARIFFS_CSV:', TARIFFS_CSV);
  console.log('Saving to:', `${MONGO_URL} → db=${DB_NAME}, collection=${COLL_NAME}`);
  console.log(`Reset each run: ${RESET} | Concurrency: ${CONCURRENCY} | Batch: ${BATCH_SIZE}\n`);

  const files = await fg([READINGS_GLOB], { absolute: true, caseSensitiveMatch: false, windowsPathsNoEscape: true });
  if (!files.length) { console.error('No CSV files matched:', READINGS_GLOB); process.exit(1); }
  files.sort();

  const tariffs = loadTariffsFromCsv(TARIFFS_CSV);
  console.log(`Tariffs loaded: ${tariffs.breaks.length} timestamps\n`);

  const client = new MongoClient(MONGO_URL);
  await client.connect();
  const db = client.db(DB_NAME);
  const col = db.collection(COLL_NAME);

  // Reset collection for clean re-runs
  if (RESET) {
    await col.drop().catch(() => {});           // ignore if not exists
    // (re)create empty collection explicitly to avoid first-insert racing with index build
    await db.createCollection(COLL_NAME).catch(() => {});
  }

  // Progress bars
  const bars = new cliProgress.MultiBar({
    clearOnComplete: true, hideCursor: true,
  }, cliProgress.Presets.shades_classic);
  const filesBar = bars.create(files.length, 0, {}, { format: '{bar} {percentage}% | {value}/{total} files' });

  const limit = pLimit(CONCURRENCY);
  let totalRows = 0, totalInserted = 0;

  // Process files concurrently (bounded)
  await Promise.all(files.map(f => limit(async () => {
    const res = await ingestFile(f, col, tariffs, bars);
    totalRows += res.rows;
    totalInserted += res.inserted;
    filesBar.increment();
  })));

  bars.stop();

  // Build indexes AFTER loading (much faster)
  console.log('\nBuilding indexes…');
  await col.createIndex({ meter_id: 1, ts: 1 }, { unique: true });
  await col.createIndex({ day: 1 });
  await col.createIndex({ tariff_label: 1, day: 1 });

  await client.close();
  console.log(`\nDone. Rows seen: ${totalRows}. Inserted: ${totalInserted}.`);
  console.log(`Data saved to MongoDB → db: "${DB_NAME}", collection: "${COLL_NAME}".`);
}

// ----- robust "run-if-main" check (Windows-safe) -----
const thisFile = fileURLToPath(import.meta.url);
if (process.argv[1] && path.resolve(process.argv[1]) === path.resolve(thisFile)) {
  main().catch(err => { console.error(err); process.exit(1); });
}
