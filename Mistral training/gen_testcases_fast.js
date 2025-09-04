#!/usr/bin/env node
/* gen_testcases_fast.js
 * Ultra-fast, append-only JSONL generator for NL→Mongo test cases.
 * - No database access. Everything is sampled synthetically but plausibly.
 * - Deterministic split filtering (train/val/test) via hash buckets.
 * - Minimal memory footprint: writes each example immediately.
 *
 * Usage:
 *   node gen_testcases_fast.js --out data/test.jsonl --n 1000 [options...]
 *
 * Useful options:
 *   --split train|val|test           Partition selector (default: none → accept all)
 *   --split-pcts "80,10,10"          Train/val/test percentages
 *   --split-key auto|meter|meter_day|meter_range   (default: auto)
 *   --split-salt "v1"                Changes hash buckets deterministically
 *   --date-range "2011-01-01,2014-12-31"
 *   --meter-min 1  --meter-max 400   Generates MAC000001..MAC000400
 *   --tariffs "Std,ToU,E7"
 *   --seed 42
 *   --no-validate                    Skip Ajv validation for speed
 */

"use strict";

const fs = require("fs");
const path = require("path");
const crypto = require("crypto");
const Ajv = require("ajv");
const addFormats = require("ajv-formats");

// ----------------------- CLI -----------------------
const args = process.argv.slice(2).reduce((acc, cur, i, arr) => {
  if (cur.startsWith("--")) {
    const k = cur.slice(2);
    const v = arr[i + 1] && !arr[i + 1].startsWith("--") ? arr[i + 1] : true;
    acc[k] = v;
  }
  return acc;
}, {});

const OUT_PATH   = args.out || "data/test.jsonl";
const N          = parseInt(args.n || "2000", 10);
const SEED       = args.seed ? parseInt(args.seed, 10) : 1337;

const DATE_RANGE = (args["date-range"] || "2011-01-01,2014-12-31").split(",");
const DATE_MIN   = new Date(DATE_RANGE[0] + "T00:00:00Z");
const DATE_MAX   = new Date(DATE_RANGE[1] + "T23:59:59Z");

const METER_MIN  = parseInt(args["meter-min"] || "1", 10);
const METER_MAX  = parseInt(args["meter-max"] || "400", 10); // pick a few hundred by default

const TARIFFS    = (args.tariffs || "Std,ToU,E7").split(",").map(s => s.trim()).filter(Boolean);

const VALIDATE   = !args["no-validate"];

// Split options
const SPLIT         = args.split || null; // train|val|test|null
const SPLIT_PCTS    = (args["split-pcts"] || "80,10,10").split(",").map(x => parseInt(x.trim(), 10));
const SPLIT_KEYMODE = (args["split-key"] || "auto");
const SPLIT_SALT    = (args["split-salt"] || "v1");

if (SPLIT_PCTS.length !== 3 || SPLIT_PCTS.some(isNaN) || SPLIT_PCTS.reduce((a,b)=>a+b,0) !== 100) {
  console.error('Error: --split-pcts must be "A,B,C" integers summing to 100. e.g., "80,10,10"');
  process.exit(1);
}
const ranges = (() => {
  const [tr, va, te] = SPLIT_PCTS;
  return {
    train: [0, tr - 1],
    val:   [tr, tr + va - 1],
    test:  [tr + va, tr + va + te - 1]
  };
})();
const inSplit = (bucket, split) => !split || (bucket >= ranges[split][0] && bucket <= ranges[split][1]);

// ----------------------- RNG (seeded) -----------------------
function mulberry32(a) {
  return function () {
    let t = (a += 0x6d2b79f5);
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}
const rand = mulberry32(SEED);
const randint = (min, max) => Math.floor(rand() * (max - min + 1)) + min;
const pick = (arr) => arr[Math.floor(rand() * arr.length)];

// ----------------------- Schema & validator -----------------------
const QuerySchema = {
  type: "object",
  additionalProperties: false,
  required: ["mongo", "postprocess"],
  properties: {
    mongo: {
      type: "object",
      additionalProperties: false,
      properties: {
        operation: { enum: ["find", "aggregate"] },
        filter: { type: "object" },
        projection: { type: "object" },
        sort: { type: "object" },
        limit: { type: "integer", minimum: 1, maximum: 10000 },
        pipeline: { type: "array", items: { type: "object" } }
      }
    },
    postprocess: {
      type: "object",
      additionalProperties: false,
      properties: {
        type: { enum: ["none", "summary", "timeseries", "topk"] },
        nlg: { type: "string" },
        fields: { type: "array", items: { type: "string" } }
      },
      required: ["type", "nlg", "fields"]
    }
  }
};

let validate = () => true;
if (VALIDATE) {
  const ajv = new Ajv({ allErrors: true, strict: false });
  addFormats(ajv);
  validate = ajv.compile(QuerySchema);
}

// ----------------------- System prompt -----------------------
const SYSTEM = `You translate natural-language questions about the LCL smart meter dataset (2011–2014, London) into MongoDB for db "energy", collection "readings".
Fields: meter_id (string), ts (ISODate), day (YYYY-MM-DD), kwh (number), tariff_type (string).
Always return ONE JSON object with keys "mongo" and "postprocess". No explanations, no prose.
Prefer day-based filters; when grouping by hour/day use Europe/London timezone.`;

// ----------------------- Utils -----------------------
const ensureDir = (p) => fs.mkdirSync(path.dirname(p), { recursive: true });

const pad = (n, w=6) => String(n).padStart(w, "0");
const meterId = (n) => `MAC${pad(n)}`;

function dayStr(d) {
  const y = d.getUTCFullYear();
  const m = String(d.getUTCMonth() + 1).padStart(2, "0");
  const dd = String(d.getUTCDate()).padStart(2, "0");
  return `${y}-${m}-${dd}`;
}
const isoStart = (day) => `${day}T00:00:00Z`;
const isoEnd   = (day) => `${day}T23:59:59Z`;

function randomDayWithin(minDate, maxDate) {
  const t = randint(minDate.getTime(), maxDate.getTime());
  const d = new Date(t);
  // normalize to UTC midnight
  return dayStr(new Date(Date.UTC(d.getUTCFullYear(), d.getUTCMonth(), d.getUTCDate())));
}
function addDays(day, k) {
  const d = new Date(day + "T00:00:00Z");
  const d2 = new Date(d.getTime() + k * 86400000);
  return dayStr(d2);
}
function orderedRange(dayA, dayB) {
  return dayA <= dayB ? [dayA, dayB] : [dayB, dayA];
}

function writeLine(stream, line) {
  return new Promise((resolve) => {
    if (stream.write(line + "\n")) return resolve();
    stream.once("drain", resolve);
  });
}

// Split hashing
function bucket100(key, salt) {
  const h = crypto.createHash("sha256").update(`${salt}|${key}`).digest("hex");
  const n = parseInt(h.slice(0, 8), 16);
  return n % 100;
}

// Decide split key per pattern
function makeSplitKey(mode, pattern, { meter_id, day, startDay, endDay }) {
  switch (mode) {
    case "meter":       return `${meter_id}`;
    case "meter_day":   return `${meter_id}|${day || startDay || endDay || "NA"}`;
    case "meter_range": return `${meter_id}|${startDay || day || "NA"}|${endDay || day || "NA"}`;
    case "auto":
    default:
      if (pattern === "latestN") return `${meter_id}`;
      if (pattern === "avgDailyInRange" || pattern === "topDaysInRange") return `${meter_id}|${startDay}|${endDay}`;
      return `${meter_id}|${day}`;
  }
}

// ----------------------- Paraphrase templates -----------------------
const T_daily = [
  d => `How much energy did ${d.meter} use on ${d.day}?`,
  d => `Total kWh for ${d.meter} on ${d.day}?`,
  d => `What was ${d.meter}'s consumption on ${d.day}?`,
  d => `Sum the kWh for ${d.meter} on ${d.day}.`
];
const T_hourly = [
  d => `Give me the hourly kWh for ${d.meter} on ${d.day}.`,
  d => `Break down ${d.meter} usage by hour on ${d.day}.`,
  d => `Hourly consumption time series for ${d.meter} on ${d.day}?`
];
const T_latest = [
  d => `Show the latest ${d.n} readings for ${d.meter}.`,
  d => `Fetch the most recent ${d.n} rows for ${d.meter}.`,
  d => `Return ${d.n} newest readings from ${d.meter}.`
];
const T_avgRange = [
  d => `What was the average daily kWh for ${d.meter} from ${d.start} to ${d.end}?`,
  d => `Between ${d.start} and ${d.end}, what's ${d.meter}'s mean per-day kWh?`
];
const T_topDays = [
  d => `Between ${d.start} and ${d.end}, what are the top ${d.k} days by total kWh for ${d.meter}?`,
  d => `Top ${d.k} highest-consumption days for ${d.meter} between ${d.start} and ${d.end}?`
];
const T_tariff = [
  d => `How much energy did ${d.meter} use on ${d.day} when tariff type was ${d.tariff}?`,
  d => `On ${d.day}, kWh for ${d.meter} under tariff ${d.tariff}?`
];

// ----------------------- Pattern generators -----------------------
function genDailyTotal(meter_id, day) {
  return {
    user: pick(T_daily)({ meter: meter_id, day }),
    assistantObj: {
      mongo: {
        operation: "aggregate",
        pipeline: [
          { $match: { meter_id, day } },
          { $group: { _id: null, total_kwh: { $sum: "$kwh" } } }
        ],
        limit: 1000
      },
      postprocess: {
        type: "summary",
        nlg: "{meter_id} used {total_kwh} kWh on {day}.",
        fields: ["total_kwh", "day", "meter_id"]
      }
    }
  };
}

function genHourlySeries(meter_id, day) {
  return {
    user: pick(T_hourly)({ meter: meter_id, day }),
    assistantObj: {
      mongo: {
        operation: "aggregate",
        pipeline: [
          { $match: { meter_id, day } },
          {
            $group: {
              _id: { $dateTrunc: { date: "$ts", unit: "hour", timezone: "Europe/London" } },
              kwh: { $sum: "$kwh" }
            }
          },
          { $sort: { _id: 1 } }
        ],
        limit: 1000
      },
      postprocess: {
        type: "timeseries",
        nlg: "Hourly kWh for {meter_id} on {day}.",
        fields: ["by_hour"]
      }
    }
  };
}

function genLatestN(meter_id) {
  const n = randint(3, 15);
  return {
    user: pick(T_latest)({ meter: meter_id, n }),
    assistantObj: {
      mongo: {
        operation: "find",
        filter: { meter_id },
        projection: { _id: false },
        sort: { ts: -1 },
        limit: n
      },
      postprocess: { type: "none", nlg: "", fields: [] }
    }
  };
}

function genAvgDailyInRange(meter_id, startDay, endDay) {
  const startISO = isoStart(startDay);
  const endISO   = isoEnd(endDay);
  return {
    user: pick(T_avgRange)({ meter: meter_id, start: startDay, end: endDay }),
    assistantObj: {
      mongo: {
        operation: "aggregate",
        pipeline: [
          { $match: { meter_id, ts: { $gte: { $date: startISO }, $lte: { $date: endISO } } } },
          {
            $group: {
              _id: { $dateTrunc: { date: "$ts", unit: "day", timezone: "Europe/London" } },
              daily_kwh: { $sum: "$kwh" }
            }
          },
          { $group: { _id: null, avg_daily_kwh: { $avg: "$daily_kwh" } } }
        ],
        limit: 1000
      },
      postprocess: {
        type: "summary",
        nlg: "Average daily kWh for {meter_id} from {start} to {end} was {avg_daily_kwh}.",
        fields: ["avg_daily_kwh", "start", "end", "meter_id"]
      }
    }
  };
}

function genTopDaysInRange(meter_id, startDay, endDay) {
  const k = randint(3, 7);
  const startISO = isoStart(startDay);
  const endISO   = isoEnd(endDay);
  return {
    user: pick(T_topDays)({ meter: meter_id, start: startDay, end: endDay, k }),
    assistantObj: {
      mongo: {
        operation: "aggregate",
        pipeline: [
          { $match: { meter_id, ts: { $gte: { $date: startISO }, $lte: { $date: endISO } } } },
          {
            $group: {
              _id: { $dateTrunc: { date: "$ts", unit: "day", timezone: "Europe/London" } },
              total_kwh: { $sum: "$kwh" }
            }
          },
          { $sort: { total_kwh: -1 } },
          { $limit: k }
        ],
        limit: 1000
      },
      postprocess: {
        type: "topk",
        nlg: "Top {k} days by total kWh for {meter_id} between {start} and {end}.",
        fields: ["top_days", "k", "start", "end", "meter_id"]
      }
    }
  };
}

function genByTariff(meter_id, day, tariff_type) {
  return {
    user: pick(T_tariff)({ meter: meter_id, day, tariff: tariff_type }),
    assistantObj: {
      mongo: {
        operation: "aggregate",
        pipeline: [
          { $match: { meter_id, day, tariff_type } },
          { $group: { _id: null, total_kwh: { $sum: "$kwh" } } }
        ],
        limit: 1000
      },
      postprocess: {
        type: "summary",
        nlg: "{meter_id} used {total_kwh} kWh on {day} with tariff {tariff_type}.",
        fields: ["total_kwh", "day", "meter_id", "tariff_type"]
      }
    }
  };
}

// Pattern pool (weights tuned for variety)
const PATTERNS = [
  { name: "dailyTotal",      weight: 28, gen: (m, d) => genDailyTotal(m, d) },
  { name: "hourlySeries",    weight: 22, gen: (m, d) => genHourlySeries(m, d) },
  { name: "latestN",         weight: 18, gen: (m)    => genLatestN(m) },
  { name: "avgDailyInRange", weight: 16, gen: (m,s,e)=> genAvgDailyInRange(m, s, e) },
  { name: "topDaysInRange",  weight: 10, gen: (m,s,e)=> genTopDaysInRange(m, s, e) },
  { name: "byTariff",        weight:  6, gen: (m,d,t)=> genByTariff(m, d, t) }
];

function weightedPickName(items) {
  const total = items.reduce((s, x) => s + x.weight, 0);
  let r = rand() * total;
  for (const it of items) { r -= it.weight; if (r <= 0) return it.name; }
  return items[items.length - 1].name;
}

// ----------------------- Main -----------------------
(async () => {
  ensureDir(OUT_PATH);
  const out = fs.createWriteStream(OUT_PATH, { flags: "a" });

  let written = 0;
  let attempts = 0;
  const MAX_ATTEMPTS = N * 5; // plenty, to account for split rejections

  while (written < N && attempts < MAX_ATTEMPTS) {
    attempts++;

    // Synthetic sample
    const meter_id = meterId(randint(METER_MIN, METER_MAX));
    const dayA = randomDayWithin(DATE_MIN, DATE_MAX);

    // 70% chance pick a short range (<=14 days), else arbitrary order
    const span = rand() < 0.7 ? randint(1, 14) : randint(1, 60);
    const dayB = addDays(dayA, span);
    const [startDay, endDay] = orderedRange(dayA, dayB);

    const tariff_type = pick(TARIFFS);

    const pattern = weightedPickName(PATTERNS);

    let example, splitKeyDetails = { meter_id, day: dayA, startDay, endDay };

    switch (pattern) {
      case "latestN":
        example = genLatestN(meter_id);
        splitKeyDetails = { meter_id };
        break;
      case "dailyTotal":
        example = genDailyTotal(meter_id, dayA);
        break;
      case "hourlySeries":
        example = genHourlySeries(meter_id, dayA);
        break;
      case "avgDailyInRange":
        example = genAvgDailyInRange(meter_id, startDay, endDay);
        break;
      case "topDaysInRange":
        example = genTopDaysInRange(meter_id, startDay, endDay);
        break;
      case "byTariff":
        example = genByTariff(meter_id, dayA, tariff_type);
        break;
      default:
        continue;
    }

    // Split filtering
    const splitKey = makeSplitKey(SPLIT_KEYMODE, pattern, splitKeyDetails);
    const bucket = bucket100(splitKey, SPLIT_SALT);
    if (!inSplit(bucket, SPLIT)) continue;

    // Validate (optional)
    if (!validate(example.assistantObj)) continue;

    const record = {
      messages: [
        { role: "system", content: SYSTEM },
        { role: "user", content: example.user },
        { role: "assistant", content: JSON.stringify(example.assistantObj) }
      ]
    };

    await writeLine(out, JSON.stringify(record));
    written++;

    if (written % 200 === 0) {
      console.log(`Appended ${written}/${N}... (split: ${SPLIT || "none"})`);
    }
  }

  await new Promise((r) => out.end(r));
  console.log(`Done. Appended ${written} examples → ${OUT_PATH}. (attempts=${attempts})`);
  if (SPLIT) {
    const [lo, hi] = ranges[SPLIT];
    console.log(`Split "${SPLIT}" uses buckets [${lo}..${hi}] with key mode "${SPLIT_KEYMODE}" and salt "${SPLIT_SALT}".`);
  }
})().catch((e) => {
  console.error(e);
  process.exit(1);
});
