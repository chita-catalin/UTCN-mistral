const { MongoClient } = require("mongodb");
const fs = require("fs");

const uri = "mongodb://127.0.0.1:27017";
const client = new MongoClient(uri);

async function run() {
  try {
    await client.connect();
    const db = client.db("energy");
    const readings = db.collection("readings");

    const total = await readings.estimatedDocumentCount();
    console.log(`Exporting ${total} documents...`);

    const cursor = readings.find();
    const output = fs.createWriteStream("readings.json");
    output.write("[\n");

    let count = 0;
    for await (const doc of cursor) {
      const cleanDoc = {
        ...doc,
        _id: doc._id.toString(),
        ts: doc.ts ? doc.ts.toISOString() : null,
      };

      output.write(JSON.stringify(cleanDoc, null, 2));
      count++;

      if (count < total) output.write(",\n");
      else output.write("\n");

      // progress bar
      if (count % 100 === 0 || count === total) {
        const percent = ((count / total) * 100).toFixed(1);
        process.stdout.write(`\rProgress: ${count}/${total} (${percent}%)`);
      }
    }

    output.write("]\n");
    output.end();

    console.log("\n✅ Export finished.");
  } finally {
    await client.close();
  }
}

run().catch(console.error);
