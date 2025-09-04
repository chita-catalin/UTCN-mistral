const { MongoClient } = require("mongodb");

const uri = "mongodb://127.0.0.1:27017"; // adjust if your Mongo runs elsewhere
const client = new MongoClient(uri);

async function run() {
  try {
    await client.connect();
    const db = client.db("energy");
    const readings = db.collection("readings");

    // Example: find one reading
    const doc = await readings.findOne();
    console.log("Sample doc:", doc);

    // Example: query for a specific meter_id
    const cursor = readings.find({ meter_id: "MAC000008" }).limit(5);
    const results = await cursor.toArray();
    console.log("Meter readings:", results);
  } catch (err) {
    console.error(err);
  } finally {
    await client.close();
  }
}

run();
