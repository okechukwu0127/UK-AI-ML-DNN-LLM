const express = require("express");
const mysql = require("mysql2");
const morgan = require("morgan");
const { PythonShell } = require("python-shell");

const app = express();
const PORT = 3000;

app.use(morgan("combined")); // Logs for analytics

const db = mysql.createConnection({
  host: "localhost",
  user: "root",
  password: "password",
  database: "sqli",
});

db.connect((err) => {
  if (err) throw err;
  console.log("DB connected.");
});

async function detectInjectionAI(input) {
  return new Promise((resolve, reject) => {
    let options = {
      mode: "text",
      pythonOptions: ["-u"], // get print results in real-time
      pythonPath: "/Users/oeze/Documents/wlv/sqli/.venv/bin/python3", //  <-- ADD THIS LINE
      args: [input], // <-- Pass the actual query/username here!
    };

    PythonShell.run("predict_lstm.py", options, function (err, results) {
      //if (err) return reject(err);
      if (err) throw err;
      // results is an array consisting of messages collected
      console.log("results: %j", results);
      resolve(parseInt(results[0]) === 1);
    });

    /*  PythonShell.run("predict_lstm.py", { args: [input] }, (err, results) => {
      if (err) return reject(err);
      resolve(parseInt(results[0]) === 1);
    }); */
  });
}

app.get("/ai-detect", async (req, res) => {
  const input = req.query.username;
  try {
    const isAttack = await detectInjectionAI(input);
    if (isAttack) {
      console.log(`[BLOCKED] SQL Injection detected: ${input}`);
      return res.status(403).send("Blocked: Potential SQL Injection detected");
    }

    db.execute(
      "SELECT * FROM users WHERE username = ?",
      [input],
      (err, results) => {
        if (err) return res.status(500).send(err.sqlMessage);
        res.json(results);
      }
    );
  } catch (err) {
    res.status(500).send("Detection failed", { err });
  }
});

app.get("/user", (req, res) => {
  const username = req.query.username;
  let query = "";
  if (!username) {
    query = `SELECT * FROM users`;
  } else {
    query = `SELECT * FROM users WHERE username = '${username}'`;
  }

  db.query(query, (err, results) => {
    if (err) return res.status(500).send(err.sqlMessage);
    res.json(results);
  });
});

/////////////

function detectSQLi(username) {
  // Construct the SQL query string as your model was trained on
  const query = `SELECT * FROM users WHERE username = '${username}'`;

  let options = {
    mode: "text",
    pythonOptions: ["-u"],
    pythonPath: "/Users/oeze/Documents/wlv/sqli/.venv/bin/python3",
    scriptPath: "/Users/oeze/Documents/wlv/sqli", // Add this line!
    args: [query], // Pass the full SQL query
  };

  //console.log({ options, query });

  return new Promise((resolve, reject) => {
    PythonShell.run("predict.py", options, function (err, results) {
      console.log("PythonShell results:", results, "Error:", err);
      if (err) return reject(err);
      resolve(parseInt(results[0]) === 1);
    });
  });
}

app.get("/user-secure", async (req, res) => {
  const username = req.query.username;
  try {
    console.log({ username });
    const isAttack = await detectSQLi(username);
    console.log({ isAttack });
    if (isAttack)
      return res.status(403).send("Potential SQL injection blocked.");

    // Safe query (parametrized version)
    db.execute(
      "SELECT * FROM users WHERE username = ?",
      [username],
      (err, results) => {
        if (err) return res.status(500).send(err.sqlMessage);
        res.json(results);
      }
    );
  } catch (e) {
    res.status(500).send("Detection failed", { e });
  }
});

app.listen(PORT, () => console.log(`API running on http://localhost:${PORT}`));
