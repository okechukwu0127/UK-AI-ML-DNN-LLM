const express = require("express");
const mysql = require("mysql2");
const morgan = require("morgan");
const { PythonShell } = require("python-shell");
const fs = require("fs");

function getPythonPath() {
  const venvPython = __dirname + "/.venv/bin/python3";
  if (fs.existsSync(venvPython)) {
    console.log({ venvPython });
    return venvPython;
  }
  // fallback to system python3
  console.log("Using system python3");
  return "python3";
}

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

function runPythonScript(scriptName, args = [], options = {}) {
  const defaultOptions = {
    mode: "text",
    pythonOptions: ["-u"],
    pythonPath: getPythonPath(),
    scriptPath: __dirname,
    args,
  };
  const mergedOptions = { ...defaultOptions, ...options };
  return new Promise((resolve, reject) => {
    let settled = false;
    const timeout = setTimeout(() => {
      if (!settled) {
        settled = true;
        reject(new Error("Python script timed out"));
      }
    }, 10000); // 10 seconds timeout
    PythonShell.run(scriptName, mergedOptions, function (err, results) {
      if (settled) return;
      settled = true;
      clearTimeout(timeout);
      console.log(
        `PythonShell results for ${scriptName}:`,
        results,
        "Error:",
        err
      );
      if (err) return reject(err);
      if (!results || results.length === 0) {
        return reject(new Error("No output from Python script"));
      }
      resolve(results);
    });
  });
}

async function detectInjectionAI(input) {
  try {
    const results = await runPythonScript("predict_lstm.py", [input], {
      scriptPath: undefined, // keep original behavior for predict_lstm.py
    });
    return parseInt(results[0]) === 1;
  } catch (err) {
    throw err;
  }
}

function detectSQLi(username) {
  const query = `SELECT * FROM users WHERE username = '${username}'`;
  return runPythonScript("predict.py", [query]).then((results) => {
    return parseInt(results[0]) === 1;
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

//http://localhost:3000/user-secure?username=admin
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
